#!/usr/bin/env python3
"""
Start or delete one RunPod Pod:

  - 1x H100 SXM-style GPU selection
  - Network volume: revolutionary_ivory_porcupine
  - Template: friendly_crimson_chimpanzee
  - Data center: automatically pinned to the network volume's dataCenterId
  - Non-interruptible / on-demand

Requirements:

  pip install requests
  export RUNPOD_API_KEY="..."

Run:

  python 08_hopper_gemm/runpod_script.py
  python 08_hopper_gemm/runpod_script.py kill
"""

import json
from os import environ
import pathlib
import sys
import time
import typing
import urllib.parse as urllib_parse

import requests

SCRIPT_FILE = pathlib.Path(__file__)
SCRIPT_PATH = SCRIPT_FILE.resolve()
LESSON_DIR = SCRIPT_PATH.parent
REPO_ROOT = LESSON_DIR.parent

BASE_URL = "https://rest.runpod.io/v1"

NETWORK_VOLUME_NAME = "revolutionary_ivory_porcupine"
TEMPLATE_NAME = "friendly_crimson_chimpanzee"

POD_NAME = "h100-revolutionary-ivory"
GPU_TYPE_ID = "NVIDIA H100 80GB HBM3"
GPU_COUNT = 1

WAIT_TIMEOUT_MINUTES = 20
WAIT_TIMEOUT_SECONDS = WAIT_TIMEOUT_MINUTES * 60
POLL_SECONDS = 10

SSH_CONFIG_PATH = "~/.ssh/config"
SSH_HOST_ALIAS = "runpod-h100"
SSH_USER = "root"
SSH_IDENTITY_FILE = "~/.ssh/id_ed25519"

RUNPOD_STATE_DIR = REPO_ROOT / ".runpod"
RUNPOD_STATE_PATH = RUNPOD_STATE_DIR / "h100-pod.json"


class RunPodAPIError(RuntimeError):
    def __init__(self, method: str, path: str, status_code: int, detail: typing.Any):
        self.method = method
        self.path = path
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{method} {path} -> HTTP {status_code}: {detail}")


def runpod_request(
    method: str,
    path: str,
    api_key: str,
    *,
    json_body: dict[str, typing.Any] | None = None,
    timeout: int = 60,
) -> typing.Any:
    """Send one authenticated request to the RunPod REST API."""
    headers = {"Authorization": f"Bearer {api_key}"}

    if json_body is not None:
        headers["Content-Type"] = "application/json"

    request_url = f"{BASE_URL}{path}"
    response = requests.request(
        method, request_url, headers=headers, json=json_body, timeout=timeout
    )
    status_code = response.status_code

    if status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text

        raise RunPodAPIError(method, path, status_code, detail)

    response_text = response.text

    if not response_text:
        return None

    try:
        return response.json()
    except ValueError:
        return response.text


def print_create_pod_error(*, error: RunPodAPIError, data_center_id: typing.Any) -> bool:
    """Print a friendly create-Pod error when the failure is expected."""
    if not _is_no_capacity_error(error):
        return False

    print(
        f"No H100s available in RunPod's {data_center_id} right now. Try again later.",
        file=sys.stderr,
    )
    return True


def resolve_network_volume(
    api_key: str, volume_name_or_id: str
) -> dict[str, typing.Any]:
    """
    Resolve by exact network volume ID or exact network volume name.

    Network volume names are not necessarily unique, so this raises if more
    than one matching volume is found.
    """
    volumes = runpod_request("GET", "/networkvolumes", api_key)

    if not isinstance(volumes, list):
        raise RuntimeError(f"Unexpected /networkvolumes response: {volumes!r}")

    matches = []

    for volume in volumes:
        volume_id = volume.get("id")
        volume_name = volume.get("name")

        if volume_id == volume_name_or_id or volume_name == volume_name_or_id:
            matches.append(volume)

    if not matches:
        raise RuntimeError(
            f"No network volume found with ID or exact name: {volume_name_or_id!r}"
        )

    if len(matches) > 1:
        choice_lines = []

        for volume in matches:
            volume_id = volume.get("id")
            volume_name = volume.get("name")
            data_center_id = volume.get("dataCenterId")
            choice = (
                f"  id={volume_id} name={volume_name} " f"dataCenterId={data_center_id}"
            )
            choice_lines.append(choice)

        choices = "\n".join(choice_lines)
        raise RuntimeError(
            "More than one network volume matched. Use the network volume ID instead:\n"
            f"{choices}"
        )

    matched_volume = matches[0]
    return matched_volume


def resolve_template(api_key: str, template_name_or_id: str) -> dict[str, typing.Any]:
    """
    Resolve by exact template ID or exact template name.

    The Pod creation API wants templateId, so this converts the UI template
    name into the actual template ID.
    """
    templates = runpod_request("GET", "/templates", api_key)

    if not isinstance(templates, list):
        raise RuntimeError(f"Unexpected /templates response: {templates!r}")

    matches = []

    for template in templates:
        template_id = template.get("id")
        template_name = template.get("name")

        if template_id == template_name_or_id or template_name == template_name_or_id:
            matches.append(template)

    if not matches:
        raise RuntimeError(
            f"No template found with ID or exact name: {template_name_or_id!r}"
        )

    if len(matches) > 1:
        choice_lines = []

        for template in matches:
            template_id = template.get("id")
            template_name = template.get("name")
            image_name = template.get("imageName")
            choice = f"  id={template_id} name={template_name} image={image_name}"
            choice_lines.append(choice)

        choices = "\n".join(choice_lines)
        raise RuntimeError(
            "More than one template matched. Use the template ID instead:\n" f"{choices}"
        )

    template = matches[0]
    is_serverless = template.get("isServerless")

    if is_serverless is True:
        raise RuntimeError(
            f"Template {template_name_or_id!r} is marked as serverless; this script "
            "needs a Pod template."
        )

    return template


def build_create_pod_payload(
    *, volume: dict[str, typing.Any], template: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    """Build the RunPod create-Pod payload from resolved resources."""
    data_center_id = volume["dataCenterId"]
    network_volume_id = volume["id"]
    template_id = template["id"]
    template_volume_mount_path = template.get("volumeMountPath")
    volume_mount_path = template_volume_mount_path or "/workspace"

    payload = {
        "name": POD_NAME,
        "cloudType": "SECURE",
        "computeType": "GPU",
        # 1x H100 SXM-style selection.
        "gpuTypeIds": [GPU_TYPE_ID],
        "gpuTypePriority": "custom",
        "gpuCount": GPU_COUNT,
        # Force the Pod into the same data center as the network volume.
        "dataCenterIds": [data_center_id],
        "dataCenterPriority": "custom",
        # Create from the template.
        # Do not send imageName or dockerStartCmd here; let the template define them.
        "templateId": template_id,
        # Attach the existing network volume.
        "networkVolumeId": network_volume_id,
        "volumeMountPath": volume_mount_path,
        # Explicitly non-interruptible / on-demand.
        "interruptible": False,
        # Secure Cloud always gets a public IP, but keeping this explicit is fine.
        "supportPublicIp": True,
    }

    return payload


def summarize_network_volume(volume: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Return the network volume fields worth printing for this lesson."""
    volume_id = volume.get("id")
    volume_name = volume.get("name")
    data_center_id = volume.get("dataCenterId")
    volume_size = volume.get("size")

    return {
        "id": volume_id,
        "name": volume_name,
        "dataCenterId": data_center_id,
        "size": volume_size,
    }


def summarize_template(template: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Return the template fields worth printing for this lesson."""
    template_id = template.get("id")
    template_name = template.get("name")
    image_name = template.get("imageName")
    docker_start_cmd = template.get("dockerStartCmd")
    volume_mount_path = template.get("volumeMountPath")
    ports = template.get("ports")
    is_serverless = template.get("isServerless")

    return {
        "id": template_id,
        "name": template_name,
        "imageName": image_name,
        "dockerStartCmd": docker_start_cmd,
        "volumeMountPath": volume_mount_path,
        "ports": ports,
        "isServerless": is_serverless,
    }


def summarize_pod(pod: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Return the Pod fields worth printing before and after startup."""
    gpu = pod.get("gpu")
    machine = pod.get("machine")
    network_volume = pod.get("networkVolume")

    if not isinstance(gpu, dict):
        gpu = {}

    if not isinstance(machine, dict):
        machine = {}

    if not isinstance(network_volume, dict):
        network_volume = {}

    pod_id = pod.get("id")
    pod_name = pod.get("name")
    desired_status = pod.get("desiredStatus")
    image = pod.get("image")
    template_id = pod.get("templateId")
    gpu_id = gpu.get("id")
    gpu_display_name = gpu.get("displayName")
    gpu_count = gpu.get("count")
    machine_id = pod.get("machineId")
    data_center_id = machine.get("dataCenterId")
    public_ip = pod.get("publicIp")
    ports = pod.get("ports")
    port_mappings = pod.get("portMappings")
    interruptible = pod.get("interruptible")
    network_volume_id = network_volume.get("id")
    network_volume_name = network_volume.get("name")
    network_volume_data_center_id = network_volume.get("dataCenterId")
    network_volume_size = network_volume.get("size")

    return {
        "pod_id": pod_id,
        "name": pod_name,
        "desiredStatus": desired_status,
        "image": image,
        "templateId": template_id,
        "gpu": {"id": gpu_id, "displayName": gpu_display_name, "count": gpu_count},
        "machineId": machine_id,
        "dataCenterId": data_center_id,
        "publicIp": public_ip,
        "ports": ports,
        "portMappings": port_mappings,
        "interruptible": interruptible,
        "networkVolume": {
            "id": network_volume_id,
            "name": network_volume_name,
            "dataCenterId": network_volume_data_center_id,
            "size": network_volume_size,
        },
    }


def _wait_for_connect_info(
    *, api_key: str, pod_id: str, timeout_seconds: int
) -> dict[str, typing.Any]:
    """Poll one Pod until RunPod exposes SSH connection details."""
    deadline = time.time() + timeout_seconds
    last_pod: dict[str, typing.Any] = {}
    last_error: RuntimeError | None = None
    quoted_pod_id = urllib_parse.quote(pod_id, safe="")
    pod_path = f"/pods/{quoted_pod_id}"

    while time.time() < deadline:
        pod = runpod_request("GET", pod_path, api_key)

        if isinstance(pod, dict):
            last_pod = pod

            try:
                _extract_ssh_connect_info(pod)
                return pod
            except RuntimeError as error:
                last_error = error

        time.sleep(POLL_SECONDS)

    last_pod_summary = summarize_pod(last_pod)
    last_pod_json = json.dumps(last_pod_summary, indent=2)

    raise RuntimeError(
        f"Timed out after {timeout_seconds} seconds waiting for SSH connect info.\n"
        f"Last parse error: {last_error}\n"
        f"Last Pod summary:\n{last_pod_json}"
    )


def _extract_ssh_connect_info(pod: dict[str, typing.Any]) -> tuple[str, int]:
    """Extract the public IP and public SSH port from RunPod Pod data."""
    pod_public_ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings")
    runtime = pod.get("runtime")
    public_ip = None

    if isinstance(pod_public_ip, str) and pod_public_ip:
        public_ip = pod_public_ip

    if not public_ip:
        public_ip = _extract_public_ip_from_port_data(port_mappings)

    if not public_ip and isinstance(runtime, dict):
        runtime_ports = runtime.get("ports")
        public_ip = _extract_public_ip_from_port_data(runtime_ports)

    ssh_port = _extract_ssh_port(port_mappings)

    if ssh_port is None and isinstance(runtime, dict):
        runtime_ports = runtime.get("ports")
        ssh_port = _extract_ssh_port(runtime_ports)

    if not public_ip or ssh_port is None:
        raise RuntimeError(
            f"RunPod did not return complete SSH connect info yet. "
            f"publicIp={public_ip!r}, ssh_port={ssh_port!r}"
        )

    return public_ip, ssh_port


def update_ssh_config_file(*, public_ip: str, ssh_port: int) -> pathlib.Path:
    """Create or update the SSH config entry for the RunPod alias."""
    ssh_config_path = _get_ssh_config_path()
    lines = _read_ssh_config_lines(ssh_config_path)
    host_block = _find_host_block(lines=lines, alias=SSH_HOST_ALIAS)

    if host_block is not None:
        block_start, block_end = host_block
        block_lines = lines[block_start:block_end]
        updated_block_lines = _update_host_block_lines(
            block_lines=block_lines, public_ip=public_ip, ssh_port=ssh_port
        )

        lines_before_block = lines[:block_start]
        lines_after_block = lines[block_end:]
        updated_lines = lines_before_block + updated_block_lines + lines_after_block
        _write_ssh_config_lines(ssh_config_path, updated_lines)

        return ssh_config_path

    if lines:
        last_line = lines[-1]
        stripped_last_line = last_line.strip()

        if stripped_last_line:
            lines.append("\n")

    new_host_block_lines = _build_new_host_block(public_ip=public_ip, ssh_port=ssh_port)
    lines.extend(new_host_block_lines)
    _write_ssh_config_lines(ssh_config_path, lines)

    return ssh_config_path


def _read_host_block_text(ssh_config_path: pathlib.Path, alias: str) -> str:
    """Read one SSH Host block from disk for terminal output."""
    lines = _read_ssh_config_lines(ssh_config_path)
    host_block = _find_host_block(lines=lines, alias=alias)

    if host_block is None:
        raise RuntimeError(f"Could not find SSH config Host block for {alias!r}")

    block_start, block_end = host_block
    block_lines = lines[block_start:block_end]

    return _render_host_block_text(block_lines)


def configure_ssh_for_pod(*, api_key: str, pod_id: str) -> dict[str, typing.Any]:
    """Wait for one Pod's SSH details and update the local SSH config."""
    print(
        f"Waiting up to {WAIT_TIMEOUT_MINUTES} minutes for Pod {pod_id} connect "
        "info...",
        file=sys.stderr,
    )
    pod = _wait_for_connect_info(
        api_key=api_key, pod_id=pod_id, timeout_seconds=WAIT_TIMEOUT_SECONDS
    )

    public_ip, ssh_port = _extract_ssh_connect_info(pod)
    ssh_config_path = update_ssh_config_file(public_ip=public_ip, ssh_port=ssh_port)
    ssh_config_entry = _read_host_block_text(ssh_config_path, SSH_HOST_ALIAS)

    print(f"Updated SSH config entry in {ssh_config_path}:", file=sys.stderr)
    print(ssh_config_entry, file=sys.stderr)
    print(f"\nConnect with: ssh {SSH_HOST_ALIAS}", file=sys.stderr)

    final_pod_summary = summarize_pod(pod)
    final_pod_json = json.dumps(final_pod_summary, indent=2)
    print(final_pod_json)

    return pod


def write_pod_state(pod: dict[str, typing.Any]) -> pathlib.Path:
    """Save the last Pod created by this script so `kill` can delete it later."""
    pod_id = pod.get("id")
    pod_name = pod.get("name")
    desired_status = pod.get("desiredStatus")
    template_id = pod.get("templateId")
    network_volume = pod.get("networkVolume")

    if not isinstance(pod_id, str):
        raise RuntimeError(f"Cannot save Pod state without string id: {pod!r}")

    state = {
        "pod_id": pod_id,
        "name": pod_name,
        "desiredStatus": desired_status,
        "templateId": template_id,
        "networkVolume": network_volume,
    }

    state_path = _get_runpod_state_path()
    state_dir = state_path.parent
    state_dir.mkdir(parents=True, exist_ok=True)

    state_text = json.dumps(state, indent=2)
    state_path.write_text(f"{state_text}\n")

    try:
        state_path.chmod(0o600)
    except OSError:
        pass

    return state_path


def kill_saved_pod(*, api_key: str) -> int:
    """Kill the last RunPod Pod created by this script."""
    pod_id = _read_saved_pod_id()

    print(f"Deleting RunPod Pod {pod_id}...", file=sys.stderr)

    delete_response = _delete_pod(api_key=api_key, pod_id=pod_id)
    delete_response_json = json.dumps(delete_response, indent=2)

    print("Delete response:", file=sys.stderr)
    print(delete_response_json, file=sys.stderr)

    state_path = _get_runpod_state_path()
    _clear_saved_pod_state()

    print(f"Cleared saved Pod state at {state_path}", file=sys.stderr)
    print(f"Deleted RunPod Pod {pod_id}.", file=sys.stderr)
    return 0


def _read_saved_pod_id() -> str:
    """Read the last Pod ID created by this script."""
    state_path = _get_runpod_state_path()

    if not state_path.exists():
        raise RuntimeError(
            f"No saved RunPod Pod state found at {state_path}.\nStart a Pod "
            "with this script before running kill."
        )

    state_text = state_path.read_text()
    state = json.loads(state_text)

    if not isinstance(state, dict):
        raise RuntimeError(f"Saved RunPod state was not a JSON object: {state!r}")

    pod_id = state.get("pod_id")

    if not isinstance(pod_id, str) or not pod_id:
        raise RuntimeError(
            f"Saved RunPod state did not contain a valid pod_id: {state!r}"
        )

    return pod_id


def _parse_port(value: typing.Any) -> int | None:
    if isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_ssh_container_port(value: typing.Any) -> bool:
    text = str(value)
    stripped_text = text.strip()
    port_text = stripped_text.lower()

    return port_text in {"22", "22/tcp", "tcp/22"}


def _mapping_targets_ssh(mapping: dict[str, typing.Any]) -> bool:
    for key in ("containerPort", "privatePort", "internalPort", "targetPort"):
        value = mapping.get(key)

        if _is_ssh_container_port(value):
            return True

    for key in ("port", "name"):
        value = mapping.get(key)

        if _is_ssh_container_port(value):
            return True

    return False


def _mapping_protocol_is_tcp(mapping: dict[str, typing.Any]) -> bool:
    for key in ("protocol", "type"):
        value = mapping.get(key)

        if value is None:
            continue

        text = str(value)
        stripped_text = text.strip()
        protocol = stripped_text.lower()

        if protocol != "tcp":
            return False

    return True


def _extract_public_port_from_mapping(mapping: dict[str, typing.Any]) -> int | None:
    for key in ("hostPort", "publicPort", "externalPort", "mappedPort"):
        value = mapping.get(key)
        port = _parse_port(value)

        if port is not None:
            return port

    # Some APIs use "port" for the externally exposed port while using "privatePort" or
    # "containerPort" for the container-side port.
    port_value = mapping.get("port")
    port = _parse_port(port_value)

    if port is not None and port != 22:
        return port

    return None


def _extract_public_port_from_value(value: typing.Any) -> int | None:
    port = _parse_port(value)

    if port is not None:
        return port

    if isinstance(value, dict):
        port = _extract_public_port_from_mapping(value)

        if port is not None:
            return port

        return _extract_ssh_port(value)

    if isinstance(value, list):
        for item in value:
            port = _extract_public_port_from_value(item)

            if port is not None:
                return port

    return None


def _extract_ssh_port(port_data: typing.Any) -> int | None:
    if isinstance(port_data, dict):
        if _mapping_targets_ssh(port_data) and _mapping_protocol_is_tcp(port_data):
            port = _extract_public_port_from_mapping(port_data)

            if port is not None:
                return port

        for key, value in port_data.items():
            if _is_ssh_container_port(key):
                port = _extract_public_port_from_value(value)

                if port is not None:
                    return port

        for value in port_data.values():
            port = _extract_ssh_port(value)

            if port is not None:
                return port

    if isinstance(port_data, list):
        for item in port_data:
            port = _extract_ssh_port(item)

            if port is not None:
                return port

    return None


def _extract_public_ip_from_port_data(port_data: typing.Any) -> str | None:
    if isinstance(port_data, dict):
        if _mapping_targets_ssh(port_data):
            for key in ("ip", "publicIp", "hostIp", "host"):
                value = port_data.get(key)

                if isinstance(value, str) and value:
                    return value

        for value in port_data.values():
            public_ip = _extract_public_ip_from_port_data(value)

            if public_ip:
                return public_ip

    if isinstance(port_data, list):
        for item in port_data:
            public_ip = _extract_public_ip_from_port_data(item)

            if public_ip:
                return public_ip

    return None


def _ssh_config_line_directive(line: str) -> str | None:
    stripped = line.strip()

    if not stripped or stripped.startswith("#"):
        return None

    parts = stripped.split(None, 1)

    if not parts:
        return None

    directive = parts[0]
    return directive.lower()


def _is_host_line(line: str) -> bool:
    directive = _ssh_config_line_directive(line)
    return directive == "host"


def _host_line_matches_alias(line: str, alias: str) -> bool:
    stripped = line.strip()

    if not stripped or stripped.startswith("#"):
        return False

    parts = stripped.split()

    if not parts:
        return False

    directive = parts[0]
    directive_name = directive.lower()

    if directive_name != "host":
        return False

    aliases = parts[1:]
    return alias in aliases


def _with_newline(line: str) -> str:
    if line.endswith("\n"):
        return line

    return f"{line}\n"


def _update_host_block_lines(
    *, block_lines: list[str], public_ip: str, ssh_port: int
) -> list[str]:
    """Update the managed directives inside one SSH Host block."""
    replacements = {
        "hostname": ("HostName", public_ip),
        "user": ("User", SSH_USER),
        "port": ("Port", str(ssh_port)),
        "identityfile": ("IdentityFile", SSH_IDENTITY_FILE),
        "serveraliveinterval": ("ServerAliveInterval", "60"),
        "serveralivecountmax": ("ServerAliveCountMax", "120"),
        "forwardagent": ("ForwardAgent", "yes"),
    }

    updated_lines = []
    seen = set()

    for line in block_lines:
        directive = _ssh_config_line_directive(line)

        if directive in replacements:
            name, value = replacements[directive]
            left_stripped_line = line.lstrip()
            indent_length = len(line) - len(left_stripped_line)
            indent = line[:indent_length] or "  "

            updated_lines.append(f"{indent}{name} {value}\n")
            seen.add(directive)
        else:
            updated_lines.append(_with_newline(line))

    for directive, replacement in replacements.items():
        if directive in seen:
            continue

        name, value = replacement
        updated_lines.append(f"  {name} {value}\n")

    return updated_lines


def _build_new_host_block(public_ip: str, ssh_port: int) -> list[str]:
    """Build a new SSH Host block for the RunPod alias."""
    return [
        f"Host {SSH_HOST_ALIAS}\n",
        f"  HostName {public_ip}\n",
        f"  User {SSH_USER}\n",
        f"  Port {ssh_port}\n",
        f"  IdentityFile {SSH_IDENTITY_FILE}\n",
        "  ServerAliveInterval 60\n",
        "  ServerAliveCountMax 120\n",
        "  ForwardAgent yes\n",
    ]


def _get_ssh_config_path() -> pathlib.Path:
    """Resolve the configured SSH config path."""
    configured_path = pathlib.Path(SSH_CONFIG_PATH)
    return configured_path.expanduser()


def _read_ssh_config_lines(ssh_config_path: pathlib.Path) -> list[str]:
    """Read SSH config lines and normalize the final newline."""
    if ssh_config_path.exists():
        config_text = ssh_config_path.read_text()
        lines = config_text.splitlines(keepends=True)
    else:
        lines = []

    if lines:
        last_line = lines[-1]

        if not last_line.endswith("\n"):
            lines[-1] = f"{last_line}\n"

    return lines


def _write_ssh_config_lines(ssh_config_path: pathlib.Path, lines: list[str]) -> None:
    """Persist SSH config lines and keep the config file private."""
    ssh_config_dir = ssh_config_path.parent
    ssh_config_dir.mkdir(parents=True, exist_ok=True)

    config_text = "".join(lines)
    ssh_config_path.write_text(config_text)

    try:
        ssh_config_path.chmod(0o600)
    except OSError:
        pass


def _find_host_block(*, lines: list[str], alias: str) -> tuple[int, int] | None:
    """Find the line range for one SSH Host block."""
    index = 0

    while index < len(lines):
        line = lines[index]

        if not _host_line_matches_alias(line, alias):
            index += 1
            continue

        block_start = index
        block_end = index + 1

        while block_end < len(lines):
            block_line = lines[block_end]

            if _is_host_line(block_line):
                break

            block_end += 1

        return block_start, block_end

    return None


def _render_host_block_text(block_lines: list[str]) -> str:
    """Render one SSH Host block for terminal output."""
    host_block_text = "".join(block_lines)
    return host_block_text.rstrip()


def _get_runpod_state_path() -> pathlib.Path:
    """Resolve the path used to remember the last created Pod."""
    return RUNPOD_STATE_PATH


def _delete_pod(*, api_key: str, pod_id: str) -> typing.Any:
    """Terminate/delete one RunPod Pod."""
    quoted_pod_id = urllib_parse.quote(pod_id, safe="")
    pod_path = f"/pods/{quoted_pod_id}"

    return runpod_request("DELETE", pod_path, api_key)


def _clear_saved_pod_state() -> None:
    """Remove the saved Pod state after the Pod has been deleted."""
    state_path = _get_runpod_state_path()

    try:
        state_path.unlink()
    except FileNotFoundError:
        pass


def _runpod_error_detail_text(error: RunPodAPIError) -> str:
    """Return the most useful text from a RunPod API error detail."""
    detail = error.detail

    if isinstance(detail, dict):
        error_message = detail.get("error")

        if isinstance(error_message, str):
            return error_message

        return json.dumps(detail)

    if isinstance(detail, str):
        return detail

    return repr(detail)


def _is_no_capacity_error(error: RunPodAPIError) -> bool:
    """Return whether RunPod rejected Pod creation because capacity is unavailable."""
    if error.method != "POST":
        return False

    if error.path != "/pods":
        return False

    if error.status_code != 500:
        return False

    detail_text = _runpod_error_detail_text(error)
    detail_text = detail_text.lower()
    has_no_instances = "no instances" in detail_text
    has_available = "available" in detail_text

    return has_no_instances and has_available


def main() -> int:
    api_key = environ.get("RUNPOD_API_KEY")

    if not api_key:
        print("Set RUNPOD_API_KEY in your environment.", file=sys.stderr)
        return 2

    arguments = sys.argv[1:]
    argument_count = len(arguments)
    usage = "Usage: python 08_hopper_gemm/runpod_script.py [kill]"

    if argument_count:
        command = arguments[0]

        if command != "kill":
            print(f"Unknown command: {command}", file=sys.stderr)
            print(usage, file=sys.stderr)
            return 2

        if argument_count > 1:
            print(usage, file=sys.stderr)
            return 2

        return kill_saved_pod(api_key=api_key)

    volume = resolve_network_volume(api_key, NETWORK_VOLUME_NAME)
    template = resolve_template(api_key, TEMPLATE_NAME)
    network_volume_summary = summarize_network_volume(volume)
    template_summary = summarize_template(template)
    network_volume_json = json.dumps(network_volume_summary, indent=2)
    template_json = json.dumps(template_summary, indent=2)

    print("Resolved network volume:", network_volume_json, file=sys.stderr)
    print("Resolved template:", template_json, file=sys.stderr)

    payload = build_create_pod_payload(volume=volume, template=template)
    payload_json = json.dumps(payload, indent=2)
    data_center_id = volume.get("dataCenterId")

    print("Creating Pod with payload:", file=sys.stderr)
    print(payload_json, file=sys.stderr)

    try:
        pod = runpod_request("POST", "/pods", api_key, json_body=payload)
    except RunPodAPIError as error:
        handled = print_create_pod_error(error=error, data_center_id=data_center_id)

        if handled:
            return 1

        raise

    if not isinstance(pod, dict):
        raise RuntimeError(f"Unexpected create-Pod response: {pod!r}")

    pod_id_value = pod.get("id")

    if not isinstance(pod_id_value, str):
        raise RuntimeError(f"Unexpected create-Pod response without string id: {pod!r}")

    pod_id = pod_id_value
    state_path = write_pod_state(pod)
    created_pod_summary = summarize_pod(pod)
    created_pod_json = json.dumps(created_pod_summary, indent=2)

    print(created_pod_json)
    print(f"Saved Pod state to {state_path}", file=sys.stderr)

    configure_ssh_for_pod(api_key=api_key, pod_id=pod_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
