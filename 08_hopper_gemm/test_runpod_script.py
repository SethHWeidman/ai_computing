#!/usr/bin/env python3

import importlib.util as importlib_util
import pathlib
import tempfile
import textwrap
import types
import unittest

TEST_FILE = pathlib.Path(__file__)
TEST_FILE_PATH = TEST_FILE.resolve()
TEST_DIR = TEST_FILE_PATH.parent

RUNPOD_SCRIPT_CANDIDATES = [
    TEST_DIR / "runpod_script.py",
    TEST_DIR / "08_hopper_gemm" / "runpod_script.py",
]


def _load_runpod_script() -> types.ModuleType:
    """Load runpod_script.py without importing it by package name."""
    script_path = None

    for candidate in RUNPOD_SCRIPT_CANDIDATES:
        candidate_exists = candidate.exists()

        if candidate_exists:
            script_path = candidate
            break

    if script_path is None:
        choices = "\n".join(f"  {path}" for path in RUNPOD_SCRIPT_CANDIDATES)
        raise RuntimeError(f"Could not find runpod_script.py. Tried:\n{choices}")

    spec = importlib_util.spec_from_file_location(
        "runpod_script_under_test", script_path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {script_path}")

    loader = spec.loader
    module = importlib_util.module_from_spec(spec)
    loader.exec_module(module)

    return module


def _get_config_line_parts(line: str) -> list[str]:
    """Split one SSH config line into directive tokens."""
    stripped_line = line.strip()
    return stripped_line.split()


def _is_host_parts(parts: list[str]) -> bool:
    """Return whether split SSH config tokens start a Host block."""
    if not parts:
        return False

    directive = parts[0]
    directive_name = directive.lower()

    return directive_name == "host"


def get_host_block(config_text: str, alias: str) -> str | None:
    """Return the SSH Host block text for one alias."""
    lines = config_text.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index]
        parts = _get_config_line_parts(line)

        if not _is_host_parts(parts):
            index += 1
            continue

        block_start = index
        block_end = index + 1

        while block_end < len(lines):
            next_line = lines[block_end]
            next_parts = _get_config_line_parts(next_line)

            if _is_host_parts(next_parts):
                break

            block_end += 1

        aliases = parts[1:]

        if alias in aliases:
            block_lines = lines[block_start:block_end]
            return "\n".join(block_lines)

        index = block_end

    return None


class UpdateSshConfigFileTests(unittest.TestCase):
    def setUp(self) -> None:
        module = _load_runpod_script()
        self.module = module

        temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir = temp_dir
        self.addCleanup(temp_dir.cleanup)

        temp_dir_path = pathlib.Path(temp_dir.name)
        ssh_dir = temp_dir_path / ".ssh"
        config_path = ssh_dir / "config"
        ssh_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = config_path

        module.SSH_CONFIG_PATH = str(config_path)
        module.SSH_HOST_ALIAS = "runpod-h100"
        module.SSH_USER = "root"
        module.SSH_IDENTITY_FILE = "~/.ssh/id_ed25519"

    def write_config(self, text: str) -> None:
        dedented_text = textwrap.dedent(text)
        config_text = dedented_text.lstrip()
        self.config_path.write_text(config_text)

    def read_config(self) -> str:
        return self.config_path.read_text()

    def test_returns_path_not_tuple(self) -> None:
        self.write_config("""
            Host runpod-h100
              HostName 213.181.122.136
              User root
              Port 15454
              IdentityFile ~/.ssh/id_ed25519
            """)

        result = self.module.update_ssh_config_file(
            public_ip="213.181.122.162", ssh_port=14291
        )

        self.assertIsInstance(result, pathlib.Path)
        self.assertEqual(result, self.config_path)
        self.assertNotIsInstance(result, tuple)

    def test_updates_only_runpod_h100_block(self) -> None:
        self.write_config("""
            Host *
              IdentityFile ~/.ssh/id_ed25519

            Host runpod-l4
              HostName 205.196.17.178
              User root
              Port 11790
              IdentityFile ~/.ssh/id_ed25519
              StrictHostKeyChecking no
              UserKnownHostsFile /dev/null

            Host runpod-h100
              HostName 213.181.122.136
              User root
              Port 15454
              IdentityFile ~/.ssh/id_ed25519
              ServerAliveInterval 60
              ServerAliveCountMax 120
              ForwardAgent yes

            Host github.com
              AddKeysToAgent yes
              UseKeychain yes
              IdentityFile ~/.ssh/id_ed25519
            """)

        self.module.update_ssh_config_file(public_ip="213.181.122.162", ssh_port=14291)

        config_text = self.read_config()

        wildcard_block = get_host_block(config_text, "*")
        runpod_l4_block = get_host_block(config_text, "runpod-l4")
        runpod_h100_block = get_host_block(config_text, "runpod-h100")
        github_block = get_host_block(config_text, "github.com")

        self.assertIsNotNone(wildcard_block)
        self.assertIsNotNone(runpod_l4_block)
        self.assertIsNotNone(runpod_h100_block)
        self.assertIsNotNone(github_block)

        self.assertIn("IdentityFile ~/.ssh/id_ed25519", wildcard_block)

        self.assertIn("HostName 205.196.17.178", runpod_l4_block)
        self.assertIn("Port 11790", runpod_l4_block)

        self.assertIn("HostName 213.181.122.162", runpod_h100_block)
        self.assertIn("Port 14291", runpod_h100_block)
        self.assertNotIn("HostName 213.181.122.136", runpod_h100_block)
        self.assertNotIn("Port 15454", runpod_h100_block)

        self.assertIn("AddKeysToAgent yes", github_block)
        self.assertIn("UseKeychain yes", github_block)

    def test_creates_runpod_h100_block_when_missing(self) -> None:
        self.write_config("""
            Host *
              IdentityFile ~/.ssh/id_ed25519

            Host github.com
              AddKeysToAgent yes
              UseKeychain yes
              IdentityFile ~/.ssh/id_ed25519
            """)

        self.module.update_ssh_config_file(public_ip="213.181.122.162", ssh_port=14291)

        config_text = self.read_config()
        runpod_h100_block = get_host_block(config_text, "runpod-h100")

        self.assertIsNotNone(runpod_h100_block)
        self.assertIn("Host runpod-h100", runpod_h100_block)
        self.assertIn("HostName 213.181.122.162", runpod_h100_block)
        self.assertIn("User root", runpod_h100_block)
        self.assertIn("Port 14291", runpod_h100_block)
        self.assertIn("IdentityFile ~/.ssh/id_ed25519", runpod_h100_block)
        self.assertIn("ServerAliveInterval 60", runpod_h100_block)
        self.assertIn("ServerAliveCountMax 120", runpod_h100_block)
        self.assertIn("ForwardAgent yes", runpod_h100_block)

    def test_does_not_treat_runpod_h100_old_as_match(self) -> None:
        self.write_config("""
            Host runpod-h100-old
              HostName 1.2.3.4
              User root
              Port 9999
              IdentityFile ~/.ssh/id_ed25519
            """)

        self.module.update_ssh_config_file(public_ip="213.181.122.162", ssh_port=14291)

        config_text = self.read_config()

        old_block = get_host_block(config_text, "runpod-h100-old")
        new_block = get_host_block(config_text, "runpod-h100")

        self.assertIsNotNone(old_block)
        self.assertIsNotNone(new_block)

        self.assertIn("HostName 1.2.3.4", old_block)
        self.assertIn("Port 9999", old_block)

        self.assertIn("HostName 213.181.122.162", new_block)
        self.assertIn("Port 14291", new_block)


if __name__ == "__main__":
    unittest.main()
