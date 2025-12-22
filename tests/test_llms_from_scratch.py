import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LLMS_ROOT = REPO_ROOT / "LLMs-from-scratch"
CH04_ROOT = LLMS_ROOT / "ch04"


def run_python(script: str | pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Run a Python script relative to the repo root and capture output."""
    script_path = script if isinstance(script, pathlib.Path) else REPO_ROOT / script
    result = subprocess.run(
        [sys.executable, str(script_path)], cwd=REPO_ROOT, text=True, capture_output=True
    )
    # Make it easy to see script errors in test failures
    if result.returncode != 0:
        raise AssertionError(
            f"Script {script} failed with code {result.returncode}:\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return result


def test_multihead_attention_example_shapes_match() -> None:
    """multihead_attention_example should report matching output shapes."""
    proc = run_python(
        LLMS_ROOT / "ch03/01_main-chapter-code/multihead_attention_reference.py"
    )
    out = proc.stdout

    # Ensure the script ran and printed the two shape lines
    assert "SingleHeadAttention output shape:" in out
    assert "MultiHeadAttention output shape:" in out

    # Optionally, check that both shapes include the expected size tuple
    assert "torch.Size([2, 6, 4])" in out


def test_gpt_kv_cache_reference_matches_ch04_output() -> None:
    """KV-cache reference GPT should produce the same text as the ch04 GPT script."""
    proc_baseline = run_python(
        CH04_ROOT / "03_kv-cache/gpt_without_kv_cache_reference.py"
    )
    proc_cached = run_python(CH04_ROOT / "03_kv-cache/gpt_with_kv_cache_reference.py")

    out_base = proc_baseline.stdout
    out_cached = proc_cached.stdout

    # Extract the generated output text lines for comparison
    def extract_output_text(full: str) -> str:
        for line in full.splitlines():
            if line.startswith("Output text:"):
                return line
        raise AssertionError("No 'Output text:' line found in script output")

    text_base = extract_output_text(out_base)
    text_cached = extract_output_text(out_cached)

    assert text_base == text_cached


def test_mla_reference_script_runs() -> None:
    """Ensure the MLA KV-cache reference script executes without errors."""
    proc = run_python(CH04_ROOT / "05_mla/gpt_with_kv_mla_reference.py")
    assert "Output text:" in proc.stdout


def test_moe_reference_script_runs() -> None:
    """Ensure the MoE KV-cache reference script executes without errors."""
    proc = run_python(CH04_ROOT / "07_moe/gpt_with_kv_moe_reference.py")
    assert "Output text:" in proc.stdout
