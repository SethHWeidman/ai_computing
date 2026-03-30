import torch


def _print_rows(label: str, t: torch.Tensor, fmt: str, row_width: int = 8) -> None:
    print(label)
    for i, row in enumerate(t):
        vals = row.tolist()
        chunks = [vals[j : j + row_width] for j in range(0, len(vals), row_width)]
        for k, chunk in enumerate(chunks):
            prefix = f"  [{i}]" if k == 0 else "     "
            print(prefix + "  " + "  ".join(fmt.format(v) for v in chunk))


def print_tensor(label: str, t: torch.Tensor, fmt: str = "{:8.3f}") -> None:
    _print_rows(label, t, fmt=fmt, row_width=8)
