import torch


def pack_nibbles(payload_codes_fp4: torch.Tensor) -> torch.Tensor:
    """
    Packs 4-bit payload codes into bytes.
    Convention used here:
      - first FP4 value goes in low nibble
      - second FP4 value goes in high nibble
    This packing convention is a host-side convenience, not a claim about
    NVIDIA library internal storage layout.
    """
    flat = payload_codes_fp4.reshape(-1).to(torch.uint8)

    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)], dim=0)

    lo = flat[0::2] & 0x0F
    hi = (flat[1::2] & 0x0F) << 4
    return lo | hi


if __name__ == "__main__":
    # Case 1: even number of codes — show bit layout for each pair
    print("Case 1: four codes packed into two bytes")
    codes_even = torch.tensor([5, 10, 3, 7], dtype=torch.uint8)
    packed_even = pack_nibbles(codes_even)
    print(
        f"  Input codes: {codes_even.tolist()}  "
        f"(binary: {[f'{c:04b}' for c in codes_even.tolist()]})"
    )
    for i, byte in enumerate(packed_even.tolist()):
        lo = codes_even[i * 2].item()
        hi = codes_even[i * 2 + 1].item()
        print(
            f"  byte[{i}]: 0x{byte:02x} = {byte:08b}  ← hi={hi:04b} ({hi})  lo={lo:04b} "
            f"({lo})"
        )

    # Case 2: odd number of codes — last nibble is zero-padded
    print()
    print("Case 2: three codes (odd) → last byte is zero-padded in high nibble")
    codes_odd = torch.tensor([5, 10, 3], dtype=torch.uint8)
    packed_odd = pack_nibbles(codes_odd)
    print(f"  Input codes: {codes_odd.tolist()}")
    print(
        f"  byte[0]: 0x{packed_odd[0].item():02x} = {packed_odd[0].item():08b}  ← "
        f"{codes_odd[1].item():04b} | {codes_odd[0].item():04b}"
    )
    print(
        f"  byte[1]: 0x{packed_odd[1].item():02x} = {packed_odd[1].item():08b}  ← 0000 "
        f"(pad) | {codes_odd[2].item():04b}"
    )

    # Case 3: round-trip — pack then unpack manually to verify
    print()
    print("Case 3: round-trip unpack (low nibble = & 0x0F, high nibble = >> 4)")
    codes_rt = torch.tensor([0, 1, 5, 7, 15, 8], dtype=torch.uint8)
    packed_rt = pack_nibbles(codes_rt)
    unpacked = []
    for byte in packed_rt.tolist():
        unpacked.append(byte & 0x0F)
        unpacked.append((byte >> 4) & 0x0F)
    print(f"  Original: {codes_rt.tolist()}")
    print(f"  Packed:   {[f'0x{b:02x}' for b in packed_rt.tolist()]}")
    print(f"  Unpacked: {unpacked}")
