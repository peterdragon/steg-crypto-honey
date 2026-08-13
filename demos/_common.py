"""Shared bit helpers used by the demos."""


def bytes_to_bits(data):
    """MSB-first list of 0/1 ints for the given bytes."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(bits):
    """Inverse of bytes_to_bits; trailing bits are zero-padded to a byte."""
    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        if len(chunk) < 8:
            chunk = chunk + [0] * (8 - len(chunk))
        value = 0
        for b in chunk:
            value = (value << 1) | b
        out.append(value)
    return bytes(out)


def int_to_bits(value, width):
    """MSB-first fixed-width bit list for a non-negative integer."""
    return [(value >> i) & 1 for i in range(width - 1, -1, -1)]


def bits_to_int(bits):
    value = 0
    for b in bits:
        value = (value << 1) | b
    return value


class BitReader:
    """Reads bits left-to-right; pads with zeros once exhausted."""

    def __init__(self, bits):
        self.bits = bits
        self.pos = 0

    def read(self, n):
        value = 0
        for _ in range(n):
            bit = self.bits[self.pos] if self.pos < len(self.bits) else 0
            self.pos += 1
            value = (value << 1) | bit
        return value

    def exhausted(self):
        return self.pos >= len(self.bits)
