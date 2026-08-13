#!/usr/bin/env python3
"""Keyed CFG mimic function: Wayner (1992) mimic + a passphrase-keyed keystream,
with a per-message nonce so one secret can safely encode many messages.

The message rides the *path* through an unambiguous grammar (see cfg_mimic.py).
A secret is layered on top as a stream cipher:

    key       = Argon2id(secret, fixed salt)               # KDF, keyed on secret
    keystream = SHA-256(key || nonce || counter) counter mode   # CSPRNG bytes
    ciphertext = pad(length-prefixed message) XOR keystream     # stream cipher
    cover text = keyless mimic-encode(ciphertext)               # Wayner path

Equivalently: stream-cipher the padded payload, then apply Wayner's keyless
mimic to the ciphertext. Encoding and decoding both perform a left-most grammar
traversal in the same order, so they stay in lockstep.

    encode(secret, data)        -> (text, nonce)
    decode(secret, text, nonce) -> data

Distribution model: the nonce is handed out WITH the secret (out-of-band, like a
key + IV pair); the cover text is the only thing sent over the public channel.
The payload carries a 2-byte length header and is padded to PAD_BLOCK bytes, so
the cover text is self-describing and its length only leaks the message length
rounded up to a block (see the side-channel notes below).

--- Side-channel notes ------------------------------------------------------
Secrecy lives ONLY in the KDF and keystream. The grammar/DTE mapping is public:
anyone can parse the cover text back into the ciphertext bits without the key
(that is exactly the wrong-key path). So the Huffman/grammar traversal branches
only on ciphertext, which is not secret, and is NOT a key-leaking channel. That
leaves three things to harden, all standard stream-cipher hygiene:
  * KDF: use vetted native Argon2id (data-independent first pass resists
    cache-timing); run it isolated and mlock its memory in production.
  * Keystream/XOR: SHA-256 (hashlib) is native and constant-time; the XOR is
    done at the byte level via big-integer XOR (value-independent).
  * Key lifetime: keys are held in bytearrays and wiped after use (best-effort
    in a GC'd runtime; the argon2 return value is a short-lived immutable copy).
  * Length: payload is padded to PAD_BLOCK to blunt length-based traffic
    analysis; pad to a fixed maximum for stronger hiding.
Deliberately NO message MAC: authenticating the plaintext would hand an attacker
an oracle to recognise the correct key offline, destroying the honey property.
Put integrity/authentication at the transport layer under a separate high-entropy
key (where a constant-time compare such as hmac.compare_digest is mandatory).
Pure Python cannot guarantee constant time; for a co-resident or physical
attacker, delegate every secret-touching primitive to vetted native libraries.

Argon2id needs argon2-cffi (in ./.venv); falls back to stdlib scrypt otherwise.

Run:  ./.venv/bin/python keyed_cfg_mimic.py   (or: python3 keyed_cfg_mimic.py)
"""

import hashlib
import os

from _common import bytes_to_bits, bits_to_bytes, int_to_bits, bits_to_int
from cfg_mimic import GRAMMAR, START, _code_width

try:
    from argon2.low_level import hash_secret_raw, Type
    _HAVE_ARGON2 = True
except ImportError:
    _HAVE_ARGON2 = False

FIXED_SALT = b"keyed-cfg-mimic/v1"   # application constant for the KDF
KEY_LEN = 32
NONCE_LEN = 16                        # 128-bit public nonce (sent with the secret)
LENGTH_BYTES = 2                      # message-length header (max 65535 bytes)
PAD_BLOCK = 16                        # pad payload to a multiple of this many bytes


def _wipe(buf):
    """Best-effort zeroing of a mutable key buffer."""
    if isinstance(buf, bytearray):
        for i in range(len(buf)):
            buf[i] = 0


def _xor_bytes(a, b):
    """Native byte-level XOR of the common prefix of a and b."""
    n = min(len(a), len(b))
    return (int.from_bytes(a[:n], "big") ^ int.from_bytes(b[:n], "big")).to_bytes(n, "big")


def pack_payload(data):
    """length-header || data, zero-padded up to a multiple of PAD_BLOCK bytes."""
    payload = len(data).to_bytes(LENGTH_BYTES, "big") + data
    pad = (-len(payload)) % PAD_BLOCK
    return payload + b"\x00" * pad


def unpack_payload(raw):
    """Recover data from a (possibly over-long) decoded payload via its header."""
    length = int.from_bytes(raw[:LENGTH_BYTES], "big")
    return raw[LENGTH_BYTES:LENGTH_BYTES + length]


def derive_key(secret):
    """Passphrase or number -> 32-byte key (bytearray) via a slow, memory-hard KDF."""
    data = secret.encode() if isinstance(secret, str) else str(secret).encode()
    if _HAVE_ARGON2:
        raw = hash_secret_raw(data, FIXED_SALT, time_cost=3, memory_cost=65536,
                              parallelism=4, hash_len=KEY_LEN, type=Type.ID)
    else:
        raw = hashlib.scrypt(data, salt=FIXED_SALT, n=2 ** 14, r=8, p=1, dklen=KEY_LEN)
    return bytearray(raw)


class Keystream:
    """CSPRNG stream: SHA-256(key || nonce || counter), consumed left to right."""

    def __init__(self, key, nonce=b""):
        self.key = bytearray(key)
        self.nonce = bytes(nonce)
        self.counter = 0
        self.buf = []

    def _refill(self):
        block = hashlib.sha256(
            bytes(self.key) + self.nonce + self.counter.to_bytes(8, "big")).digest()
        self.counter += 1
        for byte in block:
            for i in range(7, -1, -1):
                self.buf.append((byte >> i) & 1)

    def bits(self, n):
        while len(self.buf) < n:
            self._refill()
        out = self.buf[:n]
        self.buf = self.buf[n:]
        return out

    def bytes(self, n):
        return bits_to_bytes(self.bits(8 * n))

    def wipe(self):
        _wipe(self.key)


def encode(secret, data, nonce=None, key=None):
    """Hide bytes in keyed grammatical text. Returns (text, nonce).

    A random nonce is generated unless one is supplied (supplying one is only for
    testing / reproducibility). Distribute the returned nonce with the secret.
    Pass a pre-derived `key` (bytearray) to skip the slow KDF across many calls;
    the caller then owns and wipes that key.
    """
    if nonce is None:
        nonce = os.urandom(NONCE_LEN)
    owned_key = key is None
    k = derive_key(secret) if owned_key else key
    try:
        payload = pack_payload(data)
        ks = Keystream(k, nonce)
        cipher = _xor_bytes(payload, ks.bytes(len(payload)))
        cbits = bytes_to_bits(cipher)
        pos = 0
        tokens = []

        def expand(sym):
            nonlocal pos
            if sym not in GRAMMAR:
                tokens.append(sym)
                return
            prods = GRAMMAR[sym]
            if len(prods) == 1:
                idx = 0
            else:
                width = _code_width(len(prods))
                chunk = cbits[pos:pos + width]
                if len(chunk) < width:
                    chunk = chunk + [0] * (width - len(chunk))
                idx = bits_to_int(chunk)
                pos += width
            for s in prods[idx]:
                expand(s)

        while pos < len(cbits):
            expand(START)
        if not tokens:
            expand(START)
        return " ".join(tokens), nonce
    finally:
        if owned_key:
            _wipe(k)


def decode(secret, text, nonce, key=None):
    """Recover bytes from keyed cover text using (secret, grammar, nonce)."""
    owned_key = key is None
    k = derive_key(secret) if owned_key else key
    try:
        ks = Keystream(k, nonce)
        tokens = text.split()
        pos = 0
        cbits = []

        def parse(sym):
            nonlocal pos
            if sym not in GRAMMAR:
                assert tokens[pos] == sym, f"expected {sym!r}, got {tokens[pos]!r}"
                pos += 1
                return
            prods = GRAMMAR[sym]
            if len(prods) == 1:
                idx = 0
            else:
                idx = None
                for i, prod in enumerate(prods):
                    if prod[0] == tokens[pos]:
                        idx = i
                        break
                assert idx is not None, f"no production of {sym} starts with {tokens[pos]!r}"
                cbits.extend(int_to_bits(idx, _code_width(len(prods))))
            for s in prods[idx]:
                parse(s)

        while pos < len(tokens):
            parse(START)
        cipher = bits_to_bytes(cbits)
        payload = _xor_bytes(cipher, ks.bytes(len(cipher)))
        return unpack_payload(payload)
    finally:
        if owned_key:
            _wipe(k)


def main():
    print(f"KDF in use : {'Argon2id (argon2-cffi)' if _HAVE_ARGON2 else 'scrypt (stdlib fallback)'}")
    secret = "correct horse battery staple"
    message = b"HI!"
    print(f"secret     : {secret!r}")
    print(f"message    : {message!r}")

    text, nonce = encode(secret, message)
    print(f"\nnonce #1   : {nonce.hex()}   (distribute this with the secret)")
    print("cover text #1 (the only thing sent on the public channel):")
    print(f"  {text}")
    recovered = decode(secret, text, nonce)
    print(f"decode (correct secret + nonce) -> {recovered!r}")
    assert recovered == message, "round-trip failed"

    text2, nonce2 = encode(secret, message)
    print(f"\nnonce #2   : {nonce2.hex()}")
    print("cover text #2 (same secret, same message, fresh nonce):")
    print(f"  {text2}")
    assert decode(secret, text2, nonce2) == message
    print(f"same (secret,message) but different text: {text != text2}")
    print("=> one secret can safely encode many messages (no keystream reuse)")

    wrong = decode("hunter2", text, nonce)
    print(f"\ndecode with WRONG secret -> {wrong!r}  (valid parse, bogus plaintext)")

    reused, _ = encode(secret, message, nonce=nonce)
    print(f"\nreusing (secret, nonce) reproduces identical text: {reused == text}")
    print("=> nonces must be unique; that is why they are generated randomly")

    # Length hiding: messages that fit in the same padded block share a length.
    a, _ = encode(secret, b"Y")
    b, _ = encode(secret, b"Yes!!")
    print(f"\nlength hiding: 1-byte and 5-byte messages -> "
          f"{len(a.split())} vs {len(b.split())} tokens "
          f"({'same' if len(a.split()) == len(b.split()) else 'different'})")
    print(f"=> both padded to PAD_BLOCK={PAD_BLOCK} bytes, so length leaks only per block")

    print("\nround-trip OK")


if __name__ == "__main__":
    main()
