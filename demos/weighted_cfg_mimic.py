#!/usr/bin/env python3
"""Weighted (statistical-mimicry) keyed CFG mimic: Wayner's frequency matching
layered on top of the keyed scheme in keyed_cfg_mimic.py.

Plain keyed_cfg_mimic uses fixed-length codes over equiprobable, power-of-two
production counts, so every word in a slot is equally likely. Real text is not
uniform ("the" >> "yonder"). Wayner's fix is variable-length codes: give each
production a weight and build a per-variable Huffman code, so a frequent word
gets a short code (chosen often) and a rare word a long one (chosen seldom).

Why the key layer makes this clean:
    payload  --XOR keystream-->  ciphertext (pseudo-uniform)  --Huffman--> text
The CSPRNG keystream whitens the payload, so the bits driving the grammar are
uniform regardless of the message. Under uniform input, a Huffman leaf of code
length L is chosen with probability 2^-L, so the emitted word frequencies track
the code -- i.e. the target weights -- no matter what the plaintext looks like.

This reuses derive_key / Keystream / nonce / length-header / padding from
keyed_cfg_mimic, so it keeps every property of that scheme (key confidentiality,
per-message nonce, honey-style wrong-key decode, fixed-length padding, native
byte-level XOR, bytearray key wiping, and the same side-channel reasoning) and
adds only the frequency shaping. See keyed_cfg_mimic.py for the side-channel
notes; they apply unchanged here (secrecy lives only in the KDF + keystream).

Bonus: Huffman needs neither power-of-two counts nor dyadic weights, so this
layer also lifts the power-of-two restriction of the fixed-length version.

    encode(secret, data)        -> (text, nonce)
    decode(secret, text, nonce) -> data

Run:  ./.venv/bin/python weighted_cfg_mimic.py   (or: python3 weighted_cfg_mimic.py)
"""

import os
import random

from _common import bytes_to_bits, bits_to_bytes
from cfg_mimic import GRAMMAR, START
from keyed_cfg_mimic import (derive_key, Keystream, NONCE_LEN, LENGTH_BYTES,
                             pack_payload, unpack_payload, _xor_bytes, _wipe)

# Production weights (relative frequencies). Any variable omitted here, or given
# no weights, defaults to uniform. Weights need not sum to anything, be dyadic,
# or come in power-of-two counts.
WEIGHTS = {
    "GREET":  [40, 25, 12, 8, 6, 5, 2, 2],      # "Hello," common, "Ahoy," rare
    "ADVERB": [28, 18, 10, 22, 9, 6, 4, 3],
    "VERB":   [20, 16, 15, 9, 6, 6, 6, 5, 4, 4, 2, 2, 2, 1, 1, 1],
    "ART":    [55, 30, 9, 6],                    # "the" >> "that"
    "PREP":   [30, 20, 14, 12, 8, 7, 5, 4],
    "TIME":   [26, 20, 16, 12, 9, 8, 5, 4],
    "CONJ":   [30, 22, 14, 12, 8, 7, 4, 3],
}


def _build(weights):
    """Return (tree, codes) for one variable. tree: nested (left,right) with int
    leaves; codes: {leaf_index: [bits]}."""
    import heapq
    if len(weights) == 1:
        return 0, {0: []}
    counter = len(weights)
    heap = [(w, i, i) for i, w in enumerate(weights)]  # (weight, tiebreak, node)
    heapq.heapify(heap)
    while len(heap) > 1:
        w1, _, n1 = heapq.heappop(heap)
        w2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (w1 + w2, counter, (n1, n2)))
        counter += 1
    tree = heap[0][2]
    codes = {}

    def walk(node, prefix):
        if isinstance(node, int):
            codes[node] = prefix
        else:
            walk(node[0], prefix + [0])
            walk(node[1], prefix + [1])

    walk(tree, [])
    return tree, codes


def _codebooks(weights):
    """Build Huffman tree + codes for every branching variable in the grammar."""
    books = {}
    for sym, prods in GRAMMAR.items():
        if len(prods) > 1:
            w = weights.get(sym) or [1] * len(prods)
            if len(w) != len(prods):
                raise ValueError(f"{sym}: {len(w)} weights for {len(prods)} productions")
            books[sym] = _build(w)
    return books


def encode(secret, data, nonce=None, weights=WEIGHTS, key=None):
    """Hide bytes in weighted, keyed grammatical text. Returns (text, nonce).

    Pass a pre-derived `key` (bytearray) to skip the (slow) KDF when encoding many
    messages under the same secret (e.g. benchmarks); the caller then owns/wipes
    it. Otherwise the key is derived from `secret` and wiped after use.
    """
    if nonce is None:
        nonce = os.urandom(NONCE_LEN)
    if len(data) >= (1 << (8 * LENGTH_BYTES)):
        raise ValueError("message too long for the length header")
    owned_key = key is None
    k = derive_key(secret) if owned_key else key
    try:
        payload = pack_payload(data)
        ks = Keystream(k, nonce)
        cipher = _xor_bytes(payload, ks.bytes(len(payload)))  # native byte-level XOR
        cbits = bytes_to_bits(cipher)
        books = _codebooks(weights)

        pos = 0
        tokens = []

        def next_bit():
            nonlocal pos
            bit = cbits[pos] if pos < len(cbits) else 0  # pad past end of message
            pos += 1
            return bit

        def expand(sym):
            if sym not in GRAMMAR:
                tokens.append(sym)
                return
            prods = GRAMMAR[sym]
            if len(prods) == 1:
                idx = 0
            else:
                node, _ = books[sym]
                while not isinstance(node, int):
                    node = node[0] if next_bit() == 0 else node[1]
                idx = node
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


def decode(secret, text, nonce, weights=WEIGHTS, key=None):
    """Recover bytes from weighted, keyed cover text using (secret, grammar, nonce)."""
    owned_key = key is None
    k = derive_key(secret) if owned_key else key
    try:
        ks = Keystream(k, nonce)
        books = _codebooks(weights)
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
                cbits.extend(books[sym][1][idx])
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


def expected_bits_per_sentence(weights=WEIGHTS):
    """Under uniform input, expected code length summed over one sentence's slots."""
    books = _codebooks(weights)
    total = 0.0
    for sym in GRAMMAR[START][0]:
        if sym in books:
            _, codes = books[sym]
            total += sum(len(c) * 2 ** -len(c) for c in codes.values())
    return total


def _demo_frequency_shaping(secret, weights, trials, slot="GREET"):
    """Encode many random messages; tally the chosen word in `slot` per sentence."""
    codes = _codebooks(weights)[slot][1]
    prods = GRAMMAR[slot]
    target = {prods[i][0]: 2 ** -len(codes[i]) for i in range(len(prods))}
    seen = {prods[i][0]: 0 for i in range(len(prods))}
    key = derive_key(secret)  # derive once; reuse across trials to skip slow KDF
    n = 0
    for _ in range(trials):
        data = os.urandom(random.randint(4, 24))
        text, _ = encode(secret, data, weights=weights, key=key)
        for sentence in text.split(". "):
            first = sentence.split()[0]
            if first in seen:
                seen[first] += 1
                n += 1
    _wipe(key)
    order = sorted(range(len(prods)), key=lambda i: len(codes[i]))
    print(f"\n{slot} frequency shaping over {n} sentence-openings:")
    print(f"  {'word':<14}{'bits':>5}{'target':>9}{'empirical':>11}")
    for i in order:
        word = prods[i][0]
        print(f"  {word:<14}{len(codes[i]):>5}{target[word]:>9.3f}{seen[word] / n:>11.3f}")


def main():
    from keyed_cfg_mimic import _HAVE_ARGON2
    print(f"KDF in use : {'Argon2id (argon2-cffi)' if _HAVE_ARGON2 else 'scrypt (stdlib fallback)'}")
    print(f"capacity   : ~{expected_bits_per_sentence():.1f} bits/sentence "
          f"(expected, variable-length codes)")

    secret = "correct horse battery staple"
    message = b"Rendezvous at the old pier!"
    print(f"\nsecret     : {secret!r}")
    print(f"message    : {message!r}")

    text, nonce = encode(secret, message)
    print(f"nonce      : {nonce.hex()}   (distribute with the secret)")
    print("cover text :")
    print(f"  {text}")
    recovered = decode(secret, text, nonce)
    print(f"decode (correct secret) -> {recovered!r}")
    assert recovered == message, "round-trip failed"

    wrong = decode("hunter2", text, nonce)
    print(f"decode (WRONG secret)   -> {wrong!r}  (valid parse, bogus plaintext)")

    random.seed(0)
    _demo_frequency_shaping(secret, WEIGHTS, trials=4000, slot="GREET")
    print("=> frequent words get short codes and appear often; empirical ~ target")

    print("\nround-trip OK")


if __name__ == "__main__":
    main()
