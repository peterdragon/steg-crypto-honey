#!/usr/bin/env python3
"""Discop sampling teaching demo: distribution-preserving cover bits.

lm_mimic.py uses Huffman codes, which induce token probabilities of exactly
2^-code_length.  Unless the LM distribution is dyadic, that differs from the true
P, so the stego distribution Q != P and KL(Q||P) > 0 — a signal a steganalyzer
can exploit.  Discop (Ding et al., IEEE S&P 2023) removes this gap *relative to
the deployed fixed-point sampler* (not necessarily vs raw platform text).

Discop's idea — "distribution copies" via rotation
--------------------------------------------------
Lay the token distribution P out as arcs on a circle [0, TOTAL) (a fixed-point
CDF).  A *distribution copy* is a rotation of that circle.  Sampling = draw a
shared pseudorandom point r and emit the token whose arc contains it; over
uniform r this yields exactly P.  A rotation by TOTAL/2 gives a second copy —
same arc widths, so the same marginal P.  The MESSAGE BIT chooses which copy:

    emit token_at( (r + m * TOTAL/2) mod TOTAL )          # m in {0, 1}

For any fixed m, r ~ Uniform makes the argument uniform, so the emitted token is
distributed EXACTLY as (quantized) P.  Hence zero KL divergence versus the
sampler the model itself would use — the message leaves no distributional trace.
This is the crucial difference from Huffman: security does not require the
message to be uniform; the shared randomness r carries all the entropy.

Recovery
--------
The receiver rebuilds P from the same context (deterministic LM), the same r
(public tape from the nonce), and reads the emitted token:
  * If token_at(r) == token_at(r + TOTAL/2)  -> the two copies collide on one
    wide arc: 0 bits were embedded here (a token with prob > 1/2).
  * Otherwise exactly one bit: m = 0 if the token is token_at(r), else m = 1.
Sender and receiver compute the collide/embed decision from (context, r) alone,
so they stay in lockstep with no side information.

Keyed integration (same family as keyed_cfg_mimic / lm_mimic)
-------------------------------------------------------------
    payload      = pad(length-prefixed message)
    cipher_bits  = bytes_to_bits(payload) XOR ks_A     # confidentiality (secret)
    r_i          = PublicTape(nonce)                   # sampling entropy (public)
    token_i      = Discop( P_i, r_i, next cipher bit )
Segmentation is key-independent: every key agrees on which steps embed and on
the raw bit-count. Wrong secret -> wrong ks_A unmask -> valid corpus words but
garbled payload (honey at the bit layer).

Capacity vs security trade
--------------------------
The single-bit rotation embeds ~1 bit/word (vs Huffman's ~entropy bits/word),
buying provable zero-KLD.  Discop's paper recovers the full rate by recursively
building 2^k copies via Huffman trees over the tokens; the security argument is
identical (every copy shares the marginal), only the rate changes.

Quantization note: arcs are integer widths summing to TOTAL = 2^PRECISION, so
the preserved distribution is P quantized to 1/TOTAL.  A normal sampler using
the same fixed-point CDF has the same quantization, so KLD versus the *deployed*
sampler is exactly zero (Discop, sec. IV).

Run:  python3 discop_mimic.py
      ./.venv/bin/python discop_mimic.py   (Argon2id KDF)
"""

import bisect
import hashlib
import heapq
import math
import os

from _common import bytes_to_bits, bits_to_bytes
from keyed_cfg_mimic import (
    NONCE_LEN, _wipe, derive_key, pack_payload, unpack_payload,
    Keystream, _HAVE_ARGON2,
)
from lm_mimic import LM  # reuse the bigram LM + corpus

PRECISION = 32
TOTAL = 1 << PRECISION
HALF = TOTAL >> 1
R_BYTES = PRECISION // 8

_DOMAIN_MASK = b"\x00"      # secret: payload mask ks_A
_DOMAIN_PUBLIC = b"\x02"    # public: segmentation tape from nonce only

_cdf_cache: dict = {}


class PublicTape:
    """Keyless SHA-256 counter mode driven only by the public nonce.

    Supplies per-step rotation coins so embedding positions (and the raw bit
    count) are identical for every key.
    """

    def __init__(self, nonce: bytes, domain: bytes = _DOMAIN_PUBLIC):
        self._prefix = bytes(nonce) + bytes(domain)
        self._ctr = 0
        self._buf = b""

    def bytes(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            if not self._buf:
                self._buf = hashlib.sha256(
                    self._prefix + self._ctr.to_bytes(8, "big")).digest()
                self._ctr += 1
            take = min(n - len(out), len(self._buf))
            out += self._buf[:take]
            self._buf = self._buf[take:]
        return bytes(out)


def _dist_cdf(prev: str) -> tuple:
    """Fixed-point CDF for context `prev`: (tokens, cum) with cum[-1] == TOTAL.

    Tokens are sorted so sender and receiver lay out identical arcs. Every arc
    has width >= 1 so every vocab word is reachable and decodable.
    """
    if prev in _cdf_cache:
        return _cdf_cache[prev]
    dist = LM.distribution(prev)
    tokens = sorted(dist)
    widths = [max(1, round(dist[t] * TOTAL)) for t in tokens]
    # Force the widths to sum to exactly TOTAL by adjusting the widest arc.
    widest = max(range(len(widths)), key=lambda i: widths[i])
    widths[widest] += TOTAL - sum(widths)
    cum = [0]
    for w in widths:
        cum.append(cum[-1] + w)
    result = (tokens, cum)
    _cdf_cache[prev] = result
    return result


def _token_at(cum: list, x: int) -> int:
    """Index of the arc containing point x in [0, TOTAL)."""
    return bisect.bisect_right(cum, x) - 1


def _step(prev: str, r: int):
    """Discop decision at one position given context and shared point r.

    Returns (i0, i1, tokens): arc indices for the two copies. i0 == i1 means the
    copies collide (0 bits embeddable here).
    """
    tokens, cum = _dist_cdf(prev)
    i0 = _token_at(cum, r % TOTAL)
    i1 = _token_at(cum, (r + HALF) % TOTAL)
    return i0, i1, tokens


def _next_r(tape: PublicTape) -> int:
    return int.from_bytes(tape.bytes(R_BYTES), "big")


def encode(secret, data: bytes, nonce: bytes = None, key=None) -> tuple:
    """Encode bytes as Discop-sampled cover text. Returns (text, nonce)."""
    if nonce is None:
        nonce = os.urandom(NONCE_LEN)
    owned = key is None
    k = derive_key(secret) if owned else key
    try:
        pbits = bytes_to_bits(pack_payload(data))
        ks_a = Keystream(k, nonce + _DOMAIN_MASK)
        pub = PublicTape(nonce, _DOMAIN_PUBLIC)
        cbits = [p ^ q for p, q in zip(pbits, ks_a.bits(len(pbits)))]

        words: list = []
        prev = LM.START
        pos = 0
        guard = 0
        limit = 100 * len(cbits) + 1000
        while pos < len(cbits):
            guard += 1
            if guard > limit:
                raise RuntimeError("sampler failed to embed (degenerate distribution)")
            r = _next_r(pub)
            i0, i1, tokens = _step(prev, r)
            if i0 == i1:
                word = tokens[i0]                # copies collide: 0 bits here
            else:
                word = tokens[i0] if cbits[pos] == 0 else tokens[i1]
                pos += 1
            words.append(word)
            prev = word

        return " ".join(words), nonce
    finally:
        if owned:
            _wipe(k)


def decode(secret, text: str, nonce: bytes, key=None) -> bytes:
    """Recover bytes from Discop cover text using (secret, nonce)."""
    owned = key is None
    k = derive_key(secret) if owned else key
    try:
        ks_a = Keystream(k, nonce + _DOMAIN_MASK)
        pub = PublicTape(nonce, _DOMAIN_PUBLIC)
        words = text.split()
        cbits: list = []
        prev = LM.START
        for word in words:
            r = _next_r(pub)
            i0, i1, tokens = _step(prev, r)
            if i0 != i1:                          # a bit was embedded here
                # Public segmentation: every key collects the same raw bits.
                cbits.append(1 if word == tokens[i1] else 0)
            prev = word

        pbits = [c ^ q for c, q in zip(cbits, ks_a.bits(len(cbits)))]
        return unpack_payload(bits_to_bytes(pbits))
    finally:
        if owned:
            _wipe(k)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _induced(dist: dict) -> tuple:
    """Distributions two schemes actually emit under uniform input, plus their
    KL to the true P.  Returns (huffman_q, discop_q, kl_huffman, kl_discop).

    Huffman emits symbol s with probability 2^-len(s) (only equals P if P is
    dyadic).  Discop emits width_s / TOTAL == P quantized to 1/TOTAL.
    """
    syms = sorted(dist)
    if len(syms) == 1:
        lengths = {syms[0]: 1}
    else:
        heap = [(dist[s], i, s) for i, s in enumerate(syms)]
        heapq.heapify(heap)
        ctr = len(syms)
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            heapq.heappush(heap, (a[0] + b[0], ctr, (a[2], b[2])))
            ctr += 1
        lengths = {}

        def _walk(node, depth):
            if isinstance(node, str):
                lengths[node] = max(1, depth)
            else:
                _walk(node[0], depth + 1)
                _walk(node[1], depth + 1)

        _walk(heap[0][2], 0)

    huff = {s: 2.0 ** -lengths[s] for s in syms}
    widths = [max(1, round(dist[s] * TOTAL)) for s in syms]
    widest = max(range(len(widths)), key=lambda i: widths[i])
    widths[widest] += TOTAL - sum(widths)
    discop = {s: widths[i] / TOTAL for i, s in enumerate(syms)}

    kl_h = sum(huff[s] * math.log2(huff[s] / dist[s]) for s in syms)
    kl_d = sum(discop[s] * math.log2(discop[s] / dist[s]) for s in syms if discop[s] > 0)
    return huff, discop, kl_h, kl_d


def _kld_report() -> None:
    """Show, analytically (no sampling noise), that Discop's emitted distribution
    equals the true P while Huffman's does not."""
    demo = {"the": 0.70, "of": 0.20, "a": 0.10}
    huff, discop, kl_h, kl_d = _induced(demo)
    print("\nInduced sampling distribution vs true P (skewed toy distribution):")
    print(f"  {'symbol':<8}{'P(true)':>9}{'Discop':>9}{'Huffman':>9}")
    for s in sorted(demo, key=lambda w: demo[w], reverse=True):
        print(f"  {s:<8}{demo[s]:>9.3f}{discop[s]:>9.3f}{huff[s]:>9.3f}")
    print(f"  KL(Discop || P)  = {kl_d:.2e} bits   (fixed-point quantization floor)")
    print(f"  KL(Huffman || P) = {kl_h:.4f} bits   (structural: P is not dyadic)")

    total = kd = kh = 0.0
    for ctx, cnt in LM._counts.items():
        w = sum(cnt.values())
        _, _, klh, kld = _induced(LM.distribution(ctx))
        kh += w * klh
        kd += w * kld
        total += w
    print("\nSame metric on the bigram LM, corpus-weighted over all contexts:")
    print(f"  KL(Discop || P)  = {kd / total:.2e} bits/token   (-> 0, provably)")
    print(f"  KL(Huffman || P) = {kh / total:.4f} bits/token   (structural distortion)")


def main() -> None:
    print(f"KDF        : {'Argon2id (argon2-cffi)' if _HAVE_ARGON2 else 'scrypt (stdlib fallback)'}")
    print(f"LM         : bigram over {len(LM.vocab)}-word corpus; sampler = Discop rotation")
    print(f"Precision  : {PRECISION}-bit fixed-point CDF (TOTAL = 2^{PRECISION})")

    secret = "correct horse battery staple"
    message = b"Rendezvous!"
    print(f"\nsecret  : {secret!r}")
    print(f"message : {message!r}  ({len(message) * 8} bits payload)")

    text, nonce = encode(secret, message)
    words = text.split()
    print(f"\nnonce   : {nonce.hex()}  (distribute with the secret)")
    print(f"cover   : {len(words)} words")
    print(f"  {text}")

    recovered = decode(secret, text, nonce)
    assert recovered == message, f"round-trip failed: {recovered!r}"
    print(f"decoded : {recovered!r}  (correct)")

    wrong = decode("hunter2", text, nonce)
    print(f"\nwrong key -> {wrong!r}")
    print("  (same public segmentation; wrong ks_A unmask -> garbled payload)")

    # Capacity: bits embedded per emitted word.
    pbits = len(bytes_to_bits(pack_payload(message)))
    print(f"\ncapacity: {pbits} payload bits over {len(words)} words "
          f"= {pbits / len(words):.2f} bits/word")
    print("  (single-bit rotation ~1 bit/word; Discop's Huffman recursion")
    print("   recovers the full ~entropy rate with the same zero-KLD security)")

    _kld_report()

    print("\nround-trip OK")


if __name__ == "__main__":
    main()
