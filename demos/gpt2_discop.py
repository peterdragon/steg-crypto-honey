#!/usr/bin/env python3
"""Discop over a real neural LM (GPT-2) — the neural upgrade of discop_mimic.py.

Plugs GPT-2-124M in as the sampler so cover text is fluent English. Used by
``honey_gpt2_cli.py``. Same distribution-copy idea as ``discop_mimic.py``;
only the distribution source (nucleus-truncated next-token probs) and alphabet
(sub-word BPE) change.

Nucleus truncation caveat
-------------------------
Discop's zero-KLD guarantee is versus the SAMPLER's own (quantized) target
distribution. To bound the effective vocabulary (50257 tokens) to something
where a 32-bit fixed-point CDF resolves every arc distinctly, we nucleus
(top-p) truncate and renormalize before laying out arcs. That truncation is a
small, separate, and measurable source of KL divergence versus the model's
TRUE (untruncated) distribution -- exactly the "truncation KL" a real deployment
would report alongside the (provably ~0) sampling KL.

Tokenization round-trip note
-----------------------------
The cover text must be re-tokenized by the receiver to recover token ids.
GPT-2's byte-level BPE id sequences are NOT always round-trip-stable: decoding
an arbitrary id sequence to text and re-encoding that text from scratch can
merge differently at a boundary (observed empirically roughly once per ~20
generations of ~150 tokens here) -- and because that shifts the receiver's
continuation-id count from that point on, a single bad merge corrupts every
subsequent position, not just one bit.

Fix (Meteor's "self-tokenizing" bias, Kaptchuk et al. CCS 2021): before
committing to a candidate token at each step, check whether appending it to
the sequence so far survives a decode/re-encode round trip. This check is a
deterministic function of the (public) token ids generated so far, so the
receiver -- which reconstructs the same ids step by step -- computes the
identical check with no extra information crossing the channel. If a
rotation-copy candidate is unsafe, that position degrades to a no-bit
position (like a copy collision) rather than risking corruption; capacity
drops slightly but correctness becomes structural rather than assumed. See
_self_tokenizes() / _safe_fallback() below.

Run:  ./.venv/bin/python gpt2_discop.py     (needs torch + transformers; the
      weights live under demos/models/{gpt2,distilgpt2}/ and are git-ignored.
      Populate a fresh clone with:  python fetch_models.py)
"""

import hashlib
import os
from pathlib import Path

import torch

from _common import bytes_to_bits, bits_to_bytes
from keyed_cfg_mimic import (
    NONCE_LEN, _wipe, derive_key, pack_payload, unpack_payload, Keystream,
    _HAVE_ARGON2,
)

PRECISION = 32
TOTAL = 1 << PRECISION
HALF = TOTAL >> 1
R_BYTES = PRECISION // 8
TOP_P = 0.92

_DOMAIN_MASK = b"\x00"      # secret: payload mask ks_A = F(mk, nonce||MASK)
_DOMAIN_PUBLIC = b"\x02"    # public: segmentation + selector tape from nonce only


class PublicTape:
    """Keyless SHA-256 counter mode driven only by the public nonce.

    Supplies the per-step rotation coins and filler selector bits so that
    embedding *positions* (and therefore the raw bit count) are identical for
    every key -- closing the parse-count oracle of a key-dependent sampler tape.
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


def _next_r_pub(tape: PublicTape) -> int:
    return int.from_bytes(tape.bytes(R_BYTES), "big")


def _next_bit_pub(tape: PublicTape) -> int:
    return tape.bytes(1)[0] & 1

MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_PROMPT = "The weather report for this weekend says that"

_model = None
_tok = None


def _load(model_dir: str = "gpt2"):
    global _model, _tok
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = str(MODELS_DIR / model_dir)
        _tok = AutoTokenizer.from_pretrained(path)
        _model = AutoModelForCausalLM.from_pretrained(path)
        _model.eval()
        torch.manual_seed(0)  # dropout is off in eval() anyway; belt & braces
    return _model, _tok


# ---------------------------------------------------------------------------
# Autoregressive nucleus sampler with KV-cache stepping
# ---------------------------------------------------------------------------

class GPT2State:
    __slots__ = ("past", "logits", "ids")

    def __init__(self, past, logits, ids):
        self.past = past
        self.logits = logits
        self.ids = ids


def _prime(prompt: str) -> GPT2State:
    model, tok = _load()
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, use_cache=True)
    return GPT2State(out.past_key_values, out.logits[0, -1], ids[0].tolist())


def _advance(state: GPT2State, token_id: int) -> GPT2State:
    model, _ = _load()
    t = torch.tensor([[token_id]])
    with torch.no_grad():
        out = model(t, past_key_values=state.past, use_cache=True)
    return GPT2State(out.past_key_values, out.logits[0, -1], state.ids + [token_id])


def nucleus_dist(logits: torch.Tensor, top_p: float = TOP_P) -> dict:
    """Top-p truncated, renormalized P(next token) as {token_id: prob}."""
    probs = torch.softmax(logits, dim=-1)
    sorted_p, sorted_i = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=0)
    k = int(torch.searchsorted(cum, torch.tensor(top_p)).item()) + 1
    k = max(1, min(k, sorted_p.numel()))
    idx = sorted_i[:k].tolist()
    p = sorted_p[:k].tolist()
    total = sum(p)
    return {i: pi / total for i, pi in zip(idx, p)}


def _dist_cdf(dist: dict) -> tuple:
    """Fixed-point CDF: (tokens, cum) with cum[-1] == TOTAL. Sorted by token id
    so encode/decode canonically agree even if torch.sort tie-breaking ever
    differed across runs."""
    tokens = sorted(dist)
    widths = [max(1, round(dist[t] * TOTAL)) for t in tokens]
    widest = max(range(len(widths)), key=lambda i: widths[i])
    widths[widest] += TOTAL - sum(widths)
    cum = [0]
    for w in widths:
        cum.append(cum[-1] + w)
    return tokens, cum


def _token_at(cum: list, x: int) -> int:
    import bisect
    return bisect.bisect_right(cum, x) - 1


def _next_r(ks) -> int:
    return int.from_bytes(ks.bytes(R_BYTES), "big")


def _self_tokenizes(tok, ids: list) -> bool:
    """True iff decode(ids) then re-encode reproduces ids exactly.

    Deterministic given the (public) ids, so encoder and receiver each compute
    it independently -- no extra information needs to cross the channel.
    """
    text = tok.decode(ids)
    return tok(text, add_special_tokens=False).input_ids == ids


def _safe_fallback(dist: dict, tok, prefix_ids: list) -> int:
    """Highest-probability nucleus candidate that keeps the sequence
    self-tokenizing. Widens to a few GPT-2 structural tokens, then as a last
    resort returns nucleus argmax (residual BPE risk; preferred over aborting
    generation mid-cover)."""
    for t in sorted(dist, key=dist.get, reverse=True):
        if _self_tokenizes(tok, prefix_ids + [t]):
            return t
    for tid in (50256, 220, 198, 32):  # eos, space, newline, ' '
        if _self_tokenizes(tok, prefix_ids + [tid]):
            return tid
    return max(dist, key=dist.get)


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def encode_bits(secret, bits, prompt: str = DEFAULT_PROMPT, nonce: bytes = None,
                key=None, top_p: float = TOP_P, max_new_tokens: int = 400,
                n_cover: int = None) -> tuple:
    """Embed a fixed-length bitstring as GPT-2 cover text continuing `prompt`.

    Segmentation is *public*: rotation coins and filler selector bits come from
    PublicTape(nonce), so every key agrees on which steps embed and on the raw
    bit-count. Secrecy is only in ks_A (payload mask).

    Budget modes
    ------------
    * ``n_cover is None`` (legacy): stop once all payload bits are placed.
    * ``n_cover = N`` (paper): emit exactly N continuation tokens; after the
      payload, embedding steps use public filler bits. Abort if the budget
      ends before the payload is placed.

    Returns (full_text, nonce).
    """
    if not bits:
        raise ValueError("encode_bits: empty bitstring")
    if n_cover is not None and n_cover <= 0:
        raise ValueError("encode_bits: n_cover must be positive")
    if nonce is None:
        nonce = os.urandom(NONCE_LEN)
    owned = key is None
    k = derive_key(secret) if owned else key
    try:
        ks_a = Keystream(k, nonce + _DOMAIN_MASK)
        pub = PublicTape(nonce, _DOMAIN_PUBLIC)
        cbits = [p ^ q for p, q in zip(bits, ks_a.bits(len(bits)))]

        _, tok = _load()
        state = _prime(prompt)
        pos = 0
        nbits = len(cbits)
        steps = 0
        while True:
            if n_cover is None and pos >= nbits:
                break
            if n_cover is not None and steps >= n_cover:
                break
            steps += 1
            if n_cover is None and steps > max_new_tokens:
                raise RuntimeError(
                    f"exceeded max_new_tokens={max_new_tokens} embedding "
                    f"{nbits} bits ({pos} embedded); raise the cap, "
                    f"shorten the payload, or shorten the prompt")
            if len(state.ids) >= 1024:
                raise RuntimeError(
                    f"GPT-2 context exhausted ({len(state.ids)} tokens) with "
                    f"only {pos}/{nbits} bits embedded")
            dist = nucleus_dist(state.logits, top_p)
            tokens, cum = _dist_cdf(dist)
            r = _next_r_pub(pub)
            i0 = _token_at(cum, r % TOTAL)
            i1 = _token_at(cum, (r + HALF) % TOTAL)
            if i0 == i1:
                cand = tokens[i0]
                tok_id = cand if _self_tokenizes(tok, state.ids + [cand]) \
                    else _safe_fallback(dist, tok, state.ids)
            else:
                safe0 = _self_tokenizes(tok, state.ids + [tokens[i0]])
                safe1 = _self_tokenizes(tok, state.ids + [tokens[i1]])
                if safe0 and safe1:
                    # Always burn a public selector bit so encode/decode stay
                    # aligned; override with payload while bits remain.
                    bit_pub = _next_bit_pub(pub)
                    if pos < nbits:
                        bit = cbits[pos]
                        pos += 1
                    elif n_cover is not None:
                        bit = bit_pub
                    else:
                        break
                    tok_id = tokens[i0] if bit == 0 else tokens[i1]
                elif safe0:
                    tok_id = tokens[i0]
                elif safe1:
                    tok_id = tokens[i1]
                else:
                    tok_id = _safe_fallback(dist, tok, state.ids)
            state = _advance(state, tok_id)

        if pos < nbits:
            raise RuntimeError(
                f"exceeded budget embedding {nbits} bits ({pos} embedded); "
                f"raise n_cover/max_new_tokens, shorten the payload, or "
                f"shorten the prompt")

        full_text = tok.decode(state.ids)
        return full_text, nonce
    finally:
        if owned:
            _wipe(k)


def decode_bits(secret, text: str, nonce: bytes, nbits: int,
                prompt: str = DEFAULT_PROMPT, key=None,
                top_p: float = TOP_P, return_meta: bool = False):
    """Recover a fixed-length bitstring from GPT-2 cover text.

    Replays the *public* segmentation tape, so every key collects the same raw
    bitstring w from T; secrecy is only ks_A unmasking (seed' = w ⊕ ks_A').
    Pad/truncate to nbits for a total map (legacy short transcripts).
    """
    if nbits <= 0:
        raise ValueError("decode_bits: nbits must be positive")
    owned = key is None
    k = derive_key(secret) if owned else key
    try:
        _, tok = _load()
        prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        all_ids = tok(text, return_tensors="pt").input_ids[0].tolist()
        if all_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "re-tokenized text does not start with the expected prompt "
                "tokens (BPE boundary mismatch) -- see module docstring")
        continuation_ids = all_ids[len(prompt_ids):]

        ks_a = Keystream(k, nonce + _DOMAIN_MASK)
        pub = PublicTape(nonce, _DOMAIN_PUBLIC)

        state = _prime(prompt)
        cbits: list = []
        for tok_id in continuation_ids:
            dist = nucleus_dist(state.logits, top_p)
            tokens, cum = _dist_cdf(dist)
            r = _next_r_pub(pub)
            i0 = _token_at(cum, r % TOTAL)
            i1 = _token_at(cum, (r + HALF) % TOTAL)
            if i0 != i1:
                safe0 = _self_tokenizes(tok, state.ids + [tokens[i0]])
                safe1 = _self_tokenizes(tok, state.ids + [tokens[i1]])
                if safe0 and safe1:
                    _next_bit_pub(pub)  # burn public selector (encode always drew it)
                    cbits.append(1 if tok_id == tokens[i1] else 0)
            state = _advance(state, tok_id)

        observed = len(cbits)
        if len(cbits) < nbits:
            cbits = cbits + [0] * (nbits - len(cbits))
        else:
            cbits = cbits[:nbits]
        bits = [c ^ q for c, q in zip(cbits, ks_a.bits(nbits))]
        if return_meta:
            return bits, {"observed_bits": observed,
                          "padded_bits": max(0, nbits - observed),
                          "truncated": observed > nbits,
                          "n_cover": len(continuation_ids)}
        return bits
    finally:
        if owned:
            _wipe(k)


def encode(secret, data: bytes, prompt: str = DEFAULT_PROMPT, nonce: bytes = None,
           key=None, top_p: float = TOP_P, max_new_tokens: int = 400) -> tuple:
    """Encode bytes as GPT-2-sampled cover text continuing `prompt`.

    Length-prefixed via pack_payload (legacy demo API). For DTE seeds use
    encode_bits instead -- a length header would destroy length-hiding and is
    incompatible with a fixed ell-bit seed space.
    """
    return encode_bits(secret, bytes_to_bits(pack_payload(data)), prompt=prompt,
                       nonce=nonce, key=key, top_p=top_p,
                       max_new_tokens=max_new_tokens)


def decode(secret, text: str, nonce: bytes, prompt: str = DEFAULT_PROMPT,
           key=None, top_p: float = TOP_P) -> bytes:
    """Recover bytes from GPT-2 cover text using (secret, nonce, prompt).

    Legacy pack_payload API: extracts every bit the continuation carries, then
    unpacks the length header. Prefer decode_bits when the payload length is
    known a priori (DTE seeds).
    """
    # Extract without truncating: walk once, XOR the full stream, then unpack.
    owned = key is None
    k = derive_key(secret) if owned else key
    try:
        # Reuse decode_bits with a generous nbits by extracting raw then
        # re-deriving: call the walk via a large pad and strip. Simpler to
        # duplicate the short unpack path against encode_bits' dual.
        _, tok = _load()
        prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        all_ids = tok(text, return_tensors="pt").input_ids[0].tolist()
        if all_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "re-tokenized text does not start with the expected prompt "
                "tokens (BPE boundary mismatch) -- see module docstring")
        continuation_ids = all_ids[len(prompt_ids):]
        ks_a = Keystream(k, nonce + _DOMAIN_MASK)
        pub = PublicTape(nonce, _DOMAIN_PUBLIC)
        state = _prime(prompt)
        cbits: list = []
        for tok_id in continuation_ids:
            dist = nucleus_dist(state.logits, top_p)
            tokens, cum = _dist_cdf(dist)
            r = _next_r_pub(pub)
            i0 = _token_at(cum, r % TOTAL)
            i1 = _token_at(cum, (r + HALF) % TOTAL)
            if i0 != i1:
                safe0 = _self_tokenizes(tok, state.ids + [tokens[i0]])
                safe1 = _self_tokenizes(tok, state.ids + [tokens[i1]])
                if safe0 and safe1:
                    _next_bit_pub(pub)
                    cbits.append(1 if tok_id == tokens[i1] else 0)
            state = _advance(state, tok_id)
        pbits = [c ^ q for c, q in zip(cbits, ks_a.bits(len(cbits)))]
        return unpack_payload(bits_to_bytes(pbits))
    finally:
        if owned:
            _wipe(k)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"KDF        : {'Argon2id (argon2-cffi)' if _HAVE_ARGON2 else 'scrypt (stdlib fallback)'}")
    print(f"Sampler    : GPT-2 (124M), nucleus top-p={TOP_P}, {PRECISION}-bit fixed-point CDF")

    secret = "correct horse battery staple"
    message = b"Rendezvous!"
    prompt = DEFAULT_PROMPT
    print(f"\nsecret  : {secret!r}")
    print(f"prompt  : {prompt!r}  (public, sent/known out of band like the nonce)")
    print(f"message : {message!r}  ({len(message) * 8} bits payload)")

    text, nonce = encode(secret, message, prompt)
    print(f"\nnonce   : {nonce.hex()}")
    print(f"cover   :\n  {text}")

    recovered = decode(secret, text, nonce, prompt)
    assert recovered == message, f"round-trip failed: {recovered!r}"
    print(f"\ndecoded : {recovered!r}  (correct)")

    wrong = decode("hunter2", text, nonce, prompt)
    print(f"\nwrong key -> {wrong!r}")
    print("  (same public segmentation; wrong ks_A unmask -> garbled payload bits,")
    print("   still valid GPT-2 vocabulary tokens -- honey at the bit/DTE layer)")

    print("\nround-trip OK")


if __name__ == "__main__":
    main()
