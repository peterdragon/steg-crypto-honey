#!/usr/bin/env python3
"""Randomised arithmetic-coding DTE over DistilGPT-2 (plaintext model).

Maps short token sequences to fixed-length uniform seeds and back. Wrong-key
honey decryption feeds a uniform seed into ``decode``, so decoys are fresh
samples from the same law as legitimate messages (including delayed EOS so
every draw is at least ``min_tokens`` long).

Used by ``honey_gpt2_cli.py``. Requires ``requirements-llm.txt`` and local
weights under ``models/distilgpt2/`` (see ``fetch_models.py``).
"""
from __future__ import annotations

import bisect
import random
import secrets
from pathlib import Path

import torch

from _common import bits_to_int, int_to_bits

MODELS = Path(__file__).parent / "models"
EOS = -1  # sentinel symbol (distinct from any token id)

SEED_BITS = 896         # fixed-length seed → ciphertext hides message length
STEP_TOTAL = 1 << 32    # per-step fixed-point denominator
P_EOS = 0.06            # EOS mass once allowed
MAX_TOKENS = 24         # length cap
MIN_TOKENS = 10         # forbid EOS until this many tokens (phrase-length decoys)
TOP_P = 0.90
TOP_K = 256


class Rand:
    """Cryptographic randomness by default; optional seeded PRNG for demos."""

    def __init__(self, seed=None):
        self._r = random.Random(seed) if seed is not None else None

    def below(self, n):
        return self._r.randrange(n) if self._r else secrets.randbelow(n)

    def bits(self, n):
        if self._r:
            return [self._r.getrandbits(1) for _ in range(n)]
        return [secrets.randbits(1) for _ in range(n)]


RAND = Rand()


def _load(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = str(MODELS / model_dir)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    model.eval()
    return model, tok


def _fixed_point_widths(probs, total):
    n = len(probs)
    if total < n:
        raise ValueError(f"STEP_TOTAL={total} < support={n}")
    widths = [max(1, int(p * total)) for p in probs]
    diff = total - sum(widths)
    order = sorted(range(n), key=lambda i: (-probs[i], i))
    idx = 0
    while diff > 0:
        widths[order[idx % n]] += 1
        diff -= 1
        idx += 1
    idx = 0
    while diff < 0:
        i = order[idx % n]
        if widths[i] > 1:
            widths[i] -= 1
            diff += 1
        idx += 1
    return widths


class NeuralDTE:
    """Randomised arithmetic-coding DTE over an autoregressive neural LM.

    Next-token law is nucleus-truncated with EOS carrying ``p_eos`` only after
    the first ``min_tokens`` steps. ``decode`` of a uniform seed samples that
    law; ``encode`` picks a uniform point in the message's leaf interval.
    """

    def __init__(self, model_dir="distilgpt2", seed_bits=SEED_BITS,
                 step_total=STEP_TOTAL, p_eos=P_EOS, max_tokens=MAX_TOKENS,
                 min_tokens=MIN_TOKENS, top_p=TOP_P, top_k=TOP_K):
        self.model, self.tok = _load(model_dir)
        self.bos = self.tok.eos_token_id if self.tok.eos_token_id is not None else 50256
        self.L = seed_bits
        self.T = step_total
        self.p_eos = p_eos
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.max_support = 0
        self.S_bound = self.top_k + 1
        if not (0 <= self.min_tokens < self.max_tokens):
            raise ValueError(
                f"need 0 <= min_tokens < max_tokens "
                f"(got min_tokens={self.min_tokens}, max_tokens={self.max_tokens})")
        b_dte = self.T.bit_length() - 1
        if self.L - b_dte * self.max_tokens < 2:
            raise ValueError(
                "head-room condition h = ell - b_dte*n_max >= 2 violated "
                f"(ell={self.L}, b_dte={b_dte}, n_max={self.max_tokens})")

    def _eos_allowed(self, step: int) -> bool:
        return step >= self.min_tokens

    def _prime(self):
        ids = torch.tensor([[self.bos]])
        with torch.no_grad():
            out = self.model(ids, use_cache=True)
        return out.past_key_values, out.logits[0, -1]

    def _advance(self, past, tid):
        t = torch.tensor([[tid]])
        with torch.no_grad():
            out = self.model(t, past_key_values=past, use_cache=True)
        return out.past_key_values, out.logits[0, -1]

    def _dist(self, logits, allow_eos):
        probs = torch.softmax(logits, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=0)
        k = int(torch.searchsorted(cum, torch.tensor(self.top_p)).item()) + 1
        k = max(1, min(k, sp.numel(), self.top_k))
        toks = si[:k].tolist()
        pr = sp[:k].tolist()
        Z = sum(pr)
        pairs = sorted(zip(toks, pr), key=lambda x: x[0])
        toks = [t for t, _ in pairs]
        pr = [p / Z for _, p in pairs]
        if allow_eos:
            pr = [p * (1.0 - self.p_eos) for p in pr] + [self.p_eos]
            toks = toks + [EOS]
        widths = _fixed_point_widths(pr, self.T)
        self.max_support = max(self.max_support, len(widths))
        cumw = [0]
        for w in widths:
            cumw.append(cumw[-1] + w)
        return toks, cumw

    def decode(self, seed_bits):
        value = bits_to_int(seed_bits[:self.L])
        lo, hi = 0, 1 << self.L
        past, logits = self._prime()
        out = []
        for step in range(self.max_tokens):
            toks, cum = self._dist(logits, allow_eos=self._eos_allowed(step))
            rng = hi - lo
            if rng <= 0:
                break
            cumv = ((value - lo + 1) * self.T - 1) // rng
            cumv = max(0, min(cumv, self.T - 1))
            i = bisect.bisect_right(cum, cumv) - 1
            i = max(0, min(i, len(cum) - 2))
            sym = toks[i]
            new_hi = lo + rng * cum[i + 1] // self.T
            new_lo = lo + rng * cum[i] // self.T
            lo, hi = new_lo, new_hi
            if sym == EOS:
                break
            out.append(sym)
            past, logits = self._advance(past, sym)
        return out

    def leaf(self, token_ids):
        """Half-open seed interval [lo, hi) that decodes to exactly token_ids."""
        lo, hi = 0, 1 << self.L
        past, logits = self._prime()
        if len(token_ids) > self.max_tokens:
            raise ValueError("message longer than n_max; not encodable")
        if len(token_ids) < self.min_tokens:
            raise ValueError(
                f"message shorter than min_tokens={self.min_tokens}; not encodable "
                "(EOS was forbidden for the first min_tokens steps)")
        syms = list(token_ids)
        if len(syms) < self.max_tokens:
            syms.append(EOS)
        for step, sym in enumerate(syms):
            toks, cum = self._dist(logits, allow_eos=self._eos_allowed(step))
            if sym not in toks:
                raise ValueError("token outside per-step nucleus; not encodable")
            i = toks.index(sym)
            rng = hi - lo
            new_hi = lo + rng * cum[i + 1] // self.T
            new_lo = lo + rng * cum[i] // self.T
            lo, hi = new_lo, new_hi
            if sym != EOS:
                past, logits = self._advance(past, sym)
        if hi - lo < 1:
            raise ValueError("interval underflow; raise SEED_BITS")
        return lo, hi

    def encode(self, token_ids):
        lo, hi = self.leaf(token_ids)
        return int_to_bits(lo + RAND.below(hi - lo), self.L)

    def text(self, token_ids):
        return self.tok.decode(token_ids)
