#!/usr/bin/env python3
"""File CLI for honey steganography over GPT-2 (paper composition).

Encode a short plaintext into fluent GPT-2 cover text; decode it back with the
same key, nonce, and prompt. Wrong keys open the *same* cover to independent
DistilGPT-2 DTE samples of at least ``min_tokens=10`` tokens (delayed EOS) —
phrase-length honey decoys, not one-word stubs.

Pipeline (matches the preprint)
-------------------------------
  tokens = DistilGPT-2.tokenize(plaintext)     # plaintext model pmt
  seed   = NeuralDTE.encode(tokens)            # ell-bit seed, min_tokens=10
  cover  = GPT-2 Discop.encode_bits(key, seed) # public Pub(ν) + secret ks_A
  seed'  = Discop.decode_bits(key, cover, ν)
  text'  = NeuralDTE.decode(seed')             # = plaintext under the true key

Message length (in DistilGPT-2 tokens) must satisfy
  min_tokens <= len(tokens) <= max_tokens   (default 10..24).

Setup (once)
------------
  python3 -m venv .venv
  ./.venv/bin/pip install -r requirements.txt -r requirements-llm.txt
  ./.venv/bin/python fetch_models.py          # gpt2 + distilgpt2 into models/

Examples
--------
  # encode (writes cover text; prints nonce to stderr if generated)
  ./.venv/bin/python honey_gpt2_cli.py \\
      --key 'correct-horse' --prompt 'The weather report for this weekend says that' \\
      -i message.txt -o cover.txt

  # decode
  ./.venv/bin/python honey_gpt2_cli.py --decode \\
      --key 'correct-horse' --nonce HEX --prompt 'The weather report for this weekend says that' \\
      -i cover.txt -o recovered.txt

  # stdin / stdout
  echo 'the detective found a clue near the harbour at dawn' | \\
    ./.venv/bin/python honey_gpt2_cli.py --key k --prompt '...' -o cover.txt
"""
from __future__ import annotations

import argparse
import binascii
import os
import sys
import time

import gpt2_discop as G
import neural_dte as E
from keyed_cfg_mimic import NONCE_LEN

DEFAULT_PROMPT = "The weather report for this weekend says that"


def _read_text(path: str | None) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str | None, data: str) -> None:
    if path is None or path == "-":
        sys.stdout.write(data)
        if not data.endswith("\n"):
            sys.stdout.write("\n")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
        if not data.endswith("\n"):
            f.write("\n")


def _parse_nonce(s: str) -> bytes:
    s = s.strip().lower().removeprefix("0x")
    try:
        raw = binascii.unhexlify(s)
    except binascii.Error as exc:
        raise SystemExit(f"nonce must be hex: {exc}") from exc
    if len(raw) != NONCE_LEN:
        raise SystemExit(
            f"nonce must be {NONCE_LEN} bytes ({NONCE_LEN * 2} hex chars); "
            f"got {len(raw)} bytes")
    return raw


def _load_dte():
    E.RAND = E.Rand(seed=None)
    return E.NeuralDTE(
        model_dir="distilgpt2",
        seed_bits=E.SEED_BITS,
        min_tokens=E.MIN_TOKENS,
        max_tokens=E.MAX_TOKENS,
    )


def _tokenize_message(dte: E.NeuralDTE, text: str) -> list[int]:
    text = text.strip()
    if not text:
        raise SystemExit("plaintext is empty")
    ids = dte.tok.encode(text, add_special_tokens=False)
    if len(ids) < dte.min_tokens:
        raise SystemExit(
            f"plaintext tokenizes to {len(ids)} DistilGPT-2 tokens; "
            f"need at least min_tokens={dte.min_tokens} "
            f"(delayed EOS — every wrong-key decoy is also ≥{dte.min_tokens} tokens).\n"
            f"Hint: ./.venv/bin/python honey_gpt2_cli.py --sample")
    if len(ids) > dte.max_tokens:
        raise SystemExit(
            f"plaintext tokenizes to {len(ids)} tokens; "
            f"maximum is max_tokens={dte.max_tokens} for this DTE.\n"
            f"Shorten the message or split it across several covers.")
    return ids


def _require_encodable(dte: E.NeuralDTE, ids: list[int]) -> None:
    """Messages must lie on a nucleus path of pmt (paper DTE message space)."""
    try:
        dte.leaf(ids)
    except ValueError as exc:
        raise SystemExit(
            f"plaintext is not encodable under the DistilGPT-2 DTE ({exc}).\n"
            f"Honey deniability only covers messages in the plaintext model: each "
            f"token must stay inside the per-step nucleus (top_p={dte.top_p}, "
            f"top_k={dte.top_k}). Arbitrary file bytes / rare words often fail.\n"
            f"Draw a valid phrase with:\n"
            f"  ./.venv/bin/python honey_gpt2_cli.py --sample\n"
            f"then edit lightly, or use that sample as your message.") from exc


def cmd_sample(args) -> int:
    """Emit one random pmt phrase (≥ min_tokens) that is guaranteed encodable."""
    dte = _load_dte()
    for _ in range(64):
        seed = E.RAND.bits(dte.L)
        ids = dte.decode(seed)
        if len(ids) < dte.min_tokens:
            continue
        try:
            dte.leaf(ids)
        except ValueError:
            continue
        text = dte.text(ids)
        _write_text(args.output, text)
        print(
            f"sampled {len(ids)} tokens (min_tokens={dte.min_tokens}, "
            f"max_tokens={dte.max_tokens})",
            file=sys.stderr)
        return 0
    raise SystemExit("failed to sample an encodable message; retry")


def cmd_encode(args) -> int:
    dte = _load_dte()
    ids = _tokenize_message(dte, _read_text(args.input))
    _require_encodable(dte, ids)
    seed = dte.encode(ids)
    assert len(seed) == dte.L

    if args.nonce:
        nonce = _parse_nonce(args.nonce)
    else:
        nonce = os.urandom(NONCE_LEN)
        print(f"nonce: {nonce.hex()}", file=sys.stderr)

    # Cover budget: leave margin under GPT-2's 1024-token context.
    _, tok = G._load()
    prompt_len = len(tok(args.prompt, return_tensors="pt").input_ids[0])
    n_cover = args.n_cover or max(64, 1024 - prompt_len - 8)
    if n_cover * 0.88 < dte.L:  # rough capacity hint
        print(
            f"warning: n_cover={n_cover} may be tight for ell={dte.L} "
            f"(~0.88 bits/tok typical); retries will rotate the nonce",
            file=sys.stderr)

    last_err = None
    t0 = time.time()
    for attempt in range(1, args.retries + 1):
        try:
            # Fresh nonce each retry so Pub (and capacity draw) changes.
            if attempt > 1:
                nonce = os.urandom(NONCE_LEN)
                print(f"retry {attempt}/{args.retries}; new nonce: {nonce.hex()}",
                      file=sys.stderr)
            text, nonce = G.encode_bits(
                args.key, seed, prompt=args.prompt, nonce=nonce,
                n_cover=n_cover, max_new_tokens=n_cover)
            _write_text(args.output, text)
            print(
                f"encoded {len(ids)} pmt-tokens → {n_cover} cover tokens "
                f"in {time.time() - t0:.1f}s "
                f"(min_tokens={dte.min_tokens}, ell={dte.L})",
                file=sys.stderr)
            print(f"nonce: {nonce.hex()}", file=sys.stderr)
            print(f"prompt: {args.prompt!r}", file=sys.stderr)
            return 0
        except RuntimeError as exc:
            last_err = exc
            continue
    raise SystemExit(f"encode failed after {args.retries} attempts: {last_err}")


def cmd_decode(args) -> int:
    if not args.nonce:
        raise SystemExit("decode requires --nonce (hex)")
    nonce = _parse_nonce(args.nonce)
    dte = _load_dte()
    cover = _read_text(args.input)
    seed = G.decode_bits(
        args.key, cover, nonce, dte.L, prompt=args.prompt)
    ids = dte.decode(seed)
    text = dte.text(ids)
    _write_text(args.output, text)
    print(
        f"decoded {len(ids)} tokens "
        f"(≥{dte.min_tokens} by delayed EOS under any key)",
        file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Encode/decode short messages as GPT-2 honey stegotext "
                    "(DistilGPT-2 DTE, min_tokens=10, public Pub segmentation).")
    ap.add_argument("--decode", action="store_true",
                    help="decode cover text → plaintext (default: encode)")
    ap.add_argument("--sample", action="store_true",
                    help="write one random encodable pmt phrase and exit "
                         "(no --key needed)")
    ap.add_argument("--key", default=None,
                    help="long-term secret (passphrase); required unless --sample")
    ap.add_argument("--nonce", default=None,
                    help=f"{NONCE_LEN * 2}-char hex nonce (required to decode; "
                         "generated on encode if omitted)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="public cover prompt (must match on decode)")
    ap.add_argument("-i", "--input", default="-",
                    help="input file, or - for stdin (default: -)")
    ap.add_argument("-o", "--output", default="-",
                    help="output file, or - for stdout (default: -)")
    ap.add_argument("--n-cover", type=int, default=None,
                    help="fixed continuation token budget (default: fit GPT-2 context)")
    ap.add_argument("--retries", type=int, default=24,
                    help="capacity retries with fresh nonce on encode (default: 24)")
    args = ap.parse_args(argv)

    models = E.MODELS
    for name in ("gpt2", "distilgpt2"):
        if not (models / name).is_dir():
            raise SystemExit(
                f"missing models/{name}/ — run: "
                f"./.venv/bin/python fetch_models.py")

    if args.sample:
        return cmd_sample(args)
    if not args.key:
        raise SystemExit("--key is required unless --sample")
    if args.decode:
        return cmd_decode(args)
    return cmd_encode(args)


if __name__ == "__main__":
    raise SystemExit(main())
