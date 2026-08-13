# Teaching demos

Small, mostly standard-library programs that walk the same path as the
LinkedIn article and the preprint: hide bits in fluent text, then make every
wrong key decode to a *plausible* message rather than garbage.

These are **learning demos** and a small GPT-2 CLI — enough to see the ideas
in the write-ups and preprint, not a full evaluation suite.

Undetectability in the paper is **relative to a deployed cover sampler**, not
arbitrary platform traffic. Message-recovery deniability holds when the decoy
model matches the message source; it is **not** a claim that real English and
decoys stay inseparable under every judge.

## Quick start (after `git clone`)

**Stdlib demos** — no virtualenv needed:

```bash
cd demos
python3 run_all.py
```

Optional Argon2id KDF (else `scrypt` fallback):

```bash
python3 -m pip install -r requirements.txt
```

**GPT-2 CLI** — `.venv/` is not in git; create it once:

```bash
cd demos
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt -r requirements-llm.txt
./.venv/bin/python fetch_models.py    # downloads ~860 MB into models/ (git-ignored)
```

Then use `./.venv/bin/python honey_gpt2_cli.py …` as in the section below.

## Story order (matches the articles)

| Demo | What you should see |
|------|---------------------|
| `cfg_mimic.py` | Wayner-style grammar: bits choose productions; output is grammatical English. |
| `keyed_cfg_mimic.py` | Same idea with a passphrase + public nonce; wrong secret still parses. |
| `weighted_cfg_mimic.py` | Frequency-weighted productions (statistical mimicry). |
| `lm_mimic.py` | Bigram LM + Huffman: more fluent, but Huffman **leaks** (~0.013 bits/word). |
| `discop_mimic.py` | Distribution-copy sampling + **public** segmentation tape (`Pub` from the nonce); secret mask `ks_A` only. |
| `inverse_sampling_dte.py` | Juels–Ristenpart inverse-sampling DTE (toy). |
| `honey_encryption.py` | DTE-then-encrypt: wrong passwords → plausible decoy PINs + KDF cost. |
| `ngram_nle.py` / `pcfg_nle.py` | Natural-language DTEs (Chatterjee-style). |
| `e2_decoy_plausibility.py` | **LinkedIn punchline:** seal one message, print fluent wrong-key decoys; optional matched/mismatched AUC. |

### See a wrong-key decoy in one command

```bash
python3 e2_decoy_plausibility.py --message "she was very happy" --decoys 10
```

## Design notes (aligned with the preprint)

- **Public segmentation.** Sampling coins come from `Pub = H(ν ‖ "public")`, not a
  second secret keystream. Every key agrees on embedding counts (no parse-count
  oracle). Secrecy lives in `ks_A` masking the DTE seed — classical DTE-then-OTP
  after public extraction.
- **No MAC on the honey ciphertext.** A tag would let an offline searcher
  recognise the correct key. Integrity belongs elsewhere, under a separate key.
- **Teaching vs deployment.** No claim of platform-level cover, coercion
  resistance, or side-channel freedom. See the preprint's "Not claimed" list.

## GPT-2 file CLI (paper composition)

`honey_gpt2_cli.py` seals a **short** plaintext into fluent GPT-2 cover text and
opens it again. Wrong keys yield independent phrase-length decoys
(`min_tokens=10`), not one-word stubs.

**Message space.** The plaintext must be a DistilGPT-2 DTE message: length
10–24 tokens *and* every token on the per-step nucleus path (same law the
decoys use). Arbitrary file text often fails — that is the honey model, not a
bug. Draw a valid phrase with `--sample`.

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt -r requirements-llm.txt
./.venv/bin/python fetch_models.py    # ~860 MB into models/ (git-ignored)

# draw an encodable message, then seal it
./.venv/bin/python honey_gpt2_cli.py --sample -o message.txt
./.venv/bin/python honey_gpt2_cli.py \
    --key 'correct-horse' \
    --prompt 'The weather report for this weekend says that' \
    -i message.txt -o cover.txt          # nonce printed on stderr

# decode (reuse nonce + prompt)
./.venv/bin/python honey_gpt2_cli.py --decode \
    --key 'correct-horse' --nonce HEX_FROM_STDERR \
    --prompt 'The weather report for this weekend says that' \
    -i cover.txt -o recovered.txt

# wrong key → ≥10-token honey English on stdout
./.venv/bin/python honey_gpt2_cli.py --decode \
    --key 'wrong-key' --nonce HEX_FROM_STDERR \
    --prompt 'The weather report for this weekend says that' \
    -i cover.txt
```

Needs a machine that can run GPT-2 (CPU is fine, slower). This is the LinkedIn
“weather report” path; the stdlib demos above stay the lightweight teaching set.

**Flow diagram (editable Mermaid):** [`honey-flow.md`](honey-flow.md) — encode,
true-key decode, and wrong-key honey path. Edit the source or paste into
[mermaid.live](https://mermaid.live).

## Licence

Code under the repository root `LICENSE` (MIT). The preprint PDF and markdown
write-ups remain **CC BY 4.0** — see `../LICENSE-NOTE.md`.
