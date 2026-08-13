# Honey steganography: how it actually works

*A technical cut, for engineers. This sits between the [LinkedIn write-up](linkedin-article.md), which tells the story, and the academic paper, which has the proofs. Here I just want to explain the machine: what goes in, what comes out, and why it has the two properties I claim. No proofs, but enough detail that you could rebuild it.*

## The two problems, solved at once

Two different attackers:

1. A **passive warden** who watches your traffic and asks: is there a hidden message in here at all? Modern language-model steganography already beats this one. If you sample cover text the right way, the output is statistically identical to ordinary model output, and the warden's best detector is a coin flip.

2. An **offline key-searcher** who has intercepted the traffic and now tries keys at leisure. Every ordinary scheme fails here: a wrong key produces garbage, so the searcher knows the moment they hit the right key. Honey encryption beats this one by making every wrong key decrypt to a plausible fake.

The ingredients are known. What this project adds is stating both games together, closing the parse-count oracle with a public tape, and measuring decoy plausibility. After that they compose: invisible to the first attacker and deniable to the second, in the same message — without a new cipher.

## The pipeline, end to end

Sender has a shared secret and picks a fresh public nonce per message. Then:

1. **Message → seed.** Run the plaintext through a *distribution-transforming encoder* (DTE). Out comes a fixed-length, uniform-looking seed (`ℓ = 896` bits in the neural slice).
2. **Whiten the seed.** XOR it with a **secret** mask `ks_A`, derived from the secret and nonce. Call the result the payload bits. Secrecy lives **only** here.
3. **Embed the payload.** Drive a distribution-preserving sampler (Discop) over GPT-2. Sampling coins come from a **public** tape `Pub = H(ν ‖ "public")` derived from the nonce alone — not a second secret keystream. The payload bits steer *which* sample you take at each embedding step. Out comes fluent cover text of a **fixed token budget**.
4. **Send** the cover text plus the public nonce (like an IV). There is no length header in the payload: the seed is already fixed-length.

Receiver reconstructs `ks_A` from the secret and nonce, and **the same `Pub` as everyone else**. It reads `w' = ReadBits(T, Pub)` — identical for every key — unmasks `seed' = w' ⊕ ks_A`, and runs the DTE in reverse.

A wrong key still extracts the **same** `w'` (public segmentation closes the parse-count oracle). Unmasking with the wrong `ks_A` yields a uniform seed, and the DTE turns that into a fresh, plausible decoy. After public extraction this is classical DTE-then-OTP — a feature of the composition, not a new cipher.

## Piece 1: the cover channel (Discop, in one picture)

Imagine the model's next-word distribution as a wheel, each word occupying an arc proportional to its probability. Normally you'd throw a dart (a uniform random number) and emit whichever word it lands on. That reproduces the model's distribution exactly.

Discop's trick: make a second copy of the wheel, rotated by half a turn. The next **payload** bit chooses which copy you read (the dart itself comes from the public tape). Because a rotation doesn't change how much arc each word occupies, the *frequency* of each word is untouched — so the emitted text still matches the model exactly. The message hides in which copy you used, and that choice is invisible in the statistics.

Concretely, this is why the measured KL divergence between "text carrying a message" and "ordinary model output" comes out at the noise floor of the fixed-point arithmetic (about 3e-14 bits per token in my runs), rather than the ~0.013 bits per word you get from the older Huffman-coding approach. Huffman forces every word into a power-of-a-half probability; Discop doesn't distort anything.

## Piece 2: the honey layer (a DTE from arithmetic coding)

The DTE is the part that makes wrong keys plausible. It has to do two things:

- `encode(message) → seed`: map a message to a uniform seed.
- `decode(seed) → message`: map any uniform seed back to a message drawn from the message model.

I build it as **randomized arithmetic coding** over a language model of the messages. Arithmetic coding already maps a message to an interval; randomising the choice of point in that interval gives you a uniform seed, and decoding a uniform seed samples the message model. The key property: `decode(uniform)` is distributed like the message model itself. So a wrong key, which hands `decode` a uniform seed, produces a valid sample of "what messages look like". That's your decoy.

This is the honey-encryption idea (Juels & Ristenpart, 2014) applied to the *plaintext inside a stego channel* rather than to a ciphertext you present openly. The natural-language DTE lineage is Chatterjee et al. (2015).

## Mask and public segmentation tape

Secrecy lives in one keystream: **`ks_A`** whitens the DTE seed. Sampling
randomness comes from a **public** tape `Pub = H(ν ‖ "public")` derived from the
nonce alone, so every key agrees on embedding positions and the raw bit-count
(closing a parse-count oracle that key-dependent sampling would admit).
Cover and Stego share the same `Pub`; undetectability does not hide the
sampling coins. Deniability needs `ks_A` uniform and independent of
`(Pub, T)` — classical one-time-pad honey encryption against the shared
recovered word `w' = ReadBits(T, Pub)`.

## The bit that broke: self-tokenizing

The maths assumes you can read the same bits back that you wrote. GPT-2 doesn't work in words, it works in sub-word tokens, and it turns out that decoding a token sequence to text and re-encoding it does **not** always give you back the same tokens. About one generation in nineteen hit this in my tests, and because the receiver replays `Pub` over the re-tokenized text, a single mismatch desynchronises everything after it.

The fix (from Meteor, Kaptchuk et al. 2021) is to check, at each step, that a candidate token survives the decode/re-encode round-trip before you commit to it. Both sender and receiver compute the same check, so they stay in lock-step. It's the least glamorous part of the whole thing and it was the difference between "mathematically perfect" and "actually works".

## What "secure" means here

Two levels, and being honest about which one you're claiming matters:

- With a **truly random public tape** and an independent secret mask, the cover channel is *perfectly* secure in Cachin's information-theoretic sense **relative to the deployed sampler `Cover`**: zero KL, zero advantage for any warden, even one with unlimited compute.
- In practice the mask is a **PRF** of the derived key. Cover and Stego still share the same public `H(ν)`, so that downgrades "perfect" to "computationally secure" unless you can break the PRF. Cross-key deniability additionally treats the key-derivation function as a random oracle — the PRF term is vacuous once you try every key.

The insight I keep coming back to: the security doesn't come from the *message* being random. For any message, even a completely predictable one, the emitted text follows `Cover` exactly, because Cover and Stego use the same public tape; the secret mask only whitens the payload.

## The numbers

From the GPT-2 evaluation in the preprint (N = 100 paired trials for undetectability; G = 150 / 300 decoys, `--seed 1` for deniability):

- Message-vs-channel (stego vs the deployed sampler `Cover`): trained logistic AUC **0.55**, CI **[0.47, 0.63]** — an implementation check; CIs include 0.5. Against *raw* platform text the same detectors are strongly above 0.5 (logistic ~0.85). Undetectability is **relative to Cover**, not the open web.
- Sampling KL vs Cover is 0 by construction; truncation KL ~0.110 vs 0.111 bits/token (message-independent). Discop rotation residual ~3e-14 bits/token.
- Self-tokenizing channel gap: ~1.1e-2 in total variation, the same whether or not a message is present.
- Matched-regime recovery AUC near chance (neural mean-NLL **0.547**, CI **[0.488, 0.604]**). Against held-out English, neural `p_mt` cuts separability from 0.50 (bigram) to about **0.39–0.43**; one-epoch continued pretraining on a frozen train split further cuts **holdout** separability to about **0.10–0.13** (still excluding 0.5). Real-English deniability is **not** claimed.
- Capacity: simplest Discop half-rotation carries roughly one bit per token. You pay for undetectability in bandwidth.

## The honest limit

Deniability is only as strong as your model of "normal messages". The theorem's `ε` is *arithmetic* DTE goodness: how close `decode(uniform)` is to the built-in message model `p_mt` (about `1.4×10^{-6}` at the neural slice). That is not the Gutenberg residual. If real messages look like `p_mt`, wrong-key decoys match the true message and the key-searcher is stuck. If they are richer — fluent human English, say — a strong enough judge can start telling them apart. That model-versus-reality gap is a different quantity; a better neural `p_mt` shrinks it but does not erase it, and can flip which side of the gap the decoys sit on. Real-English deniability is **not** claimed, and neither is **coercion** resistance: if they have the real secret, the real message comes out.

And there's deliberately **no message authentication**. A checksum would let the attacker recognise the correct key offline, which is precisely what we removed. Integrity, if you need it, has to live elsewhere under a separate key.

## If you want to poke at it

The private research tree has the evaluation harnesses named in the paper (`e1_undetectability.py`, `e2_neural.py`, `e2e_composition.py`, `train_pmt.py`) plus `verify_thm1_undetectability.py`, `verify_thm2_decoys.py`, and `verify_eval_cis.py`. The public repo keeps a smaller teaching set (`demos/README.md`). The formal treatment is in the preprint ([DOI 10.5281/zenodo.21922668](https://doi.org/10.5281/zenodo.21922668)).
