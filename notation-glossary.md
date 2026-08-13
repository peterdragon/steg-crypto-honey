# Notation & concept glossary

A study companion for `paper-ihmmsec.tex` / `security-model.tex` (honey
steganography). Every symbol below is taken directly from the papers' macros and
proofs. Each entry gives a plain-English meaning and a pointer to background
reading that explains the underlying machinery.

## Reading key

| Tag | Source | Note |
|-----|--------|------|
| **[JoC]** | Rosulek, *The Joy of Cryptography* | free; best on-ramp to hybrid/indistinguishability proofs |
| **[KL]** | Katz & Lindell, *Introduction to Modern Cryptography*, 3rd ed. | security definitions, PRF/PRG, reductions, ROM |
| **[BS]** | Boneh & Shoup, *A Graduate Course in Applied Cryptography* | free; rigorous game-based proofs |
| **[MU]** | Mitzenmacher & Upfal, *Probability and Computing*, 2nd ed. | Chernoff bounds, balls-and-bins/occupancy |
| **[CT]** | Cover & Thomas, *Elements of Information Theory*, 2nd ed. | entropy, KL divergence, hypothesis testing |
| **[LP]** | Levin & Peres, *Markov Chains and Mixing Times* (ch. 5) | free; coupling |
| **[Cachin]** | Cachin (2004), *An Information-Theoretic Model for Steganography* | the ε-security definition and warden game |
| **[HLvA]** | Hopper, Langford & von Ahn (2002), *Provably Secure Steganography* | the computational framing |
| **[JR14]** | Juels & Ristenpart (2014), *Honey Encryption* | DTE, message-recovery security, balls-and-bins bound |

---

## 1. Distributions and probability spaces

| Symbol | Reads as | Meaning | See |
|--------|----------|---------|-----|
| `P_C` (`\Cover`) | "P-cover" | Covertext distribution: the law of text an **innocent** user's sampler would emit. | [Cachin] |
| `P_S` (`\Stego`) | "P-stego" | Stegotext distribution: the law of text the encoder emits **while embedding** a message. Security wants `P_S = P_C`. | [Cachin] |
| `p` (`\Ptrue`) | "p-true" | The language model's **true** next-token distribution (untruncated). | — |
| `p̂` (`\Pnuc`) | "p-hat" | The **nucleus-truncated** distribution (top-p mass renormalised). | [HLvA] |
| `P̃` (`\Pquant`) | "P-tilde" | The **quantised** (fixed-point) truncated distribution — the *actual* per-step sampling law used by Discop. | — |
| `p_m` (`\pmt`) | "p-m" | The **message** distribution the DTE targets (a.k.a. `q_plaintext`). | [JR14] |
| `p_k` (`\pk`) | "p-k" | The **key/password** distribution (typically low-entropy). | [JR14] |
| `Q_h` | — | The common per-step token law at history `h`; the lemma proves `Q_h = P̃(·|h)` regardless of the embedded bit. | [LP] |
| `Unif`, `U` | "uniform" | Uniform distribution / a uniform random variable (e.g. a uniform seed). | [MU] |
| `M`, `K`, `S` (`\Mspace` etc.) | — | Message space, key space, seed space. | — |
| `ℤ₂` (`\bits`) | — | The set `{0,1}` (one bit). | — |

## 2. Information-theoretic security

| Symbol | Reads as | Meaning | See |
|--------|----------|---------|-----|
| `D(P ‖ Q)` (`\KL`) | "KL / relative entropy of P w.r.t. Q" | Kullback–Leibler divergence; the paper's security measure. `D=0` ⇔ `P=Q`. | [CT] |
| `ε`-security | "epsilon-secure" | A stegosystem is ε-secure if `D(P_C ‖ P_S) ≤ ε`; **perfectly secure** when `ε = 0`. | [Cachin] |
| `TV(P,Q)` | "total variation" | Statistical distance `½Σ|P−Q|`; equals the best distinguisher's advantage. | [CT], [LP] |
| Pinsker | — | Pinsker's inequality bounds `TV ≤ √((ln2/2)·D)`: small KL ⇒ small advantage. | [CT] |
| `α`, `β` | — | Warden's false-alarm and missed-detection rates (hypothesis-testing view). | [Cachin], [CT] |

## 3. The scheme: algorithms, keys, variables

| Symbol | Reads as | Meaning | See |
|--------|----------|---------|-----|
| `Embed` (`\Emb`) | — | Encoder: `(K, ν, M) → text`. | — |
| `Extract` (`\Ext`) | — | Decoder: `(K, ν, text) → message`. | — |
| `ReadBits` | — | Receiver's per-token routine recovering embedded bits. | — |
| `dte.enc` / `dte.dec` | "DTE encode/decode" | Distribution-transforming encoder: `message → uniform seed` and back. | [JR14] |
| `seed` | — | The (uniform) seed string the DTE maps to/from; lives in `S`. | [JR14] |
| `KDF`, `mk` | "key-derivation function", "master key" | Memory-hard `σ ↦ mk = KDF(σ)` (Argon2id). | [KL] |
| `F` | — | A **PRF** (pseudorandom function); the secret mask is `ks_A = F(mk, ν ‖ "mask")`. | [KL], [JoC] |
| `H` | — | A cryptographic **hash** (modelled as a PRG of the nonce); builds the public tape. | [KL], [JoC] |
| `ks_A` (`\ksA`) | "keystream A / mask" | Mask keystream that whitens the seed: `c = seed ⊕ ks_A`. | [JoC] |
| `Pub` (`\Pub`) | "public tape" | Nonce-only segmentation/filler tape `Pub = H(ν ‖ "public")`; supplies `rᵢ`. Not keyed. | [JoC] |
| `c` | — | Whitened payload `= seed ⊕ ks_A` (the bits actually embedded). | [JoC] |
| `w`, `w'` | — | Embedded / recovered bit string (`w = c`; `w' = ReadBits(T, Pub)`, identical for every key). | [JoC] |
| `K`, `K'` | — | The true key vs. a guessed/wrong key. | — |
| `σ` (`\sigma`) | "sigma" | The long-term secret. | — |
| `ν` (`\nu`) | "nu" | Per-message **public nonce** (sent out of band). | [KL] |
| `ℓ` (`\ell`) | "ell" | Seed / ciphertext length in bits (`= |c|`); fixed, so length leaks nothing. | — |

## 4. Discop sampler mechanics (Theorem 1)

| Symbol | Reads as | Meaning | See |
|--------|----------|---------|-----|
| `T` (`\TOTAL`) | — | Fixed-point denominator `2^PRECISION` (e.g. `2³²`); the CDF is laid out over `[0,T)`. | — |
| `H` (`\HALF`) | — | `T/2`; the rotation offset. | — |
| `s` | — | The **half-rotation** `s(r) = (r+H) mod T`; a measure-preserving **involution** (`s∘s = id`). The engine of Theorem 1. | [LP] |
| `r`, `rᵢ` | — | Uniform selector value(s) drawn from `Pub`. | [MU] |
| `tok_h(r)` (`\tokenat`) | "token-at" | Inverse-CDF map: the token whose fixed-point interval contains `r` at history `h`. | — |
| `i₀`, `i₁` | — | The two "distribution copies": `tok_h(r)` and `tok_h(s(r))`. | — |
| `B` | — | **Embedding region**: the set of `r` where `i₀ ≠ i₁` and both are safe — i.e. a bit is carried. | — |
| `β` | "beta" | The selector bit consumed at an embedding step. | — |
| `e`, `eᵢ` | — | Embedding **indicator**: `1` iff the step embeds a bit (`r ∈ B`). | — |
| `N` | — | Total emitted-token count; a **stopping time** w.r.t. the `e`-sequence. | [LP] |
| `safe_h(t)` (`\safe`) | "safe" | Self-tokenizing predicate: `enc(dec(h++t)) = h++t` — barring a token that would re-tokenize differently. | — |
| `ρ` (`\rho`) | "rho" | Nucleus **top-p** threshold; the truncation keeps mass `≥ ρ`. | — |
| `Z_h` | — | Nucleus probability mass captured at history `h` (`≥ ρ`); the truncation-KL budget. | — |
| `𝔼_h[·]` | "expectation over histories" | Average over the histories the walk visits (e.g. mean unsafe mass). | [MU] |

## 5. Honey / deniability (Theorem 2)

| Symbol | Reads as | Meaning | See |
|--------|----------|---------|-----|
| `Adv^mr(A)` (`\Adv`) | "advantage, message-recovery" | Interrogator `A`'s probability of recovering the true message minus the guessing baseline. | [JR14], [KL] |
| `q` | — | Number of **key-derivation queries** `A` makes (offline guesses). | [JR14] |
| `ε` (`\varepsilon`) | "epsilon" | DTE **goodness**: statistical distance between `dte.dec(U)` and `p_m`; the paper achieves `ε = O(2⁻ᵇ)`. | [JR14] |
| `b` | — | Fixed-point **precision bits** of the arithmetic-coding DTE. | — |
| `μ` (`\mu`) | "mu" | Unavoidable **guessing baseline** `= max(maxₖ p_k, maxₘ p_m)`. | [JR14] |
| `δ` (`\delta`) | "delta" | **Balls-and-bins remainder** in the JR14 bound; `δ → 0` as the key distribution flattens. | [JR14], [MU] |
| `κ` (`\kappa`) | "kappa" | Key min-entropy exponent (uniform key set of size `≥ 2^κ`); `δ_κ ≤ q·2⁻κ`. | [MU] |
| `⊕` | "XOR" | Bitwise exclusive-or; XORing with an independent uniform string yields a uniform result (**one-time pad**). | [JoC] |

## 6. Relational operators & recurring conventions

| Symbol | Reads as | Meaning |
|--------|----------|---------|
| `X ⊥ Y` (`\perp`) | "X independent of Y" | Statistical independence. |
| `X ∼ P` / `X ← P` | "distributed as" / "sampled from" | `X` is drawn from distribution `P`. |
| `a ‖ b` | "a concatenated with b" | String/bit concatenation (inside PRF inputs). |
| `++` | "append" | Sequence concatenation (`h ++ t` = history followed by token). |
| `x ↦ y` | "maps to" | Function mapping (e.g. `σ ↦ mk`). |
| `⌈x⌉` | "ceiling" | Least integer `≥ x`. |
| `p̄` (`\bar p`) | "p-bar" | Mean embedding probability `𝔼_h[|B|/T]`. |
| `L` | — | Public token **budget** in the fixed-budget length-hiding variant. |

---

## 7. Proof-technique glossary (the "how", not the symbols)

These are the reusable techniques the proofs lean on. If a proof step feels
opaque, it is almost always one of these.

- **Reduction** — "if you could break X, you could break Y (assumed hard), so X
  is hard too." The spine of Theorem 2 (breaking our MR ⇒ breaking JR14 HE).
  [JoC], [KL], [BS]
- **Hybrid argument** — bound the gap between two distributions by changing them
  **one piece at a time**, paying a small cost per step. The `qε` term is a
  hybrid over the `q` decoy bins. [JoC], [BS]
- **Advantage / distinguisher** — an adversary's edge over guessing; security =
  "advantage is negligible / bounded." [KL], [BS]
- **PRG / PRF** — pseudorandom generator/function: output computationally
  indistinguishable from random. In the standard model, `ks_A` is a PRF under
  `mk`, and `Pub` is a hash/PRG of the nonce alone (not keyed by the secret).
  [KL], [JoC]
- **Random oracle model (ROM)** — idealise a hash/KDF as a truly random
  function; used for the *cross-key* independence in Assumption (ROM). [KL], [BS]
- **One-time pad** — `uniform ⊕ anything independent = uniform`; why the recovered
  seed `seed' = w' ⊕ ks_A'` is uniform under a wrong key. [JoC]
- **Involution & measure-preserving map** — a bijection that is its own inverse
  and doesn't change probabilities; `s` swaps the two copies without perturbing
  the token law, giving Theorem 1. [LP]
- **Coupling** — analyse two random processes on a shared probability space to
  bound their difference (e.g. the self-tokenizing gap `≤ 2·𝔼_h[P̃(unsafe)]`). [LP]
- **Pushforward** — the law induced by pushing a distribution through a map (`r`
  uniform ⇒ `tok_h(r) ∼ P̃`). [LP], [CT]
- **Stopping time** — a random time defined only by "what's happened so far";
  the token count `N` is one, so the stopped sequence keeps the same law. [LP]
- **Balls-and-bins / occupancy** — throwing `q` balls (guesses) into key bins;
  the max-load analysis yields the `δ` remainder. [MU], [JR14]
- **Chernoff bound** — exponential tail bound; gives the `e^{−Ω(L)}` underflow
  probability in the fixed-budget variant. [MU]

---

## 8. A worked reading of the two main results

**Theorem 1 (undetectability), one sentence:** because the half-rotation `s` is a
measure-preserving involution on the embedding region `B`, the token emitted when
you embed a bit has *exactly* the same distribution `P̃(·|h)` as an honest draw,
so `P_S = P_C` and `D(P_C ‖ P_S) = 0`.
Prerequisites: pushforward + involution/coupling [LP], KL divergence [CT].

**Theorem 2 (deniability), one sentence:** a wrong key XORs the payload with an
independent keystream, so the recovered seed is uniform (one-time pad) and decodes
to a fresh `p_m` sample (a plausible decoy); hence recovering the true message is
no easier than in JR14 honey encryption, up to `qε` for the `q` decoy bins the
interrogator inspects.
Prerequisites: one-time pad + PRF/ROM [JoC], [KL]; reduction + hybrid [JoC], [BS];
balls-and-bins for the `μ + δ` evaluation [MU], [JR14].
