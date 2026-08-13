# Honey steganography — encode / decode flow

Editable source: this file is **Mermaid** inside Markdown.

- **Edit:** change the text in the ` ```mermaid ` blocks (VS Code preview, GitHub, Notion, Obsidian, etc.).
- **Tool:** [Mermaid Live Editor](https://mermaid.live) — paste a block, tweak, export SVG/PNG if needed.
- **Why Mermaid:** version-controllable, no proprietary binary; drag-and-drop alternatives (draw.io) are better for free-form posters, worse for git diffs.

Public values (travel with the cover, or are agreed out of band): **prompt**, **nonce** \(\nu\).  
Secret: **key** \(\sigma\) only.

---

## Encode (seal a message into cover text)

```mermaid
flowchart TB
  subgraph public_in["Public inputs"]
    P["prompt<br/>(cover prefix)"]
    N["nonce ν<br/>(per message)"]
  end

  subgraph secret_in["Secret"]
    K["key σ"]
  end

  M["plaintext M<br/>(10–24 DistilGPT-2 tokens<br/>in pmt nucleus)"]

  M --> TOK["Tokenize with DistilGPT-2<br/>(plaintext model pmt)"]
  TOK --> IDS["token ids"]

  IDS --> DTE["Neural DTE.encode<br/>randomised arithmetic coding<br/>min_tokens=10, n_max=24"]
  DTE --> SEED["seed<br/>ℓ-bit uniform-in-leaf"]

  K --> KDF["KDF σ → mk<br/>(Argon2id)"]
  KDF --> KSA["ks_A = F(mk, ν ‖ mask)"]
  N --> KSA

  SEED --> XOR["c = seed ⊕ ks_A<br/>(honey ciphertext bits)"]
  KSA --> XOR

  N --> PUB["Pub = H(ν ‖ public)<br/>public segmentation tape"]
  PUB --> R["r_i, filler bits"]

  P --> DISC["GPT-2 Discop sampler<br/>distribution-copy rotation"]
  R --> DISC
  XOR --> DISC

  DISC --> T["cover text T<br/>(prompt + continuation)"]

  T --> OUT["Transmit / store<br/>T + ν + prompt"]
```

---

## Decode (true key — recover M)

```mermaid
flowchart TB
  subgraph public_in["Public inputs"]
    T["cover text T"]
    P["prompt"]
    N["nonce ν"]
  end

  subgraph secret_in["Secret"]
    K["key σ"]
  end

  T --> RETOK["Re-tokenize T<br/>(must start with prompt)"]
  P --> RETOK
  N --> PUB["Pub = H(ν ‖ public)"]
  PUB --> READ["ReadBits(T, Pub)<br/>same raw word w' for every key"]
  RETOK --> READ

  K --> KDF["KDF σ → mk"]
  KDF --> KSA["ks_A"]
  N --> KSA

  READ --> XOR["seed' = w' ⊕ ks_A"]
  KSA --> XOR

  XOR --> DTE["Neural DTE.decode<br/>(DistilGPT-2 pmt)"]
  DTE --> M["plaintext M"]
```

---

## Wrong key (honey path — ≥10-token decoy)

Same cover \(T\), nonce, and prompt. Only the key differs.

```mermaid
flowchart TB
  T["cover text T"] --> READ["ReadBits(T, Pub)<br/>w' identical to true-key path"]
  N["nonce ν"] --> PUB["Pub"]
  PUB --> READ

  Kwrong["wrong key σ′"] --> KDF["KDF"]
  KDF --> KSAwrong["ks_A′ ≠ ks_A"]
  N --> KSAwrong

  READ --> XOR["seed′ = w' ⊕ ks_A′<br/>≈ uniform, independent of M"]
  KSAwrong --> XOR

  XOR --> DTE["Neural DTE.decode"]
  DTE --> DECOY["decoy message<br/>fresh pmt sample<br/>length ≥ min_tokens=10"]
```

Because segmentation is public, a wrong key **cannot** use bit-count as a correct-key test. After public extraction, honey is classical **DTE-then-OTP**.

---

## Layers at a glance

Exported PNG (for LinkedIn / posts): [`honey-layers.png`](honey-layers.png)  
Source: [`honey-layers.mmd`](honey-layers.mmd)

```mermaid
flowchart LR
  subgraph plaintext_layer["Plaintext layer — DistilGPT-2 DTE"]
    A["M ↔ seed"]
  end

  subgraph honey_layer["Honey layer — secret ks_A"]
    B["seed ⊕ ks_A ↔ c"]
  end

  subgraph cover_layer["Cover layer — GPT-2 Discop + public Pub"]
    C["c ↔ fluent cover T"]
  end

  A --> B --> C
```

| Layer | Model | Secret? | Job |
|-------|--------|---------|-----|
| Plaintext DTE | DistilGPT-2 | no (public algorithm) | Map message ↔ uniform seed; wrong seeds → fluent decoys |
| Honey mask | Argon2id + PRF | **key** (+ nonce) | OTP so wrong keys see uniform seeds |
| Cover | GPT-2 Discop | no for `Pub`; mask only via \(c\) | Hide bits in innocent continuation of **prompt** |

---

## Editing tips

1. Open [mermaid.live](https://mermaid.live) and paste one `flowchart` block.
2. Or preview this file in VS Code (`Markdown: Open Preview`).
3. Keep node labels short; put detail in the tables above.
4. If you later want drag-and-drop: File → Export as SVG from mermaid.live, then import into [diagrams.net](https://app.diagrams.net).
