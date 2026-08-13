<!--
  This is the README for the NEW PUBLIC repo. Copy it there and rename to
  README.md. It is deliberately venue-free. Do not add the target venue name.
-->
# Honey Steganography

Undetectable *and* brute-force-deniable messaging over language models.

A message can be embedded in fluent, machine-generated text so that it is (i)
**undetectable** to a passive observer — the text is statistically
indistinguishable from ordinary language-model output — and (ii)
**brute-force-deniable** — every key, right or wrong, decodes to a distinct,
plausible message, so an attacker who searches keys offline is left with a pile
of equally credible plaintexts and no way to know which is real.

This repository hosts the author preprint, plain-language explanations, and a
proof-of-concept implementation.

## Preprint

- **Paper (PDF):** `paper-arxiv.pdf` in this repo.
- **DOI:** [10.5281/zenodo.21922668](https://doi.org/10.5281/zenodo.21922668)
- Author preprint, licensed **CC BY 4.0** — reuse is welcome with attribution.

## What's here

| File | What it is |
|------|------------|
| `paper-arxiv.pdf` | The preprint: theory, construction, and proof-of-concept results. |
| `linkedin-article.md` | The story, for a general/technical audience: the history, the idea, and what building it taught me. |
| `blog-technical-cut.md` | An engineer-level walkthrough of how the scheme actually works (between the article and the paper). |
| `notation-glossary.md` | Plain-English key to the symbols and proof techniques in the paper. |
| `CITATION.cff` | How to cite this work. |
| `LICENSE-NOTE.md` | Licensing and reuse terms. |
| `demos/` | Proof-of-concept code: keyed grammar encoder, LM/Discop samplers, plaintext DTE, evaluation harness. |

## Citing

Please cite via `CITATION.cff` (GitHub shows a **"Cite this repository"** button),
or use DOI [10.5281/zenodo.21922668](https://doi.org/10.5281/zenodo.21922668).

## Reference implementation

See `demos/` (and `demos/README.md`) for the proof-of-concept: a keyed grammar
encoder, bigram and GPT-2 samplers, the Discop zero-leak sampler, a plaintext
distribution-transforming encoder, and an evaluation harness. These are teaching
demos, not a production implementation.

## Licence

The manuscript and write-ups are licensed **CC BY 4.0**. See `LICENSE-NOTE.md`
for details and for the code licence (added with the implementation).

## Author

Peter Edwards — ORCID [0009-0007-4514-2028](https://orcid.org/0009-0007-4514-2028)
— Independent Researcher, United Kingdom.
