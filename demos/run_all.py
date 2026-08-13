#!/usr/bin/env python3
"""Run the public teaching demos in story order.

These are stdlib (or Argon2-optional) programs that mirror the LinkedIn /
blog arc: grammar mimic → keyed honey → Huffman leak → Discop → DTE →
wrong-key decoys. They are *not* the paper's GPT-2 evaluation harnesses.
"""

import runpy

DEMOS = [
    "cfg_mimic",
    "keyed_cfg_mimic",
    "weighted_cfg_mimic",
    "lm_mimic",
    "discop_mimic",
    "inverse_sampling_dte",
    "honey_encryption",
    "ngram_nle",
    "pcfg_nle",
    "e2_decoy_plausibility",
]

for name in DEMOS:
    print("=" * 60)
    print(name)
    print("=" * 60)
    runpy.run_module(name, run_name="__main__")
    print()
