#!/usr/bin/env python3
"""PCFG natural-language encoder for passwords (Chatterjee et al. 2015).

A password is described by a parse: a structure (sequence of component classes)
plus the choice made inside each class. The DTE represents that parse as a
vector of integers X:

  - encode(password): parse the password into its structure and per-class
    choices, emit X.
  - decode(X): read the structure, consume the class choices, rebuild the string.

Decoding a vector of uniform random integers yields a plausible decoy password.
(For simplicity this grammar is unambiguous -- one parse per password -- so
encode is direct; the paper samples uniformly over the whole parse forest.)

Run:  python3 pcfg_nle.py
"""

import random

WORDS = ["dragon", "monkey", "shadow", "master", "ninja", "summer",
         "qwerty", "football", "password", "hunter", "purple", "silver"]
SYMBOLS = ["!", "@", "#", "$", "%", "&"]

# Structures alternate classes so a maximal-run tokenizer recovers them exactly.
STRUCTURES = [["W", "D"], ["W", "D", "Y"], ["D", "W"], ["W"], ["W", "Y", "D"]]

MAX_DIGIT_LEN = 4


def _classify(ch):
    if ch.isalpha():
        return "W"
    if ch.isdigit():
        return "D"
    return "Y"


def _tokenize(password):
    """Split into maximal same-class runs -> list of (class, substring)."""
    tokens = []
    run = password[0]
    cls = _classify(password[0])
    for ch in password[1:]:
        c = _classify(ch)
        if c == cls:
            run += ch
        else:
            tokens.append((cls, run))
            run, cls = ch, c
    tokens.append((cls, run))
    return tokens


def encode(password):
    """Password -> integer vector X (the parse)."""
    tokens = _tokenize(password)
    structure = [cls for cls, _ in tokens]
    if structure not in STRUCTURES:
        raise ValueError(f"structure {structure} not in grammar")
    X = [STRUCTURES.index(structure)]
    for cls, sub in tokens:
        if cls == "W":
            X.append(WORDS.index(sub))
        elif cls == "D":
            X.append(len(sub) - 1)  # decode does 1 + (code % MAX_DIGIT_LEN)
            X.append(int(sub))
        else:
            X.append(SYMBOLS.index(sub))
    return X


def decode(X):
    """Integer vector X -> password."""
    it = iter(X)
    structure = STRUCTURES[next(it) % len(STRUCTURES)]
    parts = []
    for cls in structure:
        if cls == "W":
            parts.append(WORDS[next(it) % len(WORDS)])
        elif cls == "D":
            length = 1 + (next(it) % MAX_DIGIT_LEN)
            value = next(it) % (10 ** length)
            parts.append(str(value).zfill(length))
        else:
            parts.append(SYMBOLS[next(it) % len(SYMBOLS)])
    return "".join(parts)


def _rand_int():
    """Stand-in for one uniform seed integer (cf. the UNIF encoder)."""
    return random.getrandbits(64)


def decoy():
    """Generate a plausible decoy password from uniform random integers."""
    structure = STRUCTURES[_rand_int() % len(STRUCTURES)]
    X = [STRUCTURES.index(structure)]
    for cls in structure:
        if cls == "D":
            X.append(_rand_int())  # length
            X.append(_rand_int())  # value
        else:
            X.append(_rand_int())
    return decode(X)


def main():
    random.seed(7)
    samples = ["dragon42", "shadow7!", "99master", "ninja", "purple!5"]
    print("round-trip (encode -> X -> decode):")
    for pw in samples:
        X = encode(pw)
        back = decode(X)
        print(f"  {pw:>12}  X={X}  ->  {back}")
        assert back == pw, "round-trip failed"
    print("all round-trip OK")

    print("\nplausible decoys from uniform random integer vectors:")
    for _ in range(8):
        print(f"  {decoy()}")


if __name__ == "__main__":
    main()
