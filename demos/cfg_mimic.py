#!/usr/bin/env python3
"""Wayner (1992) context-free-grammar mimic function.

Bits of the secret are hidden in the *choice of production* at each grammar
variable, not in the terminal words themselves. Encoding drives a left-most
derivation from a bit stream; decoding parses the text back into the unique
production sequence and recovers the bits.

The grammar is unambiguous and LL(1): every alternative of a variable starts
with a distinct terminal word, so a predictive parser can recover the choice.
Production counts are powers of two, so a variable with N alternatives carries
log2(N) bits. Slots here mix 4-, 8-, and 16-way choices, so one sentence carries
BITS_PER_SENTENCE bits (see sentence_capacity()).

Run:  python3 cfg_mimic.py
"""

from _common import bytes_to_bits, bits_to_bytes, int_to_bits

START = "S"


def _words(*options):
    """Each alternative is a single terminal word (its own distinct first token)."""
    return [[w] for w in options]


# One sentence:
#   GREET NAME ADVERB VERB ART ADJ NOUN PREP ADJ2 PLACE TIME CONJ CLOSER
#   e.g. "Hello, Alice quietly observed the ancient library near misty harbour
#         at dawn while gulls circled."
# Widths: 3  4    3     4   2   4    4    3    4     4    3    3     4  = 45 bits / sentence.
GRAMMAR = {
    "S": [["GREET", "NAME", "ADVERB", "VERB", "ART", "ADJ", "NOUN",
           "PREP", "ADJ2", "PLACE", "TIME", "CONJ", "CLOSER"]],
    "GREET": _words("Hello,", "Greetings,", "Howdy,", "Salutations,",
                    "Hey,", "Welcome,", "Yo,", "Ahoy,"),
    "NAME": _words("Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace",
                   "Heidi", "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj",
                   "Olivia", "Peggy"),
    "ADVERB": _words("quietly", "boldly", "slyly", "quickly",
                     "calmly", "eagerly", "silently", "cheerfully"),
    "VERB": _words("observed", "approached", "found", "described", "sketched",
                   "guarded", "entered", "praised", "studied", "painted",
                   "photographed", "mapped", "inspected", "admired", "restored",
                   "discovered"),
    "ART": _words("the", "a", "one", "that"),
    "ADJ": _words("ancient", "gloomy", "radiant", "crooked", "silent", "gilded",
                  "weathered", "hidden", "crimson", "frozen", "restless", "humble",
                  "jagged", "velvet", "ivory", "hollow"),
    "NOUN": _words("library", "lantern", "fountain", "statue", "doorway", "mural",
                   "bridge", "garden", "tower", "archway", "cellar", "balcony",
                   "corridor", "courtyard", "staircase", "alcove"),
    "PREP": _words("near", "beside", "inside", "behind",
                   "beneath", "above", "beyond", "opposite"),
    "ADJ2": _words("misty", "distant", "sunlit", "shadowed", "quiet", "bustling",
                   "sacred", "ruined", "verdant", "sprawling", "narrow", "grand",
                   "coastal", "moonlit", "forgotten", "windswept"),
    "PLACE": _words("harbour", "market", "plaza", "orchard", "chapel", "arcade",
                    "terrace", "quay", "boulevard", "meadow", "wharf", "citadel",
                    "promenade", "cloister", "vineyard", "esplanade"),
    "TIME": [["at", "dawn"], ["by", "noon"], ["near", "dusk"],
             ["past", "midnight"], ["before", "sunrise"], ["after", "twilight"],
             ["toward", "evening"], ["until", "nightfall"]],
    "CONJ": _words("while", "as", "before", "after",
                   "since", "though", "until", "whenever"),
    "CLOSER": [["everyone", "watched."], ["nobody", "stirred."],
               ["gulls", "circled."], ["bells", "rang."],
               ["shadows", "lengthened."], ["lanterns", "flickered."],
               ["travellers", "paused."], ["merchants", "haggled."],
               ["children", "laughed."], ["sailors", "sang."],
               ["dancers", "whirled."], ["monks", "chanted."],
               ["pilgrims", "knelt."], ["ravens", "gathered."],
               ["strangers", "lingered."], ["guards", "saluted."]],
}


def _code_width(nprod):
    """Bits consumed to select among nprod equiprobable productions."""
    if nprod & (nprod - 1) != 0:
        raise ValueError("this demo assumes a power-of-two production count")
    return nprod.bit_length() - 1


def sentence_capacity():
    """Bits hidden in one full sentence (sum of log2(#productions) per slot)."""
    return sum(_code_width(len(GRAMMAR[sym]))
               for sym in GRAMMAR[START][0]
               if sym in GRAMMAR and len(GRAMMAR[sym]) > 1)


def encode(data):
    """Hide bytes in generated text. Returns (text, n_secret_bits)."""
    bits = bytes_to_bits(data)
    pos = 0
    tokens = []

    def expand(sym):
        nonlocal pos
        if sym not in GRAMMAR:
            tokens.append(sym)
            return
        prods = GRAMMAR[sym]
        if len(prods) == 1:
            idx = 0
        else:
            width = _code_width(len(prods))
            value = 0
            for _ in range(width):
                bit = bits[pos] if pos < len(bits) else 0
                pos += 1
                value = (value << 1) | bit
            idx = value
        for s in prods[idx]:
            expand(s)

    # Emit whole sentences until every real bit has been consumed.
    while pos < len(bits):
        expand(START)
    if not tokens:
        expand(START)
    return " ".join(tokens), len(bits)


def decode(text, n_secret_bits):
    """Parse cover text back to the original bytes."""
    tokens = text.split()
    pos = 0
    bits = []

    def parse(sym):
        nonlocal pos
        if sym not in GRAMMAR:
            assert tokens[pos] == sym, f"expected {sym!r}, got {tokens[pos]!r}"
            pos += 1
            return
        prods = GRAMMAR[sym]
        if len(prods) == 1:
            idx = 0
        else:
            idx = None
            for i, prod in enumerate(prods):
                if prod[0] == tokens[pos]:
                    idx = i
                    break
            assert idx is not None, f"no production of {sym} starts with {tokens[pos]!r}"
            bits.extend(int_to_bits(idx, _code_width(len(prods))))
        for s in prods[idx]:
            parse(s)

    while pos < len(tokens):
        parse(START)
    return bits_to_bytes(bits[:n_secret_bits])


def main():
    cap = sentence_capacity()
    print(f"capacity     : {cap} bits/sentence (~{cap / 8:.1f} bytes)")
    secret = b"Rendezvous!"
    print(f"secret bytes : {secret!r}")
    text, nbits = encode(secret)
    n_sentences = text.count(".")
    print(f"secret bits  : {nbits}  -> {n_sentences} sentence(s)")
    print("cover text   :")
    print(f"  {text}")
    recovered = decode(text, nbits)
    print(f"recovered    : {recovered!r}")
    assert recovered == secret, "round-trip failed"
    print("round-trip OK")


if __name__ == "__main__":
    main()
