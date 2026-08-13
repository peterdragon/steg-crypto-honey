#!/usr/bin/env python3
"""n-gram natural-language encoder (Chatterjee et al. 2015, "NG" DTE).

A character-level (n-1)-order Markov model is turned into a DTE by building a
per-context prefix (Huffman) code over the alphabet:

  - decode / generate: read secret bits, walk the context's code tree to a leaf,
    emit that character, slide the context. Bit choices ARE the payload.
  - encode / extract: re-walk the text, emitting each character's codeword.

Every context is Laplace-smoothed over the whole alphabet, so each code is at
least one bit long and generation always makes progress. This is the same idea
Wayner used, at the character level; neural linguistic steganography (Ziegler
2019, Shen 2020) swaps the n-gram model for a neural LM plus arithmetic coding
for near-optimal capacity.

Run:  python3 ngram_nle.py
"""

import collections
import heapq
import os

from _common import bytes_to_bits, bits_to_bytes

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a message hidden in plain sight is safe from the careless eye. "
    "grammars and models let us mimic the shape of ordinary language. "
    "when every decode looks like english the reader learns nothing at all. "
    "she sells sea shells by the sea shore on a calm and quiet morning. "
).lower()

ORDER = 3            # context length = ORDER - 1
PAD = " "


def _build_model(corpus, order):
    alphabet = sorted(set(corpus))
    counts = collections.defaultdict(collections.Counter)
    context = tuple(PAD * (order - 1))
    for ch in corpus:
        counts[context][ch] += 1
        context = (context + (ch,))[-(order - 1):]
    return alphabet, counts


class NGramNLE:
    def __init__(self, corpus=CORPUS, order=ORDER):
        self.order = order
        self.alphabet, self.counts = _build_model(corpus, order)
        self._tree_cache = {}
        self._code_cache = {}

    def _tree(self, context):
        if context in self._tree_cache:
            return self._tree_cache[context]
        ctx_counts = self.counts.get(context, {})
        # Laplace smoothing over the full alphabet guarantees >= 2 leaves.
        heap = []
        for order_idx, ch in enumerate(self.alphabet):
            weight = ctx_counts.get(ch, 0) + 1
            heapq.heappush(heap, (weight, order_idx, ch))
        tie = len(self.alphabet)
        while len(heap) > 1:
            w1, _, n1 = heapq.heappop(heap)
            w2, _, n2 = heapq.heappop(heap)
            heapq.heappush(heap, (w1 + w2, tie, (n1, n2)))
            tie += 1
        tree = heap[0][2]
        self._tree_cache[context] = tree
        return tree

    def _codes(self, context):
        if context in self._code_cache:
            return self._code_cache[context]
        codes = {}

        def walk(node, prefix):
            if isinstance(node, str):
                codes[node] = prefix
                return
            left, right = node
            walk(left, prefix + [0])
            walk(right, prefix + [1])

        walk(self._tree(context), [])
        self._code_cache[context] = codes
        return codes

    def _slide(self, context, ch):
        return (context + (ch,))[-(self.order - 1):]

    def generate(self, bits):
        """Turn a bit list into cover text (decode direction)."""
        context = tuple(PAD * (self.order - 1))
        pos = 0
        out = []
        while pos < len(bits):
            node = self._tree(context)
            while not isinstance(node, str):
                bit = bits[pos] if pos < len(bits) else 0
                pos += 1
                node = node[0] if bit == 0 else node[1]
            out.append(node)
            context = self._slide(context, node)
        return "".join(out)

    def recover(self, text, nbits):
        """Turn cover text back into the bit list (encode direction)."""
        context = tuple(PAD * (self.order - 1))
        bits = []
        for ch in text:
            bits.extend(self._codes(context)[ch])
            context = self._slide(context, ch)
        return bits[:nbits]

    def hide(self, data):
        bits = bytes_to_bits(data)
        return self.generate(bits), len(bits)

    def reveal(self, text, nbits):
        return bits_to_bytes(self.recover(text, nbits))


def main():
    nle = NGramNLE()
    secret = b"attack at dawn"
    text, nbits = nle.hide(secret)
    print(f"secret bytes : {secret!r}  ({nbits} bits)")
    print("cover text   :")
    print(f"  {text!r}")
    recovered = nle.reveal(text, nbits)
    print(f"recovered    : {recovered!r}")
    assert recovered == secret, "round-trip failed"
    print("round-trip OK")

    print("\ndecoding uniformly random bits -> plausible decoy text:")
    rand_bits = bytes_to_bits(os.urandom(24))
    print(f"  {nle.generate(rand_bits)!r}")


if __name__ == "__main__":
    main()
