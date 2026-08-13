#!/usr/bin/env python3
"""Autoregressive LM keyed mimic — teaching step that exposes the Huffman leak.

Replaces the hand-crafted CFG in keyed_cfg_mimic.py with a word-level bigram
language model.  Huffman coding from P(next|prev) is fluent but *not*
distribution-preserving unless P is dyadic — the ~0.013 bits/word leak the
LinkedIn article mentions.  The next demo (`discop_mimic.py`) removes that
sampling gap relative to the deployed quantized LM.

Architecture (same structure as Meteor, Kaptchuk et al. CCS 2021):

    payload      = length_header || data || zero-padding  (PAD_BLOCK aligned)
    keystream    = SHA-256-CTR(Argon2id(secret), nonce)
    cipher_bits  = bytes_to_bits(payload) XOR keystream
    text         = LM_huffman_encode(cipher_bits)

Decode is the exact reverse.  Wrong secret → wrong keystream → wrong Huffman
paths → valid corpus words but garbled plaintext (honey / deniability property
inherited from keyed_cfg_mimic).

Bit alignment note
------------------
Huffman code lengths vary, so the decoded bit stream may be a few bits longer
than the original cipher_bits (the last word's code zero-pads the trailing
bits).  Both sides operate bit-by-bit; the keystream is consumed to exactly
len(cipher_bits) in encode and len(recovered_bits) in decode.  The extra bits
XOR to noise, which unpack_payload discards via the length header.

--- Extending to a real neural LM (e.g. distilgpt2 / GPT-2) ---------------
Replace BigramLM.distribution(prev_token) with a transformer call:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    logits = model(input_ids)[0][0, -1, :]
    probs  = torch.softmax(logits, -1)
    # nucleus (top-p) sampling to bound the effective vocab:
    sorted_p, sorted_i = probs.sort(descending=True)
    mask = (sorted_p.cumsum(0) - sorted_p) < top_p
    dist = {int(sorted_i[j]): sorted_p[j].item()
            for j in range(len(sorted_i)) if mask[j]}

The encode / decode logic below is unchanged; only distribution() changes.
With GPT-2 the cover text is fluent English indistinguishable from model output,
approaching zero KL divergence (the Discop / MEC security regime).
----------------------------------------------------------------------------

Run:  python3 lm_mimic.py
      ./.venv/bin/python lm_mimic.py   (Argon2id KDF)
"""

import heapq
import os
import re
from collections import Counter, defaultdict

from _common import bytes_to_bits, bits_to_bytes
from keyed_cfg_mimic import (
    NONCE_LEN, _wipe, derive_key, pack_payload, unpack_payload,
    Keystream, _HAVE_ARGON2,
)

# ---------------------------------------------------------------------------
# Embedded corpus: Pride and Prejudice, Jane Austen (public domain).
# Larger corpus → richer bigram coverage → more natural-looking cover text.
# ---------------------------------------------------------------------------
_CORPUS = """\
it is a truth universally acknowledged that a single man in possession of a
good fortune must be in want of a wife however little known the feelings or
views of such a man may be on his first entering a neighbourhood this truth is
so well fixed in the minds of the surrounding families that he is considered as
the rightful property of some one or other of their daughters my dear mr bennet
said his lady to him one day have you heard that netherfield park is let at last
mr bennet replied that he had not but it is returned she for mrs long has just
been here and she told me all about it mr bennet made no answer do you not want
to know who has taken it cried his wife impatiently you want to tell me and i
have no objection to hearing it this was invitation enough why my dear you must
know mrs long says that netherfield is taken by a young man of large fortune
from the north of england that is all very well and it is nothing to us my dear
mr bennet replied his wife how can you be so tiresome you must know that i am
thinking of his marrying one of them is that his design in settling here
nonsense how can you talk so but it is very likely that he may fall in love
with one of them and therefore you must visit him as soon as he comes i see no
occasion for that you and the girls may go or you may send them by themselves
which perhaps will be still better for as you are as handsome as any of them
mr bingley might like you the best of the party oh my dear cried his wife i am
quite delighted with him he is so excessively handsome and his sisters are
charming women i never in my life saw anything more elegant than their dresses
i dare say replied jane i should not care about it myself but these things are
a great deal to many people to be sure they are said her mother well my dear
child i can think of nothing else the whole family spent the morning in the park
walking about in search of pleasure the weather was fine and the company most
agreeable they spoke of many things of books and music and the affairs of the
neighbourhood it was a pleasant scene and all were satisfied nothing remarkable
occurred yet all felt the day to be a memorable one the evening brought fresh
conversation and the family sat together in good spirits until it was time to
retire for the night the house was still and the night was clear and the stars
were bright and all the world was at peace it had been a long day and a pleasant
one and all were content with what they had seen and heard and done and spoken
and thought and felt in the hours of that good and quiet day now gone forever
into the past but remembered with kindness and with gratitude by all who had
been present and had shared in the simple pleasures of the hour and so to rest
"""


def _tokenize(text: str) -> list:
    """Lowercase alphabetic words only (no punctuation tokens)."""
    return re.findall(r"[a-z']+", text.lower())


# ---------------------------------------------------------------------------
# Bigram language model
# ---------------------------------------------------------------------------

class BigramLM:
    """Word-level bigram model with Laplace (add-1) smoothing.

    Both encode and decode call huffman(prev) for the same sequence of
    context words, so they always build identical trees without exchanging
    extra information.  Alphabetical tie-breaking in the heap makes the
    Huffman tree fully deterministic across Python versions.
    """

    START = "<s>"

    def __init__(self, corpus: str) -> None:
        tokens = _tokenize(corpus)
        self.vocab = sorted(set(tokens))          # sorted → deterministic
        self._V = len(self.vocab)
        self._counts: dict = defaultdict(Counter)
        self._counts[self.START][tokens[0]] += 1
        for prev, word in zip(tokens, tokens[1:]):
            self._counts[prev][word] += 1
        self._cache: dict = {}

    def distribution(self, prev: str) -> dict:
        """Laplace-smoothed P(word | prev) for every word in the vocab."""
        c = self._counts.get(prev, Counter())
        denom = sum(c.values()) + self._V
        return {w: (c.get(w, 0) + 1) / denom for w in self.vocab}

    def huffman(self, prev: str) -> tuple:
        """Return (tree, codes) for context `prev`, cached after first build.

        tree  — nested tuples of str leaves.
        codes — {word: [0/1, ...]} for encoding; look up to decode.
        """
        if prev in self._cache:
            return self._cache[prev]
        dist = self.distribution(prev)
        heap: list = []
        for i, w in enumerate(sorted(dist)):      # alphabetical → deterministic
            heapq.heappush(heap, (dist[w], i, w))
        ctr = len(dist)
        while len(heap) > 1:
            w1, _, n1 = heapq.heappop(heap)
            w2, _, n2 = heapq.heappop(heap)
            heapq.heappush(heap, (w1 + w2, ctr, (n1, n2)))
            ctr += 1
        tree = heap[0][2]
        codes: dict = {}

        def _walk(node, bits: list) -> None:
            if isinstance(node, str):
                codes[node] = bits
            else:
                _walk(node[0], bits + [0])
                _walk(node[1], bits + [1])

        _walk(tree, [])
        result = (tree, codes)
        self._cache[prev] = result
        return result

    def avg_code_length(self) -> float:
        """Expected bits per word, measured over the corpus itself."""
        tokens = _tokenize(_CORPUS)
        prev = self.START
        total = 0
        for tok in tokens:
            _, codes = self.huffman(prev)
            total += len(codes.get(tok, []))
            prev = tok
        return total / len(tokens)


# Single shared LM instance (Huffman trees are cached after first build).
LM = BigramLM(_CORPUS)


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def encode(secret, data: bytes, nonce: bytes = None, key=None) -> tuple:
    """Encode bytes as LM-sampled cover text.  Returns (text, nonce).

    Pass a pre-derived key bytearray to skip the slow KDF across many calls
    (the caller then owns and is responsible for wiping it).
    """
    if nonce is None:
        nonce = os.urandom(NONCE_LEN)
    owned = key is None
    k = derive_key(secret) if owned else key
    try:
        payload = pack_payload(data)
        pbits = bytes_to_bits(payload)
        ks = Keystream(k, nonce)
        kbits = ks.bits(len(pbits))
        cbits = [p ^ q for p, q in zip(pbits, kbits)]   # stream-cipher

        words: list = []
        pos = 0
        prev = LM.START

        while pos < len(cbits):
            tree, _ = LM.huffman(prev)
            node = tree
            while not isinstance(node, str):
                bit = cbits[pos] if pos < len(cbits) else 0
                pos += 1
                node = node[0] if bit == 0 else node[1]
            words.append(node)
            prev = node

        return " ".join(words), nonce
    finally:
        if owned:
            _wipe(k)


def decode(secret, text: str, nonce: bytes, key=None) -> bytes:
    """Recover bytes from LM cover text using (secret, nonce).

    Wrong secret → wrong keystream → wrong Huffman traversal → correct vocab
    words but garbled payload (honey property: no decryption error to observe).
    """
    owned = key is None
    k = derive_key(secret) if owned else key
    try:
        ks = Keystream(k, nonce)
        words = text.split()
        cbits: list = []
        prev = LM.START

        for word in words:
            _, codes = LM.huffman(prev)
            if word not in codes:
                raise ValueError(f"word {word!r} not in vocab for context {prev!r}")
            cbits.extend(codes[word])
            prev = word

        # cbits may be slightly longer than the original pbits (Huffman padding).
        # XOR with keystream of the same length; extra bits become noise that
        # unpack_payload discards via the length header.
        kbits = ks.bits(len(cbits))
        pbits = [c ^ q for c, q in zip(cbits, kbits)]
        return unpack_payload(bits_to_bytes(pbits))
    finally:
        if owned:
            _wipe(k)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"KDF        : {'Argon2id (argon2-cffi)' if _HAVE_ARGON2 else 'scrypt (stdlib fallback)'}")
    print(f"LM         : bigram over {len(LM.vocab)}-word corpus (Pride & Prejudice)")
    avg = LM.avg_code_length()
    print(f"Capacity   : ~{avg:.1f} bits/word  "
          f"(bigram Huffman; GPT-2 nucleus-p=0.95 gives ~10-13 bits/token)")

    secret = "correct horse battery staple"
    message = b"Rendezvous!"
    print(f"\nsecret  : {secret!r}")
    print(f"message : {message!r}  ({len(message) * 8} bits payload)")

    text, nonce = encode(secret, message)
    words = text.split()
    print(f"\nnonce   : {nonce.hex()}  (distribute with the secret)")
    print(f"cover   : {len(words)} words")
    print(f"  {text}")

    recovered = decode(secret, text, nonce)
    assert recovered == message, f"round-trip failed: {recovered!r}"
    print(f"decoded : {recovered!r}  (correct)")

    text2, nonce2 = encode(secret, message)
    print(f"\nnonce2  : {nonce2.hex()}")
    print(f"cover2  : {text2}")
    print(f"same secret+message, fresh nonce -> different text: {text != text2}")

    wrong = decode("hunter2", text, nonce)
    print(f"\nwrong key -> {wrong!r}")
    print("  (wrong keystream -> wrong Huffman paths -> valid corpus words, garbled payload)")

    print()
    print("Cover text uses corpus vocabulary and bigram transitions.")
    print("With GPT-2 as the LM, output is fluent English indistinguishable")
    print("from model-generated text, approaching zero KL divergence.")
    print()
    print("round-trip OK")


if __name__ == "__main__":
    main()
