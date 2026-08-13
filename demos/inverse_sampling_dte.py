#!/usr/bin/env python3
"""Inverse-sampling DTE (Juels & Ristenpart 2014, Section 4 / Appendix B).

A distribution-transforming encoder maps a message drawn from a known
distribution p_m to a near-uniform seed, and decodes a uniform seed back to a
message distributed like p_m. This is the inverse-CDF trick with fixed-point
seeds in [0, 2**ELL):

  - each message M_i owns the seed sub-interval [F(M_{i-1}), F(M_i)),
    whose width is proportional to p_m(M_i);
  - encode(M_i) picks a uniform seed inside M_i's interval;
  - decode(S) returns the message whose interval contains S.

Run:  python3 inverse_sampling_dte.py
"""

import bisect
import collections
import random

from _common import int_to_bits, bits_to_int


class InverseSamplingDTE:
    def __init__(self, distribution, ell=32):
        """distribution: {message: weight}. ell: seed width in bits."""
        self.ell = ell
        self.top = 1 << ell
        self.messages = list(distribution.keys())
        total = sum(distribution.values())

        # Scale cumulative weights to integer boundaries in [0, 2**ell].
        self.lo = []
        self.hi = []
        cum = 0
        for m in self.messages:
            lo = (cum * self.top) // total
            cum += distribution[m]
            hi = (cum * self.top) // total
            self.lo.append(lo)
            self.hi.append(hi)
        self.hi[-1] = self.top  # absorb rounding remainder
        if any(self.hi[i] <= self.lo[i] for i in range(len(self.messages))):
            raise ValueError("ell too small for this distribution; increase it")

    def encode(self, message):
        i = self.messages.index(message)
        return random.randrange(self.lo[i], self.hi[i])

    def decode(self, seed):
        i = bisect.bisect_right(self.lo, seed) - 1
        return self.messages[i]

    def encode_bits(self, message):
        return int_to_bits(self.encode(message), self.ell)

    def decode_bits(self, bits):
        return self.decode(bits_to_int(bits))


def main():
    random.seed(1)
    # A deliberately skewed distribution (e.g. user-chosen PINs).
    dist = {"1234": 1000, "0000": 400, "1111": 250, "4321": 60,
            "2580": 40, "9999": 30, "7777": 12, "8068": 1}
    dte = InverseSamplingDTE(dist, ell=32)

    print("round-trip (encode then decode a specific message):")
    for m in dist:
        assert dte.decode(dte.encode(m)) == m
    print("  all messages round-trip OK")

    print("\ndecoding 100000 uniform random seeds -> empirical distribution:")
    counts = collections.Counter(dte.decode(random.randrange(dte.top))
                                 for _ in range(100000))
    total = sum(dist.values())
    print(f"  {'msg':>6} {'target%':>8} {'seen%':>8}")
    for m in dist:
        target = 100 * dist[m] / total
        seen = 100 * counts[m] / 100000
        print(f"  {m:>6} {target:8.2f} {seen:8.2f}")
    print("\nSeeds are ~uniform, yet decode reproduces p_m: this is what makes")
    print("every wrong-key decryption look like a plausible sample.")


if __name__ == "__main__":
    main()
