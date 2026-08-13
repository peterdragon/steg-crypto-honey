#!/usr/bin/env python3
"""Honey encryption via DTE-then-Encrypt (Juels & Ristenpart 2014, Fig. 3),
with Argon2id KDF hardening.

Encryption:  key = KDF(password, salt)
             S   = DTE.encode(M)
             C   = (salt, R, H(R || key) xor S)
Decryption:  key = KDF(password, salt)
             S   = C2 xor H(R || key);  M = DTE.decode(S)

Two independent defences are on display:

  * Honey encryption: decrypting under a WRONG password still yields a valid
    seed and hence a plausible "honey" message from p_m, so the attacker cannot
    recognise a correct decryption.
  * KDF hardening: each password guess must run a slow, memory-hard KDF
    (Argon2id), so the *cost per guess* is large. The two are complementary --
    hardening raises the price of a guess; honey encryption removes the ability
    to tell a right guess from a wrong one.

Argon2id needs the `argon2-cffi` package (installed in ./.venv). Run with:
    ./.venv/bin/python honey_encryption.py
If argon2 is unavailable this falls back to stdlib scrypt (also memory-hard).

Run:  python3 honey_encryption.py
"""

import hashlib
import os
import time

from inverse_sampling_dte import InverseSamplingDTE

try:
    from argon2.low_level import hash_secret_raw, Type
    _HAVE_ARGON2 = True
except ImportError:
    _HAVE_ARGON2 = False

# --------------------------------------------------------------------------
# Design note: authentication and side channels (NOT implemented in this demo)
#
# Authentication: a normal encrypted vault would use authenticated encryption
# (AES-GCM, Encrypt-then-HMAC) so a wrong key fails to verify. Honey encryption
# must do the OPPOSITE: the ciphertext is deliberately UNauthenticated so that
# every key decrypts to a valid-looking seed. A MAC/tag would be a free offline
# "is this the right key?" oracle -- exactly what we are trying to remove.
# Authentication is therefore pushed to a rate-limited ONLINE step: the attacker
# must try each decoy against the real service (the NoCrack model), where guesses
# are throttled, logged, and lockable.
#
# Side channels: a real build must not leak secrets through timing/cache/power.
# Use constant-time comparison (e.g. hmac.compare_digest, never ==) for any
# secret-dependent check, a side-channel-resistant KDF (Argon2id -- chosen here
# partly for that reason) and cipher (hardware AES-CTR), and uniform error/timing
# on decryption so "wrong password" and "bad format" are indistinguishable.
# This teaching demo uses plain Python comparisons/dict lookups and so is NOT
# constant-time.
# --------------------------------------------------------------------------

# Argon2id cost parameters (RFC 9106 style; 64 MiB memory-hard).
ARGON2_TIME_COST = 3
ARGON2_MEMORY_KIB = 65536   # 64 MiB
ARGON2_PARALLELISM = 4
KEY_LEN = 32


def derive_key(password, salt):
    """Password + salt -> 32-byte key via a slow, memory-hard KDF."""
    if _HAVE_ARGON2:
        return hash_secret_raw(
            password.encode(), salt,
            time_cost=ARGON2_TIME_COST, memory_cost=ARGON2_MEMORY_KIB,
            parallelism=ARGON2_PARALLELISM, hash_len=KEY_LEN, type=Type.ID)
    # Stdlib fallback: scrypt is memory-hard too (used if argon2-cffi absent).
    return hashlib.scrypt(password.encode(), salt=salt,
                          n=2 ** 14, r=8, p=1, dklen=KEY_LEN)


def kdf_name():
    return "Argon2id (argon2-cffi)" if _HAVE_ARGON2 else "scrypt (stdlib fallback)"


def _pad(nonce, key_bytes, nbits):
    """Keystream H(nonce || key) expanded to nbits bits."""
    out = b""
    counter = 0
    while len(out) * 8 < nbits:
        out += hashlib.sha256(nonce + key_bytes + counter.to_bytes(4, "big")).digest()
        counter += 1
    value = int.from_bytes(out, "big")
    return value >> (len(out) * 8 - nbits)


class HoneyEncryption:
    def __init__(self, dte):
        self.dte = dte
        self.ell = dte.ell

    def encrypt(self, password, message):
        salt = os.urandom(16)
        key = derive_key(password, salt)
        seed = self.dte.encode(message)
        nonce = os.urandom(16)
        c2 = seed ^ _pad(nonce, key, self.ell)
        return salt, nonce, c2

    def decrypt(self, password, ciphertext):
        salt, nonce, c2 = ciphertext
        key = derive_key(password, salt)
        seed = (c2 ^ _pad(nonce, key, self.ell)) & (self.dte.top - 1)
        return self.dte.decode(seed)


def _human_time(seconds):
    for unit, size in (("years", 365 * 24 * 3600), ("days", 24 * 3600),
                       ("hours", 3600), ("minutes", 60)):
        if seconds >= size:
            return f"{seconds / size:.1f} {unit}"
    return f"{seconds:.2f} seconds"


def main():
    dist = {"1234": 1000, "0000": 400, "1111": 250, "4321": 60,
            "2580": 40, "9999": 30, "7777": 12, "8068": 1}
    he = HoneyEncryption(InverseSamplingDTE(dist, ell=32))

    print(f"KDF in use    : {kdf_name()}")
    true_key = "correct horse"
    true_msg = "8068"
    ct = he.encrypt(true_key, true_msg)
    print(f"true password : {true_key!r}")
    print(f"true PIN      : {true_msg}")
    print(f"ciphertext    : salt={ct[0].hex()[:12]}.. nonce={ct[1].hex()[:12]}.. c2={ct[2]}")

    print(f"\ndecrypt with correct password -> {he.decrypt(true_key, ct)}")

    print("\ndecrypt with WRONG passwords -> plausible decoy PINs:")
    for guess in ["password1", "hunter2", "letmein", "qwerty", "swordfish",
                  "dragon", "iloveyou", "monkey"]:
        print(f"  {guess:>12} -> {he.decrypt(guess, ct)}")

    # KDF hardening: measure the cost of ONE guess and extrapolate.
    salt = os.urandom(16)
    t0 = time.perf_counter()
    derive_key("timing probe", salt)
    per_guess = time.perf_counter() - t0
    print(f"\nKDF hardening cost: {per_guess * 1000:.1f} ms per password guess")
    guesses_per_sec = 1.0 / per_guess
    print(f"  -> ~{guesses_per_sec:,.0f} guesses/sec single-threaded (vs. billions/sec for a bare hash)")
    for label, space in (("a 10^6 dictionary", 10 ** 6),
                         ("a 10^9 dictionary", 10 ** 9)):
        print(f"  -> brute-forcing {label}: {_human_time(space * per_guess)}")
    print("  (and even after paying that, honey decoys hide which guess was right)")


if __name__ == "__main__":
    main()
