# What if a wrong password gave you a plausible message instead of gibberish?

This was an idea I first had in 2008, and I finally got round to exploring it. 

# Also, rather than a substitution cypher, where you replace plaintext characters with encoded ones, what if you use English grammar itself, and the path you take through the grammar when generating a phrase, to represent the message, rather than the phrase itself?

For the observer, it's now difficult to know whether there is a message hidden, and also whether you have decrypted it successfully, because there is plain English everywhere.

I have built code around these ideas. I start from using a Chomsky phrase-structure grammar, to adding honey encryption, then using a GPT-2 encoder. Writing the code has been very informative in understanding the state of the art in steganographic encryption.

---



## The assumption hiding inside every brute-force attack

When you played Hangman as a child, you already knew the trick: in English, guess "E" first, then "T", because those are the most frequent vowel and consonant. You were doing frequency analysis without knowing the name, and leaning on the same assumption every brute-force attack makes, that you can tell when you've won.

When people picture someone cracking encryption, they picture a computer trying millions of keys a second until one works. There's an assumption buried in that picture we should examine: how the attacker knows they've won.

How do they know? Because the right key produces readable text with the right letter frequencies, and a wrong key produces nonsense or unlikely letter combinations. Decrypt with the correct key and you get `Meet me at the pier at nine`. Decrypt with a wrong one and you get `x8#Qv9k`. Structured English on one side, high-entropy noise on the other. The program doesn't need to understand the message. It just needs to notice that one candidate looks like language and the rest look like line noise.

That is the real engine of brute force, try everything until you get plausible English. So what I thought was, what if you removed it? What if every wrong key produced a grammatical, plausible sentence, so a computer going through billions of decryptions had no way to tell the real message from the decoys?

Here's the tour of the demonstrative code I produced, and the working proofs of the concepts that come from it.

---



## A short history of knowing you've won

Cryptography's first real weapon was frequency analysis. In the 9th century Al-Kindi noticed that some letters turn up more often than others in any language, so a substitution cipher leaks through its statistics. It works because real language has structure and gibberish doesn't. The attacker's edge has always been that plaintext is recognisable.

In 1883 Auguste Kerckhoffs stated the principle the field still runs on: a system should stay secure even if everything about it except the key is public. The secrecy lives in the key, nowhere else.

In 1949 Claude Shannon put the whole thing on a mathematical footing. He proved that the one-time pad, a random key as long as the message and used once, gives perfect secrecy: the ciphertext tells you nothing about the plaintext. He also gave us unicity distance, the amount of ciphertext after which only one key produces a sensible decryption. Below it, many keys give plausible messages and you genuinely can't tell which is right. Above it, the redundancy of English narrows things to one, and the computer knows it has won.

The same theme runs through all of it, from Al-Kindi to Shannon: natural language is structured, and structure is recognisable. So the counter-move is fun. Make the wrong answers look structured too.

Two lines of research have gone at exactly this:

- Linguistic steganography: hiding a message inside innocent-looking text, so nobody can even tell there is a message.
- Honey encryption (Juels & Ristenpart, 2014): building a cipher so every wrong key decrypts to a plausible fake ("honey") message, which takes away the attacker's ability to recognise success.

My idea sat at the intersection of the two, and that's what makes it doubly hard to unravel: steganography hides that there is a message at all, and honey encryption makes sure that even if you go looking, every key you try hands back a plausible one.

---



## Hiding the message in the grammar itself

Here's the core of it. Take a Chomsky phrase-structure grammar, expressed
in BNF (Backus-Naur Form). This gives a set of rules for building sentences, like `SENTENCE → NOUN VERB OBJECT`, where each slot has several choices. Every sentence you build is a path through the grammar: which noun, which verb, which object. Here's the clever part. The actual words you land on ("TREE", "RUN", "SWING") don't matter at all, only the path does. So the path carries the message. "First option" is `0`, "second option" is `1`, and each choice encodes one bit. String enough choices together and the path through the grammar *is* your encoded message, while the sentence it produces reads as ordinary plaintext. Intercept it, or even dump it straight out of the computer's memory, and it just looks like a sentence.

This is an old idea. Peter Wayner called them "mimic functions" in 1992 (P. Wayner, "Mimic Functions," *Cryptologia* 16(3), 1992). My own way in was a little different, and I arrived at it before I knew the literature. Those old Unix programs that generated syntactically correct sentences from a phrase-structure grammar had stuck with me. Start from a secret number, use it to seed a pseudo-random generator (Unix `srand()`), and let the generator choose the path through the grammar. Anyone with the same secret walks the same path and recovers the message. Anyone without it just sees an ordinary sentence. The grammar is public, which Kerckhoffs would like; only the seed is secret.

That gives you something quite nice. The cover text is always grammatical, and a wrong secret gives you a different but equally grammatical sentence. No gibberish, and nothing for a brute-force detector to grab onto.

I built this as a working keyed phrase-structure-grammar (PSG) encoder, then hardened it the way you would harden any real tool:

- A proper key-derivation function (Argon2id) so the secret can be a human passphrase without being cheap to guess.
- A per-message nonce (a public one-time number) so the same secret can send many messages safely.
- A keystream XORed over the message bits, so the encoding is driven by uniform-looking data whatever you are actually sending.

At that point I had a satisfying toy. Which is exactly when its limits got interesting.

---



## From grammar to language model

A hand-written grammar has two problems. First, capacity: my expanded grammar squeezed about 45 bits, under six bytes, into a sentence. Second, it reads like a hand-written grammar. The sentences are correct but oddly uniform, and the word frequencies don't match real English. A careful reader, or a statistical detector, would notice.

The fix is a natural escalation, and climbing that ladder was the most instructive part of the whole thing:

1. Frequency matching. I added Huffman coding driven by word frequencies, so common words get short codes and the output statistics start to resemble a target distribution. Better, but still tied to a small fixed vocabulary.
2. Swap the grammar for a language model. A grammar is a crude model of language. A language model is a much better one. So I replaced the grammar first with a simple bigram model (trained on *Pride and Prejudice*, because it was to hand) and then with GPT-2 itself. Now the thing driving the encoding is a real neural network's sense of what word comes next, and the cover text is fluent, varied English.

Here's an actual sentence one of the GPT-2 versions produced, secretly carrying the message `Rendezvous!`:

> The weather report for this weekend says that there's no doubt that this sea level rise is the signal that sea levels are rising on an unprecedented scale...

Nothing about it looks like ciphertext. It reads like a slightly rambling weather post. That's the point.

But the most important step wasn't the model. It was how the bits get encoded into the model's choices.

---



## Huffman coding leaks

This is the part that surprised me most, and it's the most useful thing I took away.

When you use Huffman coding to turn a message into word choices, you're forcing each word to appear with a probability that's a power of a half: 1/2, 1/4, 1/8, and so on. A real language model doesn't work like that. It might put the next word at probability 0.7. Huffman can only approximate that, so the text you generate has slightly wrong statistics. Not wrong enough for a person to notice, but a machine-learning detector trained to spot the difference certainly can.

You can measure the leak. Using KL divergence, a standard measure of how far one probability distribution sits from another (zero means identical, and undetectable), the Huffman approach leaked about 0.013 bits per word against the model's true distribution. Small, but not zero. In security, not-zero is where the attackers live.

The fix is a neat recent technique called Discop (Ding et al., 2023). Instead of *coding* the message, you *sample* the next word the way the model normally would, but you rotate the sampling wheel and let the next payload bit choose which copy you read. A rotation doesn't change how much of the wheel each word takes up, so the word frequencies come out exactly matching the model. The message hides in which copy you used, and that doesn't show up in the statistics. The dart that lands on the wheel is public — derived from a nonce anyone can see — so every key agrees on where the bits sit. Secrecy lives only in a mask on the payload, not in a second secret sampling tape.

When I swapped Huffman for Discop, the measured KL divergence dropped from 0.013 to about 0.00000000000003 bits per word, essentially the noise floor of 32-bit arithmetic. From "small but real leak" to "provably nothing" *relative to that sampler*.

The insight that stuck with me: the security doesn't come from the message being random. For any message, even a completely predictable one, the emitted text follows the deployed sampler exactly, because the innocent channel and the hidden-message channel share the same public tape; the secret only whitens the payload. That took me a while to get my head round, and it's the crux of why the scheme is provably secure against that baseline.

---



## What building it taught me

Building the thing, rather than just reading about it, turned up lessons the papers don't quite spell out.

1. Clean theory, messy tokenizers. GPT-2 doesn't work in words, it works in tokens, sub-word chunks. It turns out that decoding a sequence of tokens to text and re-encoding it doesn't always give you back the same tokens. About one generation in nineteen hit this, and because the receiver replays the public tape over the re-tokenized text, a single mismatch desynchronised everything after it. The maths was perfect; the byte-level reality broke it. The fix (borrowed from the Meteor system) is to check at each step that a candidate word survives the round-trip before committing to it, a check both sides can compute the same way. A cryptographic scheme is only as correct as its least glamorous implementation detail.
2. "Secure" means nothing until you say against what. When I built an evaluation harness, generating hundreds of messages and running a detector over them, the first question that mattered wasn't "is it secure?" but "secure relative to which idea of normal text?" Measured against the actual GPT-2 sampler we deploy, the hidden-message text and innocent channel text were statistically indistinguishable: a trained detector scored an AUC near 0.5, with confidence intervals that include 0.5 at my sample sizes, so no message leakage I could demonstrate. Against ordinary platform text that doesn't use our self-tokenizing restriction, a couple of detectors do pick up a small gap -- and, importantly, that gap was the same whether or not a message was present. It's a property of how we sample, not of the hiding. Undetectability is always relative to a stated baseline, and being honest about that baseline is the whole game.
3. The gap between "information-theoretic" and "computational" is where the honesty lives. With a truly random public tape and an independent secret mask, the scheme reaches what Christian Cachin's 1998 model calls perfect security *relative to the deployed sampler*: zero divergence, zero advantage for any warden, even one with unlimited computing power. In practice the mask is a pseudo-random function of the derived key, which downgrades "perfect" to "computationally secure" unless you can break that function. The sampling coins stay public, so undetectability is not hiding them. Writing the proof forced me to state exactly which of the two I was claiming at each step. The proof isn't the ceremony at the end; it's the thing that stops you fooling yourself.
4. Authentication is a trap here. Every instinct says add a checksum so the receiver knows the message is intact. But a checksum would hand the attacker the very thing we worked to remove: a way to recognise the correct key offline. So the scheme deliberately has no message authentication. Integrity has to live in the transport layer, under a separate key. Sometimes a "best practice" is actively wrong for your threat model.
5. There's no free lunch on capacity. The provably-secure Discop sampling, in its simplest form, carries only about one bit per word, far less than Huffman's dozen-plus. You pay for undetectability in bandwidth. That trade-off is fundamental, not incidental.
6. The decoys are only as good as your model of "normal messages". This is the caveat I'm keenest not to oversell. The scheme makes every wrong key produce a plausible message, but "plausible" means "plausible according to the decoy model you built in". When the real messages genuinely look like that model, the wrong-key decoys are statistically identical to the true message and a brute-force attacker is stuck: in my tests a detector's ability to pick the real one out sat at a coin flip. But when the real messages are richer than the decoy model, a strong enough judge can start telling them apart. That model-versus-reality gap is *not* the "ε" in the deniability theorem — ε is how faithfully the encoder samples its own decoy model, a tiny arithmetic error. The Gutenberg residual is a different quantity, and it's the honest limit of the whole idea.
   A better model helps, but not in the simple way I'd hoped. Swapping a tiny word-level bigram for DistilGPT-2 cut the judge's ability to separate decoys from real English from near-total (separability 0.50) down to about 0.39–0.43 on a held-out multi-genre corpus -- without wiping it out. And the residual flipped direction: the neural decoys came out *more* fluent than the human prose I scored them against. One epoch of continued pretraining on a frozen train split then cut *holdout* separability further to about 0.10–0.13 -- still clearly above chance, so real-English deniability is not claimed. Raising the fidelity of the decoy model doesn't just shrink the gap; it can change which side of it the decoys sit on. A model whose samples are more typical than human writing is as separable as one whose samples are less so. Deniability is only as strong as how well your decoy distribution matches real traffic. Closing the rest means a still better message model, and checks with human raters and a stronger judge.

---



## Where this leaves it

None of the ingredients are mine to claim. Wayner's grammars, Juels and Ristenpart's honey encryption, Cachin's security model, Meteor, Discop and the wider provably-secure-steganography community did the deep work. What I set out to do was join them up end to end, from a scrappy secret-seed-drives-a-grammar idea to a GPT-2 encoder with a formal security proof, and build every step so I could see where the theory met the road. The claim is both games in one definition, a public tape so every key reads the same bits, and a first look at whether the decoys survive a real-English judge — not a new cipher.

What you end up with is an odd object: a way to encode a secret so that a computer trying to brute-force it finds a plausible, fluent, innocent-looking message behind every key it tries, the real one included, and, as far as your decoy model matches real traffic (lesson 6), indistinguishable in the pile. It chips away at the thousand-year-old assumption that the attacker can tell when they've won.

It's a research toy and a learning exercise, not a product. It makes no promises against active tampering, reused nonces, side channels, or coercion — if they have the real secret, the real message comes out — and I've tried to be careful to say so.

I worked in a university computer science department years ago, and I'm writing this up properly as an academic paper (because why not). It's been the most I've learned from a project in a long time: the difference between hiding a message and hiding that there is a message, and the gap between a clean proof and the messy implementation that has to carry it.

You may wonder what retired people do. Some of us, it turns out, reinvent honey encryption for fun. :-)

---

*The proofs of concept cover a keyed grammar encoder, a frequency-matched version, bigram and GPT-2 samplers, the Discop zero-leak sampler, an evaluation harness, and a formal security write-up under Cachin's model. I'd be interested to hear what the computer science academics and cryptographers here make of it, and I'm happy to go into any part of it in the comments.*