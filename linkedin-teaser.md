I revisited an encryption idea I first had in 2008.

When you played "Hangman" as a child, you knew that in English you should start your guesses with "E" and then "T", because they are the most frequent vowel/consonant pair. Every brute-force decryption attack assumes similarly, that they can tell when they've won.

The right key gives readable English and matching letter frequencies. Every wrong key gives gibberish or unlikely letter combinations.

My idea was that using a pre-shared secret and a copy of a Chomsky Phrase Structure Grammar definition, I could use the grammar path traversal NOUN-VERB-OBJECT etc. based on the key to encode a message. The actual nouns/verbs/object values ("TREE", "RUN", "SWING") don't matter, just the choice in the grammar path. Each choice encodes one bit of information. And all the intermediate text looks like plaintext. Hard to decode if you intercept the computer's memory, eh?

I updated it with respect to the current academic literature and built the code and model. My underlying thought was: "What if every wrong key also produced a valid English sentence? How would a computer ever know which plaintext decrypt is the real one?". You see, it gets doubly difficult to unravel.

My project runs from 9th-century frequency analysis, Shannon's one-time pad, to a GPT-2 model that hides a message inside a fluent, innocent-looking weather report. A computer sifting through billions of decryptions can't easily pick the real message out of the plausible decoys.

I built my initial idea from a PSG and found out more. Huffman coding quietly leaks about 0.013 bits per word. "Secure" means nothing until you say secure against what. One GPT-2 tokenizer quirk broke a mathematically perfect scheme about 1 time in 19. A better decoy model shrinks how far wrong-key English sits from real English, but doesn't erase the gap, and can even flip which side of it the decoys sit on.

I developed a more cryptographically secure mechanism that advances our understanding of these problems.

Full write-up and code in the comments. I'd be interested to hear what the computer science academics and cryptographers make of it.
