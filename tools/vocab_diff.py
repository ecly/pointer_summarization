"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Small script for comparing two vocabulary files.
Used to compare vocabulary before and after proper noun filtering.
Example:
    python vocab_diff.py vocab.txt vocab_filtered.txt
    python vocab_diff.py vocab.txt vocab_filtered.txt 1000
"""
import sys

assert len(sys.argv) >= 3, "need at least two vocabs as argument"
vocab1 = sys.argv[1]
vocab2 = sys.argv[2]
limit = 10000 if len(sys.argv) == 3 else int(sys.argv[3])

vocab1 = set(open(vocab1, "r").read().splitlines()[:limit])
vocab2 = set(open(vocab2, "r").read().splitlines()[:limit])

removed = limit - len(vocab1.intersection(vocab2))
print("Difference of size: %d" % removed)

diff1 = vocab1.difference(vocab2)
print("Removed")
print(diff1)
diff2 = vocab2.difference(vocab1)
print("Added")
print(diff2)
