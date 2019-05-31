"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Module for everything related to Vocabularies and Extended Vocabularies.
Can be used to create a *.vocab file from a preprocessed training pair tsv-file.

Examples:
    python vocabulary.py data/train.tsv data/train.vocab 50000
    python vocabulary.py data/cnndm_abisee_train.tsv data/cnndm_abisee.vocab 50000
"""
import argparse
import fileinput
from collections import Counter
from abc import ABC, abstractmethod

from tqdm import tqdm


class VocabularyABC(ABC):
    """
    Vocabulary abstract base class enforcing shared
    functionality between `Vocabulary` and `ExtendedVocabulary`.
    """

    @abstractmethod
    def filter_oov(self, _):
        """Replace OOV tokens in tensor with UNK"""

    def filter_oov_(self, _):
        """Replace OOV tokens in tensor with UNK inplace"""

    @property
    def PAD(self):
        """Padding ID"""
        return 0

    @property
    def SOS(self):
        """Start of sequence ID"""
        return 1

    @property
    def EOS(self):
        """End of sequence ID"""
        return 2

    @property
    def UNK(self):
        """Unknown token ID"""
        return 3

    @abstractmethod
    def __contains__(self, _):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, _):
        pass


class ExtendedVocabulary(VocabularyABC):
    """
    Class representing an Extended Vocabulary.
    Made for usage with Pointer Generator.
    An ExtendedVocabulary has a base `Vocabulary`
    and can expand it, without modifying its base.

    Interface remains the same with the addition of `add_word` for extension.
    """

    def __init__(self, base):
        self.base = base
        self.base_size = len(base)
        self.w2i = {}
        self.i2w = {}

    def filter_oov(self, tensor):
        """Filters OOV words from the given `tensor` replacing them with UNK"""
        result = tensor.clone()
        result[tensor >= self.base_size] = self.UNK
        return result

    def filter_oov_(self, tensor):
        """Filters OOV words from the given `tensor` replacing them with UNK (inplace)"""
        tensor[tensor >= self.base_size] = self.UNK
        return tensor

    def add_word(self, word):
        """
        Adds the given word to the `ExtendedVocabulary` assuming
        that it is not present in extended nor base. If it is already
        present, it will simply be ignored.

        :param word: the `str` word to be added to the extended vocab
        """
        if word not in self.base and word not in self.w2i:
            idx = self.base_size + len(self.w2i)
            self.w2i[word] = idx
            self.i2w[idx] = word

    def __contains__(self, key):
        if isinstance(key, int):
            return key in self.base or key in self.i2w

        return key in self.base or key in self.w2i

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.base[key] if key in self.base else self.i2w[key]

        return self.w2i[key] if key in self.w2i else self.base[key]

    def __len__(self):
        return self.base_size + len(self.w2i)


class Vocabulary(VocabularyABC):
    """Represent a basic Vocabulary with word2int and int2word"""

    def __init__(self, words=None, vocab_size=50000):
        """
        Creates a Vocabulary of size `vocab_size`,
        given a Counter of the occurences of tokens in the corupus.

        :param words:
            Sorted list of words in descending order of frequency or a Counter instance.
        :param vocab_size:
            Truncate vocabulary to this size, ignoring the length/size of `words`.
        """
        self.reserved = [self.PAD, self.SOS, self.EOS, self.UNK]
        self.w2i = {
            "<SOS>": self.SOS,
            "<EOS>": self.EOS,
            "<UNK>": self.UNK,
            "<PAD>": self.PAD,
        }
        init_size = len(self.w2i)
        missing = vocab_size - init_size
        if isinstance(words, Counter):
            for i, (w, _) in enumerate(words.most_common(missing), init_size):
                self.w2i[w] = i
        else:
            for i, w in enumerate(words[:missing], init_size):
                self.w2i[w] = i

        self.i2w = {v: k for k, v in self.w2i.items()}

    def filter_oov(self, tensor):
        return tensor

    def filter_oov_(self, tensor):
        return tensor

    def __contains__(self, key):
        if isinstance(key, int):
            return key in self.i2w

        return key in self.w2i

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.i2w[key]

        return self.w2i.get(key, self.UNK)

    def __len__(self):
        return len(self.w2i)


def build_vocabulary(file, vocab_size, delimiter="\t"):
    """
    Make a vocabulary by reading a preprocessed input/output pair file.

    :param file: Path to preprocessed file with lines formatted as <article><delimiter><summary>
    :param vocab_size: Size of the vocabulary to be created
    :param delimiter: The delimiter between body/summary in file
    """
    wc = Counter()
    pbar = tqdm(fileinput.input(file))
    for line in pbar:
        body, summ = line.strip().split(delimiter)
        wc.update(body.split())
        wc.update(summ.split())

    return Vocabulary(wc, vocab_size)


def load_vocabulary(file, vocab_size):
    """
    Load a premade vocabulary

    :param file: Sorted list of words, by occurences in descending order.
    :param vocab_size: Size of vocabulary to be loaded. To allow truncating.
    """
    with open(file, "r") as f:
        words = list(map(lambda w: w.strip(), f.readlines()))
        return Vocabulary(words, vocab_size)


def save_vocabulary(vocab, file):
    """
    Save a vocabulary to a file. Compatible with `load_vocabulary`.

    :param vocab: The vocabulary to save
    :param file: The path for the vocab to be saved to. Append mode.
    """
    with open(file, "a") as f:
        # We start from 4 as 0,1,2,3 are reserved.
        for i in range(len(vocab.reserved), len(vocab)):
            print(vocab.i2w[i], file=f)


def main():
    """
    Builds a vocabulary from a .tsv-file of training pairs
    and outputs the vocabulary as an ordered list to the given output file
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a test file and print its rouge scores"
    )
    parser.add_argument("tsv_file", type=str, help="path to model to evaluate")
    parser.add_argument("output_file", type=str, help="path to save vocabulary file")
    parser.add_argument(
        "vocab_size",
        type=int,
        nargs="?",
        default=50000,
        help="size of vocabulary to save",
    )
    args = parser.parse_args()
    vocab = build_vocabulary(args.tsv_file, args.vocab_size, args.filter)
    save_vocabulary(vocab, args.output_file)


if __name__ == "__main__":
    main()
