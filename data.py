"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Utility classes/function for handling data.
In particular, provides the `Dataset` implementation, which
includes the custom batch generator, that can create an
`ExtendedVocabulary` for the batch being generated.
"""
import time
from pathlib import Path
from collections import namedtuple
import random

import torch

from vocabulary import VocabularyABC, ExtendedVocabulary


class Batch:
    """A single batch fed to the model"""

    def __init__(self, examples, inputs, targets, input_lengths, target_lengths, vocab):
        """
        A batch of examples.
        The inputs represent the vocab ids of the example's sources, and
        the targets represent the vocab ids of the example's targets.
        The ids should be produced using the given vocabulary, which may
        be either an instance of `Vocabulary` or `ExtendedVocabulary`.
        """
        self.examples = examples
        self.inputs = inputs
        self.targets = targets
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        self.input_mask = self.make_mask(input_lengths)
        self.target_mask = self.make_mask(target_lengths)
        self.vocab = vocab

    @staticmethod
    def make_mask(lengths):
        """
        Create a padding mask from an ordered list of lengths.

        :param lengths: The lengths of the batch of size (batch_size)
        :returns: A padding mask of size (batch_size, max_len)
        """
        batch_size = len(lengths)
        mask = torch.zeros(batch_size, lengths.max())
        for i, length in enumerate(lengths):
            mask[i, :length] = torch.ones(length)

        return mask

    def to(self, device):
        """Move all tensors in batch to given `device`"""
        self.inputs = self.inputs.to(device)
        self.input_mask = self.input_mask.to(device)
        self.targets = self.targets.to(device)
        self.target_mask = self.target_mask.to(device)
        self.input_lengths = self.input_lengths.to(device)
        self.target_lengths = self.target_lengths.to(device)

    def __len__(self):
        return len(self.examples)


def gensim_word_to_id(model, word):
    """Gets the index of a word in the given gensim model"""
    return model.vocab[word].index if word in model.vocab else 0


def words_to_ids(model, words):
    """Support for both a gensim model and a Vocabulary"""
    if isinstance(model, VocabularyABC):
        return torch.LongTensor(list(map(lambda w: model[w], words)))

    return torch.LongTensor(list(map(lambda w: gensim_word_to_id(model, w), words)))


Example = namedtuple("Example", "src tgt")


class Dataset:
    """Dataset creating input pairs from a file and a vocabulary"""

    def __init__(self, filename, vocab, cfg, evaluation=False):
        """
        :param filename: Path .tsv-file containing training pairs.
        :param vocab: `Vocabulary` instance used for creating Examples.
        :param cfg: `Config` instance to retrieve configuration from eg. truncation
        :param evaluation: If `True`, we will never filter the instances
        """
        # Store for reference
        self.filename = filename
        self.truncate_article = cfg.truncate_article
        self.truncate_summary = cfg.truncate_summary
        self.limit = cfg.limit

        self.filter_instances = cfg.filter_instances and not evaluation
        if self.filter_instances:
            self.min_article_length = cfg.min_article_length
            self.max_article_length = cfg.max_article_length
            self.min_summary_length = cfg.min_summary_length
            self.max_summary_length = cfg.max_summary_length
            self.min_compression_ratio = cfg.min_compression_ratio

        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.examples = self._make_examples()
        self.length = len(self.examples)

    def __len__(self):
        return self.length

    def filter(self, article_words, summary_words):
        """Returns `True` if instance should be filtered, otherwise `False`"""
        art_len = len(article_words)
        if art_len < self.min_article_length or art_len > self.max_article_length:
            return True

        sum_len = len(summary_words)
        if sum_len < self.min_summary_length or sum_len > self.max_summary_length:
            return True

        compression_ratio = art_len / sum_len
        if compression_ratio < self.min_compression_ratio:
            return True

        return False

    def _make_examples(self):
        """
        Creates a list of `Example` namedtuples.
        Instances have truncation applied based on `self.cfg`.
        Instances are similarly filtered based on min/max lengths from the cfg.

        :returns: A list of `Example` instances
        """
        start = time.time()
        examples = []
        filtered = 0
        with open(self.filename, "r") as input_:
            count = 0
            for line in input_:
                if count >= self.limit:
                    break

                art_words, sum_words = (x.split() for x in line.split("\t"))
                if self.filter_instances and self.filter(art_words, sum_words):
                    filtered += 1
                    continue

                # We don't bother storying the full article since
                # it consumes an awfully lot of memory - the full
                # summary is however needed for evaluation
                art_words = art_words[: self.truncate_article]

                examples.append(Example(art_words, sum_words))
                count += 1

        build_time = time.time() - start
        basename = Path(self.filename).stem
        print(
            "Built dataset for %s in %ds - size: %d, filtered: %d"
            % (basename, build_time, len(examples), filtered)
        )
        return examples

    def _create_ext_vocab(self, sources):
        """
        Create and return an extended Vocabulary based on the
        Vocabulary `self.vocab` and the given `targets`

        :param sources: A list of lists of words
        """
        ext_vocab = ExtendedVocabulary(self.vocab)
        for source in sources:
            for word in source:
                ext_vocab.add_word(word)

        return ext_vocab

    def generator(self, batch_size, ext_vocab=False, shuffle=True):
        """
        A custom generator returning a Batch of given `batch_size`
        The generator operates on one epoch at a time.

        :param batch_size: The size of the batches to generate.
        :param ext_vocab:
            Boolean indicate whether to generate an extended
            vocabulary to store ids of OOV words. Used for
            Pointer Generator at batch level.
        :param shuffle: Whether or not to shuffle examples prior to yielding
        """
        if shuffle:
            random.shuffle(self.examples)

        for ptr in range(0, len(self.examples), batch_size):
            examples = self.examples[ptr : ptr + batch_size]
            # sort by order descending for padding in RNNs
            examples.sort(key=lambda e: len(e.src), reverse=True)

            # make truncated versions of sources and targets for embedding
            sources = list(map(lambda e: e.src[: self.truncate_article], examples))
            targets = list(map(lambda e: e.tgt[: self.truncate_summary], examples))

            vocab = self.vocab
            if ext_vocab:
                vocab = self._create_ext_vocab(sources)

            src_seqs = [words_to_ids(vocab, s) for s in sources]
            src_lens = torch.LongTensor(list(map(len, src_seqs)))
            src_tensor = torch.zeros(len(examples), src_lens.max(), dtype=torch.long)

            for i, (src, src_len) in enumerate(zip(src_seqs, src_lens)):
                src_tensor[i, :src_len] = src

            tgt_seqs = [words_to_ids(vocab, t) for t in targets]
            # +1 for EOS
            tgt_lens = torch.LongTensor(list(map(lambda t: len(t) + 1, tgt_seqs)))
            tgt_tensor = torch.zeros(len(examples), tgt_lens.max(), dtype=torch.long)

            for i, (tgt, tgt_len) in enumerate(zip(tgt_seqs, tgt_lens)):
                tgt_tensor[i, : tgt_len - 1] = tgt
                tgt_tensor[i, tgt_len - 1] = vocab.EOS

            yield Batch(examples, src_tensor, tgt_tensor, src_lens, tgt_lens, vocab)
