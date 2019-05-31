"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Small script for finding outliers in a dataset.
Currently supports bin files from See et al. 2017,
and .tsv files of form <article>\t<summary>.

Tweak main function to find outliers according to the desired
definition of outlier.
"""
import sys
import fileinput
import struct

# pylint: disable=no-name-in-module
from tensorflow.core.example import example_pb2


def tsv_generator(file):
    """For using files created by preprocess.py"""
    for line in fileinput.input(file):
        article, summary = line.strip().split("\t")
        yield (article, summary)


def bin_generator(file):
    """
    For using the files provided by See et al. 2017 found at:
    https://github.com/abisee/cnn-dailymail
    """
    with open(file, "rb") as reader:
        while True:
            len_bytes = reader.read(8)
            if not len_bytes:
                break
            str_len = struct.unpack("q", len_bytes)[0]
            example_str = struct.unpack("%ds" % str_len, reader.read(str_len))[0]
            example = example_pb2.Example.FromString(example_str)
            article = example.features.feature["article"].bytes_list.value[0].decode()
            summary = example.features.feature["abstract"].bytes_list.value[0].decode()
            summary = summary.replace("<s>", "")
            summary = summary.replace("</s>", "")
            yield (article, summary)


def main():
    """main"""
    file = sys.argv[1]
    assert file.endswith(".bin") or file.endswith(".tsv")
    generator = bin_generator(file) if file.endswith(".bin") else tsv_generator(file)

    outliers = []
    for (article, summary) in generator:
        article_length = len(article.split())
        summary_length = len(summary.split())
        c = article_length / summary_length
        if c < 1.0:
            outliers.append((article, summary))

    outliers.sort(key=lambda x: len(x[0].split()) / len(x[1].split()))


if __name__ == "__main__":
    main()
