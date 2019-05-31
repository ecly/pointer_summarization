"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Preprocessing for Newsroom.
Does downcasing, basic tokenization using NLTK and fixes contractions.

We expect Newsroom data given as  *.jsonl.gz.

Example to make Newsroom test set from its *.jsonl.gz file.
    python preprocess.py data/test.jsonl.gz data/newsroom_test.tsv
"""
import os
import argparse

import contractions
from nltk.tokenize import word_tokenize

from . import jsonl


def parse(text, fix_contractions=False):
    """Parses some text into space separated tokens"""
    if fix_contractions:
        text = contractions.fix(text, slang=False)
    words = word_tokenize(text.lower())
    return " ".join(words)


def generate_pairs_newsroom(filename, fix_contractions):
    """Generate Newsroom <body\tsummary> pairs and yield them"""
    with jsonl.open(filename, gzip=True) as file:
        for entry in file:
            body = parse(entry["text"], fix_contractions)
            summary = parse(entry["summary"], fix_contractions)
            if body and summary:
                yield body + "\t" + summary
            else:
                print("Found empty article or summary, skipping.")


def prepare_arg_parser():
    """Create simple arg parser expecting 2 positional arguments"""
    parser = argparse.ArgumentParser(
        description="Preprocess Newsroom  into a .tsv-filen"
    )
    parser.add_argument(
        "input", metavar="input-path", type=str, help="path to newsroom gzip file"
    )
    parser.add_argument(
        "output",
        metavar="output-path",
        type=str,
        help="path to .tsv-file that pairs will be appended to",
    )
    parser.add_argument(
        "-f",
        "--fix",
        metavar="fix-contractions",
        action="store_true",
        help="fixes contractions if flag given",
    )

    return parser


def main():
    """Run Newsroom preprocessing and write to file"""
    args = prepare_arg_parser().parse_args()
    input_path = args.input
    output_path = args.output
    fix = args.fix

    with open(output_path, "w") as out:
        assert os.path.isfile(input_path) and input_path.endswith(".gz")
        for pair in generate_pairs_newsroom(input_path, fix):
            print(pair, file=out)


if __name__ == "__main__":
    main()
