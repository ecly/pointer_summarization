"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Simple script for producing a summary for a single input article through stdin,
using somewhat hacky and ineffective means to utilize existing infrastructure.
The resulting summary will be printed to stdout.

Examples:
    python test.py < article.txt
    python test.py log/base.tar <<< \
        "hello simon how are you today ? simon was doing good that day .
         it was a rainy day . he was wearing shoes. good for him .
         when he stepped in a puddle , his shoes got wet .
         he then put his shoes in the oven to dry them .
         that worked nicely . simon now has warm , slightly charred shoes .
         the following day , the other kids told simon that they looked cool .
         simon was indeed doing good ."
"""
import sys
import os
import argparse
import tempfile
import evaluate
import tools.preprocess_newsroom as preprocess
import train
from data import Dataset
from beam_search import BeamSearch
from util import suppress_stdout_stderr


def prepare_arg_parser():
    """Create simple arg parser expecting 1 positional arguments"""
    parser = argparse.ArgumentParser(
        description="Generate summary for given model for artice from stdin"
    )
    parser.add_argument(
        "model_file",
        metavar="model-path",
        type=str,
        help="path to model used to generate summary",
    )

    return parser


def main():
    """
    Creates a temporary file for the given input which is
    used to create a dataset, that is then evaluated on the given model.
    The generated summary is printed to standard out.
    """
    args, unknown_args = prepare_arg_parser().parse_known_args()
    model_file = args.model_file

    with suppress_stdout_stderr():
        model, _optimizer, vocab, _stats, cfg = train.load_model(
            model_file, unknown_args
        )

    _, filename = tempfile.mkstemp()
    try:
        with open(filename, "a") as f:
            input_ = sys.stdin.read()
            article = preprocess.parse(input_)
            print(f"{article}\tSUMMARY_STUB", file=f)

        with suppress_stdout_stderr():
            dataset = Dataset(filename, vocab, cfg)

        batch = next(dataset.generator(1, cfg.pointer))

        # don't enforce any min lengths (useful for short cmdline summaries")
        setattr(cfg, "min_summary_length", 1)
        bs = BeamSearch(model, cfg=cfg)
        summary = evaluate.batch_to_text(bs, batch)[0]
        print(f"SUMMARY:\n{summary}")
    finally:
        os.remove(filename)


if __name__ == "__main__":
    main()
