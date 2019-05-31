"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Evaluation script for a Seq2Seq model.
Prints average ROUGE scores to stdout, and saves configuration and results to CSV-file.
Optionally supports evaluation with 'py-rouge', reducing external dependencies.
All summaries can be saved in JSON format using `-s` or `--save`.

Example:
    python evaluate.py log/summarization.tar data/test.tsv
    python evaluate.py log/summarization.tar data/test.tsv --limit 100
    python evaluate.py log/summarization.tar data/test.tsv --limit 100 --use_python
    python evaluate.py log/summarization.tar data/test.tsv --limit 100 -s
"""
import os
import argparse
import math
import tempfile
import json
from pathlib import Path

import torch
import rouge
import pyrouge
import nltk
from tqdm import tqdm

from data import Dataset
from beam_search import BeamSearch
from util import (
    make_log_dict,
    log_results,
    suppress_stdout_stderr,
    save_summaries,
    flatten_scores,
)
import train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rouge:
    """Make `Rouge155` input/output compatible with py-rouge's `Rouge`"""

    def __init__(self, use_python=False):
        self.use_python = use_python

    @staticmethod
    def split_sentences(text, language="english"):
        """
        Rouge libraries expect one sentence per line.
        As such this splits a text of sentences into this format
        using nltk's sent_tokenize.

        :param text: The text to divide by sentences
        :param language: The language of the given text
        """
        return "\n".join(nltk.sent_tokenize(text, language))

    @staticmethod
    def _get_scores_perl(hypothesis, references):
        # get path to rouge based on __file__ path
        file_path = os.path.dirname(os.path.realpath(__file__))
        r = pyrouge.Rouge155(os.path.join(file_path, "tools/ROUGE-1.5.5"))
        ref_dir = tempfile.mkdtemp()
        hyp_dir = tempfile.mkdtemp()
        for idx, (ref, hyp) in enumerate(zip(references, hypothesis)):
            ref_file = os.path.join(ref_dir, "%06d_reference.txt" % idx)
            hyp_file = os.path.join(hyp_dir, "%06d_hypothesis.txt" % idx)
            with open(ref_file, "w") as rf, open(hyp_file, "w") as hf:
                rf.write(ref)
                hf.write(hyp)

        # model is gold standard, system is hypothesis
        r.model_dir = ref_dir
        r.system_dir = hyp_dir
        r.model_filename_pattern = "#ID#_reference.txt"
        # pylint: disable=anomalous-backslash-in-string
        r.system_filename_pattern = "(\d+)_hypothesis.txt"

        output = r.convert_and_evaluate()
        output = r.output_to_dict(output)
        return {
            "rouge-1": {
                "p": output["rouge_1_precision"],
                "r": output["rouge_1_recall"],
                "f": output["rouge_1_f_score"],
            },
            "rouge-2": {
                "p": output["rouge_2_precision"],
                "r": output["rouge_2_recall"],
                "f": output["rouge_2_f_score"],
            },
            "rouge-l": {
                "p": output["rouge_l_precision"],
                "r": output["rouge_l_recall"],
                "f": output["rouge_l_f_score"],
            },
        }

    @staticmethod
    def _get_scores_python(hypothesis, references):
        """Note: py-rouge mixes up recall/precision"""
        return rouge.Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=False,
            apply_avg=True,
            alpha=0.5,  # Default F1_score
            stemming=True,
            ensure_compatibility=True,
        ).get_scores(hypothesis, references)

    def get_scores(self, hypothesis, references):
        """
        Get rouge scores as a dict of format:
        {"rouge-1": {"p": 0.5,
                     "r": 0.3,
                     "f": 0.4},
         "rouge-2: ...,
         "rouge-l: ...
        }

        :param references: A list of reference summaries
        :param hypothesis: A list of corresponding summaries
        :returns: A dictionary with the rouge scores
        """
        refs_split = list(map(self.split_sentences, references))
        hyps_split = list(map(self.split_sentences, hypothesis))
        if self.use_python:
            return self._get_scores_python(hyps_split, refs_split)

        with suppress_stdout_stderr():
            return self._get_scores_perl(hyps_split, refs_split)


def print_scores(scores):
    """
    Pretty print rouge scores to stdout

    :param scores: The scores output by `Rouge().get_scores`

    :returns: None
    """
    for r in sorted(scores.keys()):
        precision = scores[r]["p"] * 100
        recall = scores[r]["r"] * 100
        f1 = scores[r]["f"] * 100
        print(r.upper() + ":")
        print("\t Precision: %.2f" % precision)
        print("\t Recall: %.2f" % recall)
        print("\t F1: %.2f" % f1)


def batch_to_text(bs, batch):
    """
    Utility function to create textual output for a `batch` using given
    `BeamSearch` instance.

    :param model:
        Seq2Seq model used to generate the summaries optionally wrapped in DataParallel
    :param bs: `BeamSearch` instance used for generating summaries
    :param batch: `Batch` instance used for beam search

    :returns:
        A resulting list of summaries as strings.
    """
    vocab = batch.vocab
    outputs = []
    for ids in bs.search(batch):
        tokens = []
        # skip the first SOS if it is present
        if ids[0] == vocab.SOS:
            ids = ids[1:]

        for i in ids:
            # if EOS was produced, we don't care about the rest
            if i == vocab.EOS:
                break
            tokens.append(vocab[i])

        outputs.append(" ".join(tokens))

    return outputs


def generate_summaries(model, dataset, cfg, limit=math.inf, shuffle=False, pbar=None):
    """
    Generate summaries using the given `model` on the given `dataset`.
    Expects the given model to be in eval mode.


    :param model: Use this model for evaluation
    :param dataset: The dataset to evaluate on
    :param cfg:
        The `Config` used for the given model from which we get
        info on whether it uses pointer generation or not.
    :param limit: Limit the pairs evaluated to this many
    :param shuffle: Whether to shuffle the dataset before yielding batches
    :param pbar: Optional pbar (tqdm) to update with progress
    """
    batch_size = 1  # beam_search currently only supports batch_size 1
    bs = BeamSearch(model, cfg=cfg)
    with torch.no_grad():
        generator = dataset.generator(batch_size, cfg.pointer, shuffle)
        references = []
        hypothesis = []
        for idx, batch in enumerate(generator):
            hyps = batch_to_text(bs, batch)
            refs = [" ".join(e.tgt) for e in batch.examples]
            hypothesis.extend(hyps)
            references.extend(refs)

            if batch_size * idx >= limit:
                break

            if pbar is not None:
                pbar.update(batch_size)

        pbar.close()
        return (hypothesis, references)


def evaluate(model, dataset, cfg, use_python=False, **kwargs):
    """
    Evaluate the `model` on the given `dataset`.

    :param model: Use this model for evaluation
    :param dataset: The dataset to evaluate on
    :param cfg:
        The `Config` used for the given model from which we get
        info on whether it uses pointer generation or not.
    :param limit: Limit the pairs evaluated to this many
    :param shuffle: Whether to shuffle the dataset before yielding batches
    :param progress: Print decode progress to stdout if True
    :param use_python: Use py-rouge instead of pyrouge (Perl 155 wrapper) if True
    """
    hypothesis, references = generate_summaries(model, dataset, cfg, **kwargs)
    return Rouge(use_python=use_python).get_scores(hypothesis, references)


def evaluate_json(filename, output, use_python=False):
    """
    Evaluate a JSON file of references and hypothesis as stored
    by `util.save_summaries`.

    :param filename: Path of the JSON file to evaluate
    :param output: Path to store the CSV results
    :param use_python:
        Evaluate with original ROUGE Perl implementation if `False` (default),
        otherwise uses py-rouge reimplementation.

    :returns: The resulting ROUGE scores.
    """
    with open(filename, "r") as f:
        print("Loading JSON...")
        data = json.load(f)
        summaries = data["summaries"]
        hyps, refs = list(
            zip(*(map(lambda i: (i["hypothesis"], i["reference"]), summaries)))
        )
        print("Getting ROUGE scores...")
        scores = Rouge(use_python=use_python).get_scores(hyps, refs)

        log_dict = data["log_dict"]
        log_dict["eval_package"] = "py-rouge" if use_python else "pyrouge"
        log_dict = {**log_dict, **flatten_scores(scores)}
        print("Saving results to %s" % output)
        log_results(log_dict, output)
        print_scores(scores)


def prepare_arg_parser():
    """Create simple arg parser expecting 2 positional arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a test file and print its rouge scores"
    )
    parser.add_argument(
        "model_file", metavar="model-path", type=str, help="path to model to evaluate"
    )
    parser.add_argument(
        "test_file",
        metavar="test-path",
        type=str,
        nargs="?",
        default="data/cnndm_abisee_test.tsv",
        help="path to .tsv-file with test pairs",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="don't save the references/decoded summaries in a json file",
    )
    parser.add_argument(
        "--use_python",
        action="store_true",
        help="use python ROUGE imlpementation instead of official Perl 155 implementation",
    )

    return parser


def main():
    """Run evaluation on given test_file and print ROUGE scores to console"""
    args, unknown_args = prepare_arg_parser().parse_known_args()
    model_file = args.model_file
    test_file = args.test_file

    model, _optimizer, vocab, stats, cfg = train.load_model(model_file, unknown_args)
    model.eval()

    dataset = Dataset(test_file, vocab, cfg, evaluation=True)
    print("Evaluating with %s on %d pairs:" % (DEVICE.type.upper(), len(dataset)))

    with tqdm(total=min(len(dataset), cfg.limit), ncols=0, desc="Evaluating") as pbar:
        hypothesis, references = generate_summaries(model, dataset, cfg, pbar=pbar)

    scores = Rouge(use_python=args.use_python).get_scores(hypothesis, references)

    log_dict = make_log_dict(
        model_file, test_file, scores, stats, cfg, hypothesis, args.use_python
    )
    log_results(log_dict)

    if args.save:
        destination = Path(model_file).with_suffix(".json")
        save_summaries(destination, hypothesis, references, log_dict=log_dict)

    print_scores(scores)


if __name__ == "__main__":
    main()
