"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Module with code for calculating validation scores during training,
which can be used to keep track of the currently best model on the validation set.

Can also be used to produce ROUGE scores on samples of some size from the given file of instances.
In this case, variance will be reported across the different metrics.

Examples:
    python validate.py log/cnndm.tar data/cnndm_dev.tsv 100
    python validate.py log/cnndm.tar data/cnndm_dev.tsv 500 50
    python validate.py log/cnndm.tar data/cnndm_dev.tsv 1000 10 --batch_size 16
"""
import time
import argparse
from statistics import mean, harmonic_mean, pvariance

import torch
from tqdm import tqdm

import train
import evaluate
from data import Dataset
from util import suppress_stdout_stderr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _valid_pbar(total, desc="Validating"):
    return tqdm(
        total=total,
        desc=desc,
        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed},{rate_fmt}]",
    )


def calc_validation_score(scores):
    """
    Calculate and return the validation score from a dict of scores
    The validation score is calculated as the three-way harmonic mean
    of the F1 scores for ROUGE-1, ROUGE-2, ROUGE-L.

    Formula: https://en.wikipedia.org/wiki/Harmonic_mean#Three_numbers

    :param scores: The dict of scores as output from py-rouge
    :returns: The calculated validation score
    """
    r1 = scores["rouge-1"]["f"] * 100
    r2 = scores["rouge-2"]["f"] * 100
    rl = scores["rouge-l"]["f"] * 100

    return harmonic_mean([r1, r2, rl])


def get_validation_score(model, dataset, cfg):
    """
    Compute a validation score for a model, which is represented
    as the sum of F1 scores for ROUGE-1, ROUGE-2 and ROUGE-L
    computed on `validation_size` summaries sampled from the given dataset.
    We use the py-rouge, pure Python reimplementation of ROUGE, for speed.
    Model is put into eval mode prior to validation, and back into
    training before returning.

    :param model: The `Seq2Seq` model used to generate the summaries
    :param dataset: The `Dataset` instance containing validation pairs
    :param cfg: The `Config` instance with parameters for evaluation

    :returns: The validation score as the harmonic mean of F1 scores
    """
    total = min(cfg.validation_size, len(dataset))
    with _valid_pbar(total) as pbar:
        scores = evaluate.evaluate(
            model,
            dataset,
            cfg,
            limit=cfg.validation_size,
            shuffle=True,
            pbar=pbar,
            use_python=True,
        )
        return calc_validation_score(scores)


def get_validation_loss(model, dataset, cfg):
    """
    Computes the average loss across the entirety of the given dataset.
    Assumes the given model is in eval mode.

    :param model: The `Seq2Seq` for which we calculate validation loss
    :param dataset: The `Dataset` instance containing validation pairs
    :param cfg: The `Config` instance with parameters for validation

    :returns: The average loss as a float.
    """
    with torch.no_grad(), _valid_pbar(len(dataset)) as pbar:
        generator = dataset.generator(cfg.batch_size, cfg.pointer, shuffle=True)
        losses = []
        cov_losses = []
        for _, batch in enumerate(generator):
            loss, cov_loss, _output = model(batch)
            losses.append(loss.item())
            cov_losses.append(cov_loss.item())
            pbar.update(len(batch))

        pbar.clear()
        val_loss = mean(losses)
        return val_loss


def prepare_arg_parser():
    """Create simple arg parser expecting 3 positional arguments"""
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
        help="path to .tsv-file with test pairs",
    )
    parser.add_argument(
        "set_size",
        metavar="set-size",
        type=int,
        help="size of a single randoly sampled validation set",
    )
    parser.add_argument(
        "-samples",
        "--samples",
        metavar="samples",
        type=int,
        default=50,
        help="how many samples to use to test variance",
    )

    return parser


def main():
    """
    Run evaluation for a model on a test file of given set size.
    The model will by default run 50 samples.

    Prints rouge variance of rouge scores and validation score to console,
    as well as the avg. set validation time.
    """
    args, unknown_args = prepare_arg_parser().parse_known_args()
    model_file = args.model_file
    test_file = args.test_file
    set_size = args.set_size
    samples = args.samples

    with suppress_stdout_stderr():
        model, _optimizer, vocab, _stats, cfg = train.load_model(
            model_file, unknown_args
        )

    model.eval()
    dataset = Dataset(test_file, vocab, cfg)
    r1_scores = []
    r2_scores = []
    rl_scores = []
    validation_scores = []
    times = []
    for i in range(samples):
        print("Step:", i)
        start = time.time()
        with suppress_stdout_stderr():
            scores = evaluate.evaluate(
                model, dataset, cfg, limit=set_size, shuffle=True
            )
        r1_scores.append(scores["rouge-1"]["f"] * 100)
        r2_scores.append(scores["rouge-2"]["f"] * 100)
        rl_scores.append(scores["rouge-l"]["f"] * 100)
        validation_scores.append(calc_validation_score(scores))
        times.append(time.time() - start)

    print(f"Test File: {test_file}, Set Size: {set_size}, Samples: {samples}")
    r1_v, r2_v, rl_v = pvariance(r1_scores), pvariance(r2_scores), pvariance(rl_scores)
    valid_v = pvariance(validation_scores)
    avg_time = mean(times)

    print("r1_variance,r2_variance,rl_variance,validation_variance,avg_time_for_set")
    print("%.2f,%.2f,%.2f,%.2f,%d" % (r1_v, r2_v, rl_v, valid_v, avg_time))


if __name__ == "__main__":
    main()
