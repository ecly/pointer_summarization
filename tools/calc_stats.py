"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Calculate basic metrics for training pair .TSV files of format:
<document>\t<summary>

Metrics are printed to stdout are saved to paths based on input
filename. Will be similar to:
    newsroom_train_doc_len_hist.png
    newsroom_train_sum_len_hist.png

Examples:
    python calc_stats.py ../data/newsroom_train.tsv -l 10000
    python calc_stats.py ../data/cnndm_dev.tsv -p
"""
import sys
import math
import argparse
from pathlib import Path
from statistics import stdev, mean

import numpy as np
from nltk.tokenize import RegexpTokenizer

from .plot import new_figure, save_figure

# Remove all punctuation etc.
TOKENIZER = RegexpTokenizer(r"\w+")


def edit_distance(source, target):
    """
    Iterative edit distance using DP
    Adapted from:
    https://www.python-course.eu/levenshtein_distance.php
    """
    SUB_COST = 1
    INS_COST = 1
    DEL_COST = 0

    rows = len(source) + 1
    cols = len(target) + 1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source transformed into empty strings by deletion:
    for i in range(1, rows):
        dist[i][0] = DEL_COST
    # target can be created from an empty string by inserting the chars
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            cost = 0 if source[row - 1] == target[col - 1] else SUB_COST
            dist[row][col] = min(
                dist[row - 1][col] + DEL_COST,  # deletion
                dist[row][col - 1] + INS_COST,  # insertion
                dist[row - 1][col - 1] + cost,  # substitution
            )

    return dist[rows - 1][cols - 1]


def lcs(a, b):
    """
    Calculate LCS between two strings. Modified from:
    https://rosettacode.org/wiki/Longest_common_subsequence#Python
    """
    # generate matrix of length of longest common subsequence for substrings of both words
    lengths = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    result = []
    j = len(b)
    for i in range(len(a) + 1):
        if lengths[i][j] != lengths[i - 1][j]:
            result.append(a[i - 1])

    return result


def normalize_string(string):
    """Lowercase and remove punctuation"""
    return TOKENIZER.tokenize(string.lower())


def is_outlier(points, thresh=3.5):
    """
    https://stackoverflow.com/a/11886564
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def plot_to_file(filename, lens):
    """
    Plot some lengths to file as a histogram.
    Y axis represents % of instance and X are bins of the givens lens.
    Outliers are filtered using `is_outlier` function.
    """
    lens = np.array(lens)
    lens = lens[~is_outlier(lens)]

    fig = new_figure()
    fig.suptitle(filename)
    plot = fig.add_subplot(1, 1, 1)
    plot.hist(lens, weights=np.zeros_like(lens) + 100.0 / lens.size, bins=14)
    plot.set_xlabel("words")
    plot.set_ylabel("instances")
    print("Saving plot to %s..." % filename)
    save_figure(fig, filename)


def calc_stats(generator, limit=math.inf):
    """Calculate stats and return them from the given generator"""
    doc_lens = []
    summ_lens = []
    dists = []
    novel = []
    lcs_ = []

    for idx, (document, summary) in enumerate(generator):
        if idx >= limit:
            break
        doc = normalize_string(document)
        summ = normalize_string(summary)
        # There are a few articles/summaries that are empty.
        if not doc or not summ:
            continue

        doc_lens.append(len(doc))
        summ_lens.append(len(summ))
        novel.append(len(set(summ)) / len(set(doc)))
        dists.append(edit_distance(doc, summ))
        lcs_.append(lcs(doc, summ))

    lcs_lens = list(map(len, lcs_))
    stats = {
        "document_lengths": doc_lens,
        "document_length": mean(doc_lens),
        "document_length_stddev": stdev(doc_lens),
        "summary_lengths": summ_lens,
        "summary_length": mean(summ_lens),
        "summary_length_stddev": stdev(summ_lens),
        "edit_distance": mean(dists),
        "novel_percentage": mean(novel) * 100,
        "lcs": mean(lcs_lens),
    }
    return stats


def pair_generator(filename):
    """Create a pair generator from the given filename. .tsv format"""
    with open(filename, "r") as file:
        for line in file:
            body, summary = line.split("\t")
            yield (body, summary)


def print_stats(stats, label=None):
    """Print stat to console"""
    if label is not None:
        print("Stats for", label)

    print("Document min length: %d" % min(stats["document_lengths"]))
    print("Document max length: %d" % max(stats["document_lengths"]))
    print("Document avg. length: %.3f" % stats["document_length"])
    print("Document length stdev: %.3f" % stats["document_length_stddev"])
    print("Summary min length: %d" % min(stats["summary_lengths"]))
    print("Summary max length: %d" % max(stats["summary_lengths"]))
    print("Summary avg. length: %.3f" % stats["summary_length"])
    print("Summary length stdev: %.3f" % stats["summary_length_stddev"])
    print("Edit distance avg.: %.3f" % stats["edit_distance"])
    print("Novel percentage avg.: %.3f%%" % stats["novel_percentage"])
    print("LCS avg.: %.3f" % stats["lcs"])


def make_histograms(stats, basename):
    """Save histograms to files prepending basename to the paths"""
    doc_len_hist_name = "%s-len-hist.png" % basename.replace("_", "-")
    sum_len_hist_name = "%s-sum-len-hist.png" % basename.replace("_", "-")
    plot_to_file(doc_len_hist_name, stats["document_lengths"])
    plot_to_file(sum_len_hist_name, stats["summary_lengths"])


def prepare_arg_parser():
    """Create simple arg parser expecting 1 positional args and 2 optional."""
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a test file and print its rouge scores"
    )
    parser.add_argument(
        "filename",
        metavar="file-path",
        type=str,
        help="path to .tsv-file with test pairs",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=sys.maxsize,
        help="limit amount of instances to read from test-path",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="create plots of histograms and save to files",
    )

    return parser


def main():
    """
    Get stats for given filename, optionally limiting amount
    of pairs to read. Args given as: <file> (<limit>)
    """
    args = prepare_arg_parser().parse_args()
    filename = args.filename
    limit = args.limit

    generator = pair_generator(filename)
    stats = calc_stats(generator, limit)
    print_stats(stats)

    basename = Path(filename).stem
    if args.plot:
        make_histograms(stats, basename)


if __name__ == "__main__":
    main()
