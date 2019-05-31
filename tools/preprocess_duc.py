"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Script producing incredibly ineffecient output of DUC
to get easy compatibility with calc_stats.py.

Since DUC is multi-document (10) summarization, with 4 different reference summaries,
we output tab separated instances with each document concatenated.
Each of the 4 reference summaries will all have a new copy of the 10 documents,
meaning we replicate all 10 articles 4 times.

Since DUC is such a small dataset, we currently don't care about this, and just want
compatibility. Yuck, yuck.

Example:
    python preprocess_duc.py data/duc data/duc.tsv
"""
import os
import re
import argparse
from glob import glob
from pathlib import Path

ART_REGEX = re.compile("<TEXT>(.*?)</TEXT>", re.S)


def parse_article(text):
    """
    Retrieve the body of a DUC article instance.
    We also replace newlines with spaces to TSV compatibility.
    """
    return ART_REGEX.search(text).group(1).replace("\n", " ")


def get_instances(duc_folder):
    """
    Given `duc_folder` of structore:
    - docs
    --  d30001t
    --- article1
    --- ...
    -- ...
    - sums
    -- sum_d30001t_A
    -- sum_d30001t_B
    -- ...

    We produce list of a tuples with instances (list_of_articles, list_of_summaries).
    """
    art_folders = glob(os.path.join(duc_folder, "docs/*"))
    sum_folder = os.path.join(duc_folder, "sums")
    instances = []
    for art_folder in art_folders:
        art_files = glob(os.path.join(art_folder, "*"))
        articles = []
        for file in art_files:
            with open(file, "r") as f:
                articles.append(parse_article(f.read()))

        name = Path(art_folder).stem
        sum_files = glob(os.path.join(sum_folder, "sum_%s*" % name))
        summaries = []
        for file in sum_files:
            with open(file, "r") as f:
                summaries.append(f.read().replace("\n", " "))

        instances.append((articles, summaries))

    return instances


def prepare_arg_parser():
    """Create simple arg parser expecting 2 positional arguments"""
    parser = argparse.ArgumentParser(
        description="Preprocess DUC dataset into a .tsv-file."
    )
    parser.add_argument(
        "duc_folder",
        metavar="duc-folder-path",
        type=str,
        help="path to outmost duc folder",
    )
    parser.add_argument(
        "output_file",
        metavar="output-path",
        type=str,
        help="path of resulting .tsv-file",
    )
    return parser


def main():
    args = prepare_arg_parser().parse_args()
    instances = get_instances(args.duc_folder)
    with open(args.output_file, "w") as f:
        for articles, summaries in instances:
            concat_articles = " ".join(articles)
            for summary in summaries:
                print("%s\t%s" % (concat_articles, summary), file=f)

    extra = "_individual_articles.".join(args.output_file.split("."))
    with open(extra, "w") as f:
        for articles, _ in instances:
            for a in articles:
                print("%s\t<SUM_PLACEHOLDER>" % a, file=f)


if __name__ == "__main__":
    main()
