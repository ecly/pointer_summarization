"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Tool for calcing stats from model output.
Supports both the format from See et al. (2017)
where a folder is used, and a JSON file which we use.
"""
import sys
import json
import glob
import os
from statistics import mean, stdev


def lens_folder(folder):
    """Returns amount of words in all text files in given folder as a list"""
    lens = []
    for file in glob.glob(os.path.join(folder, "*.txt")):
        with open(file, "r") as f:
            tokens = f.read().split()
            lens.append(len(tokens))

    return lens


def lens_json(filename):
    """Returns amount of words in hypothesis summaries in given JSON-file as a list"""
    lens = []
    with open(filename, "r") as f:
        print("Loading JSON...")
        data = json.load(f)
        summaries = data["summaries"]
        hyps = map(lambda s: s["hypothesis"], summaries)
        for h in hyps:
            lens.append(len(h.split()))

    return lens


def main():
    """Print length stats to stdout for given file/folder"""
    target = sys.argv[1]
    lens = lens_json(target) if target.endswith(".json") else lens_folder(target)
    print("max:", max(lens))
    print("min:", min(lens))
    print("mean:", mean(lens))
    print("stdev:", stdev(lens))


if __name__ == "__main__":
    main()
