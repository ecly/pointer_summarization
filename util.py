"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Contains utility functions across different modules.
Can be ran separately to inspect the values of a checkpoint.
Contains the CSV configuration, for producing result CSV-files.

Examples:
    python util.py log/cnndm.tar
"""
import os
import sys
import json
from statistics import mean, stdev
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from time import gmtime, strftime
from subprocess import check_output, CalledProcessError, STDOUT
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_HEADER_MAP = [
    # Identification of model
    ("Model File", "model_file"),
    ("Identifer", "model_identifier"),
    # Files
    ("Train File", "train_file"),
    ("Validation File", "valid_file"),
    ("Test File", "test_file"),
    ("Epoch", "epoch"),
    ("Iteration", "iteration"),
    # Misc.
    ("Time Training", "time_training"),
    ("Batch Size", "batch_size"),
    ("Train Limit", "limit"),
    ("Validator", "validator"),
    ("Validate Every", "validate_every"),
    ("Validation Size", "validation_size"),
    # Instance filtering
    ("Filter Instances", "filter_instances"),
    ("Truncate Article", "truncate_article"),
    ("Truncate Summary", "truncate_summary"),
    ("Min. Article Length", "min_article_length"),
    ("Max. Article Length", "max_article_length"),
    ("Min. Summary Length", "min_summary_length"),
    ("Max. Summary Length", "max_summary_length"),
    ("Min. Compression Ratio", "min_compression_ratio"),
    # Evaluation parameters
    ("Beam Size", "beam_size"),
    ("Min. Decode Steps", "min_decode_steps"),
    ("Max. Decode Steps", "max_decode_steps"),
    # Optimizer parameters
    {"Optimizer", "optimizer"},
    ("Learning Rate", "learning_rate"),
    ("Adagrad Initial Accumulator", "adagrad_init_acc"),
    ("Max. Gradient Norm", "max_grad_norm"),
    # Model parameters
    ("RNN Cell", "rnn_cell"),
    ("Embedding File", "embed_file"),
    ("Vocabulary Size", "vocab_size"),
    ("Embedding Size", "embed_size"),
    ("Hidden Size", "hidden_size"),
    ("Embedding Dropout", "embedding_dropout"),
    ("Encoder Dropout", "encoder_dropout"),
    ("Decoder Dropout", "decoder_dropout"),
    ("Encoder layers", "encoder_layers"),
    ("Decoder layers", "encoder_layers"),
    ("Pointer", "pointer"),
    ("Coverage", "coverage"),
    ("Coverage Loss Weight", "coverage_loss_weight"),
    ("Coverage Function", "coverage_func"),
    ("Attention Model", "attn_model"),
    # Results
    ("Evaluation Package", "eval_package"),
    ("ROUGE-1 (P)", "rouge-1-p"),
    ("ROUGE-1 (R)", "rouge-1-r"),
    ("ROUGE-1 (F1)", "rouge-1-f"),
    ("ROUGE-2 (P)", "rouge-2-p"),
    ("ROUGE-2 (R)", "rouge-2-r"),
    ("ROUGE-2 (F1)", "rouge-2-f"),
    ("ROUGE-L (P)", "rouge-l-p"),
    ("ROUGE-L (R)", "rouge-l-r"),
    ("ROUGE-L (F1)", "rouge-l-f"),
    # Misc / bonus stats
    ("Final Running Avg. Loss", "running_avg_loss"),
    ("Final Running Avg. Coverage Loss", "running_avg_cov_loss"),
    ("Best Validation Score", "best_validation_score"),
    ("Vocabulary File", "vocab_file"),
    ("Output Activation", "output_activation"),
    ("Penalize Unknown", "penalize_unknown"),
    ("Ignore Unknown Loss", "ignore_unknown_loss"),
    ("Sample When Unknown", "sample_when_unknown"),
    ("Output Activation", "output_activation"),
    ("Length Normalization", "length_normalize"),
    ("Length Normalization Alpha", "length_normalize_alpha"),
    ("Coverage Penalty", "coverage_penalty"),
    ("Coverage Penalty Beta", "coverage_penalty_beta"),
    ("Block N-gram Repeat", "block_ngram_repeat"),
    ("Learning Rate Decay", "learning_rate_decay"),
    ("Learning Rate Patience", "learning_rate_patience"),
    ("Average Summary Length", "avg_summary_length"),
    ("Summary Length stdev", "summary_length_stdev"),
    ("Size Average Loss", "size_average"),
    ("Block Unknown", "block_unknown"),
    ("Coverage Loss Weight Decay", "coverage_loss_weight_decay"),
]


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def get_model_identifier():
    """
    Procure an identifier for a model.
    We use the git hash for this if available. If not, use a timestamp.
    """
    try:
        return (
            check_output(["git", "describe", "--always"], stderr=STDOUT)
            .strip()
            .decode("utf-8")
        )
    except (FileNotFoundError, CalledProcessError):
        return strftime("%Y-%d-%m %H:%M", gmtime())


def count_parameters(model, learnable_only=True):
    """
    Count and return the number of parameters for a model.
    By default, only counts learnable parameters, however if
    `learnable_only=False` is given, will count all of the model's parameters.

    :param model: A pytorch module for which we count params.
    :param learnable_only: Only count learnable parameters (default=True)

    :returns: The total number of parameters for the given `model`
    """
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not learnable_only
    )


def flatten_scores(scores):
    """Flatten a ROUGE `get_scores` output to a single dict"""
    score_dict = {}
    for rouge in scores.keys():
        score_dict[rouge + "-p"] = scores[rouge]["p"] * 100
        score_dict[rouge + "-r"] = scores[rouge]["r"] * 100
        score_dict[rouge + "-f"] = scores[rouge]["f"] * 100

    return score_dict


# pylint: disable=too-many-arguments
def make_log_dict(model_file, test_file, scores, stats, cfg, hypothesis, use_python):
    """
    Create a log dict based on `CSV_HEADER_MAP`

    :param model_file: The path of the model file, for completion
    :param test_file: The path of the test file, for completion
    :param scores: The calculated rouge scores, as gotten from `Rouge().get_scores`
    :param stats: The stats dictionary saved by the trained for the model
    :param cfg: The `Config` instance that the model was trained with
    :param hypothesis: List of the generated hypothesis, used for stats
    :param use_python: `True` if evaluated with py-rouge otherwise pyrouge

    :returns: A log dict with combined stats, config and results
    """
    score_dict = flatten_scores(scores)
    config_dict = cfg.as_dict()
    merged = {**score_dict, **config_dict, **stats}
    merged["model_file"] = model_file
    merged["test_file"] = test_file
    merged["eval_package"] = "py-rouge" if use_python else "pyrouge"
    hyp_lens = list(map(lambda h: len(h.split()), hypothesis))
    merged["avg_summary_length"] = mean(hyp_lens)
    merged["summary_length_stdev"] = stdev(hyp_lens)

    return merged


def log_results(log_dict, dest=None):
    """
    Utility for logging results and config to a .CSV file.
    The results can then easily be imported to spreadsheet software for comparison

    :param log_dict: The log dict, used for logging as created by `make_log_dict`
    :param dest: Where to save, if not given, generates output from model_file in `log_dict`
    """
    destination = (
        log_dict["model_file"].split(".")[0] + "_results.csv" if dest is None else dest
    )

    header = ",".join([k for (k, _) in CSV_HEADER_MAP])
    values = ",".join([str(log_dict.get(v, "UNKNOWN")) for (_, v) in CSV_HEADER_MAP])

    with open(destination, "w") as f:
        print(header, file=f)
        print(values, file=f)

    print("Logged config and results to:", destination)


def save_summaries(file, hypothesis, references, log_dict=None):
    """
    Save summaries to a to a file as JSON.
    Saved as :
        {
            "log_dict": ...,
            "summaries": [
                {"reference": ..., "hypothesis": ...}
                ...
            ]
        }

    :param file: The path to save the summaries
    :param hypothesis: A corresponding list of hypothesis
    :param references: A list of reference summaries
    :param log_dict:
        An optional dictionary of results/config as made with
        `make_log_dict`. Will be stored under log_dict in the resulting JSON.
    """
    summaries = []
    for r, h in zip(references, hypothesis):
        summaries.append({"reference": r, "hypothesis": h})

    content = {"log_dict": log_dict, "summaries": summaries}

    print(f"Saving summaries to {file}...")
    with open(file, "w") as f:
        json.dump(content, f, indent=4)


def main():
    """Print config and stats for checkpoint to stdout"""
    assert len(sys.argv) == 2
    checkpoint = torch.load(sys.argv[1], map_location=DEVICE)

    cfg = checkpoint["config"]
    sys.stdout.write("Config:")
    for k, v in cfg.as_dict().items():
        print(f"\t{k}: {v}")

    stats = checkpoint["stats"]
    sys.stdout.write("\nStats:")
    for k, v in stats.items():
        print(f"\t{k}: {v}")

    # NOTE: for models with "frozen" parameters,
    # this value, will include these parameters.
    # See `count_parameters` for learnable only.
    state_dict = checkpoint["model_state_dict"]
    param_count = sum(p.numel() for p in state_dict.values())
    print(f"\nParams:\t{param_count}")


if __name__ == "__main__":
    main()
