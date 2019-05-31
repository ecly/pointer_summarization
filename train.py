"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Module for loading/saving/training a summarization model.

Allows options (input, output, model) and hyper-parameters
defined in config module.

Examples:
    python train.py --train_file train.tsv --output_file log/exp1.tar
    python train.py --resume_from exp1.tar --iterations 100000
    python train.py --resume_from see_et_al.tar --convert_to_coverage --iterations 233000
"""
import time
import argparse
import os
import math
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import Adadelta, Adam, Adagrad, Adamax, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import Seq2Seq
from config import Config
from data import Dataset
from vocabulary import load_vocabulary
from util import get_model_identifier
import validate as valid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Trainer class to simplify training and saving a model"""

    def __init__(self, model, optimizer, vocab, cfg, stats=defaultdict(int)):
        """
        Create a trainer instance for the given model.
        This includes creating datasets based on the `input_file` and
        `valid_file` in the given `cfg`.

        :param model: The Seq2Seq model to train
        :param optimizer: The optimizer for the `model`
        :param vocab: A `Vocabulary` instance to be used
        :param cfg: The current Config from which we get epochs/batch_size/files etc.
        :param stats:
            A dict with values such as epoch/running_avg_loss etc. when resuming training
        """

        self.cfg = cfg
        self.dataset = Dataset(cfg.train_file, vocab, cfg)
        self.validation_dataset = Dataset(cfg.valid_file, vocab, cfg, evaluation=True)

        self.model = model
        self.optimizer = optimizer
        self.validator = cfg.validator.lower()
        # if cfg.validate_every == 0, we validate once every epoch
        self.validate_every = (
            math.ceil(len(self.dataset) / cfg.batch_size)  # batches/epoch
            if cfg.validate_every == 0
            else cfg.validate_every
        )
        self.rouge_valid = self.validator == "rouge"
        self.loss_valid = self.validator == "loss"
        self.scheduler = (
            None
            if cfg.learning_rate_decay >= 1.0
            else ReduceLROnPlateau(
                optimizer,
                mode="min" if self.validator == "loss" else "max",
                factor=cfg.learning_rate_decay,
                patience=cfg.learning_rate_patience,
                verbose=True,
            )
        )
        self.coverage_loss_weight_decay = cfg.coverage_loss_weight_decay
        assert self.validator in ["loss", "rouge"]
        self.early_stopping = cfg.early_stopping
        self.patience = cfg.patience

        self.epoch = stats["epoch"]
        self.iteration = stats["iteration"]
        self.running_avg_loss = stats["running_avg_loss"]
        self.running_avg_cov_loss = stats["running_avg_cov_loss"]
        self.best_validation_score = stats["best_validation_score"]
        self.current_validation_score = stats.get("current_validation_score", 0)
        self.current_patience = stats.get("current_patience", 0)
        self.model_identifier = stats["model_identifier"]
        self.time_training = stats["time_training"]

        # Updated and managed from train function and context
        self.training_start_time = None
        self.writer = None
        self.pbar = None

    def __enter__(self):
        # Create summary writer for tensorboard
        log_dir = os.path.splitext(self.cfg.output_file)[0] + "_log"
        print(f"Tensorboard logging directory: {log_dir}")
        self.writer = SummaryWriter(log_dir)

        print(f"Training on {DEVICE.type.upper()}")

        # Create pbar instance for clean progress tracking
        # The total will be whatever delimits first, epochs or iterations
        epoch_total = (
            math.ceil(len(self.dataset) / self.cfg.batch_size) * self.cfg.epochs
        )
        total = min(epoch_total, self.cfg.iterations)
        self.pbar = tqdm(
            total=total,
            initial=self.iteration,
            desc="Training",
            postfix={"loss": self.running_avg_loss, "cov": self.running_avg_cov_loss},
            bar_format="{desc}: {n_fmt}/{total_fmt}{postfix} [{elapsed},{rate_fmt}]",
            leave=True,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._update_time_training()
        self.save_model()
        self.writer.close()
        self.pbar.close()
        if DEVICE.type.upper() == "CUDA":
            print("Emptying cuda cache...")
            torch.cuda.empty_cache()

    def _update_progress(self, loss, cov_loss):
        """Updates progress, both in tensorboardX, and in the progress bar"""
        self.writer.add_scalar("train/loss", loss, self.iteration)
        self.writer.add_scalar("train/cov_loss", cov_loss, self.iteration)
        self.writer.add_scalar(
            "train/running_avg_loss", self.running_avg_loss, self.iteration
        )
        self.writer.add_scalar(
            "train/running_avg_cov_loss", self.running_avg_cov_loss, self.iteration
        )
        self.pbar.update()
        postfix = {
            "loss": round(self.running_avg_loss, 2),
            "cov": round(self.running_avg_cov_loss, 2),
        }
        self.pbar.set_postfix(postfix)

    def _update_running_avg_loss(self, loss, cov_loss=0, decay=0.99):
        """
        Updates the running avg losses
        :param loss: The new loss to update the running loss based on
        :param cov_loss: The new optional cov_loss to update the running loss based on
        :param decay: Optional decay value for calculating the running avg (default 0.99)
        """
        self.running_avg_loss = (
            loss
            if self.running_avg_loss == 0
            else self.running_avg_loss * decay + (1 - decay) * loss
        )
        self.running_avg_cov_loss = (
            cov_loss
            if self.running_avg_cov_loss == 0
            else self.running_avg_cov_loss * decay + (1 - decay) * cov_loss
        )

    def _update_time_training(self):
        """
        Update `self.time_training` based on `self.training_start_time`
        and on the current time. Resets `self.training_start_time` to the
        current time, to avoid including time intervals more than once in
        the total training time.
        """
        if self.training_start_time is not None:
            elapsed = time.time() - self.training_start_time
            self.time_training += elapsed
            # ensure we reset training_start_time since
            # this session has been added to time_training
            self.training_start_time = time.time()

    def _validation_improved(self, new, ref):
        """
        Check whether a new validation score was better than a reference,
        according to the currently configured validator

        :param new: The score we check whether or not it was an improvement
        :param ref: The reference score we compare against
        :returns: True if the `new` score was an improvement over `ref`
        """
        return new > ref if self.rouge_valid else new < ref or ref == 0

    def _validate(self):
        """
        Run validation of the model using the method defined in `self.validator`.
        If the model evaluates to a better score than `self.best_validation_score`,
        it is saved to `self.cfg.output_file` with a '_best' suffix.

        :returns: `True` if we should early stop, otherwise `False`.
        """

        self.model.eval()
        new = (
            valid.get_validation_score(self.model, self.validation_dataset, self.cfg)
            if self.validator == "rouge"
            else valid.get_validation_loss(
                self.model, self.validation_dataset, self.cfg
            )
        )
        self.model.train()

        self.writer.add_scalar("validation/score", new, self.iteration)
        old = self.current_validation_score
        best = self.best_validation_score
        self.current_validation_score = new
        if self._validation_improved(new, best):
            self.current_patience = 0
            self.pbar.write(
                f"Validation improved: {best:5.2f} -> {new:5.2f} (new best)"
            )
            self.best_validation_score = new
            self.save_model(suffix="_best")
        else:
            self.current_patience += 1
            s = "improved" if self._validation_improved(new, old) else "declined"
            self.pbar.write(
                "Validation {}: {:5.2f} -> {:5.2f} (P: {}/{})".format(
                    s, old, new, self.current_patience, self.patience
                )
            )

            # decay coverage loss weight
            if self.coverage_loss_weight_decay < 1:
                old_w = self.model.coverage_loss_weight
                new_w = old_w * self.coverage_loss_weight_decay
                self.pbar.write(
                    f"Decaying coverage loss weight: {old_w:.2f} -> {new_w:.2f}"
                )
                self.model.coverage_loss_weight = new_w

        if self.scheduler is not None:
            self.pbar.clear()
            self.scheduler.step(new)  # inform scheduler of new validation score

    def save_model(self, suffix=""):
        """
        Saves current model to `self.cfg.output_file`.

        :param suffix:
            Optional suffix to append to default save location.
            Used to save checkpoints for best model so far (suffix="best").
        """
        self._update_time_training()
        destination = suffix.join(os.path.splitext(self.cfg.output_file))

        model_state_dict = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "vocab": self.dataset.vocab,
                "config": self.cfg,
                "stats": {
                    "epoch": self.epoch,
                    "iteration": self.iteration,
                    "running_avg_loss": self.running_avg_loss,
                    "running_avg_cov_loss": self.running_avg_cov_loss,
                    "best_validation_score": self.best_validation_score,
                    "current_validation_score": self.current_validation_score,
                    "current_patience": self.current_patience,
                    "model_identifier": self.model_identifier,
                    "time_training": self.time_training,
                },
            },
            destination,
        )

    def train(self):
        """Start training"""
        self.training_start_time = time.time()
        for _epoch in range(self.epoch, self.epoch + self.cfg.epochs):
            generator = self.dataset.generator(self.cfg.batch_size, self.cfg.pointer)
            for _batch_idx, batch in enumerate(generator):
                self.iteration += 1
                loss, cov_loss = self.train_batch(batch)
                self._update_running_avg_loss(loss, cov_loss)
                self._update_progress(loss, cov_loss)

                if self.iteration % self.validate_every == 0:
                    self._validate()
                    if self.early_stopping and self.current_patience > self.patience:
                        self.pbar.write("Early stopping...")
                        return

                if self.iteration >= self.cfg.iterations:
                    return

                if self.iteration % self.cfg.save_every == 0:
                    self.save_model()

            self.epoch += 1

        self._update_time_training()

    def train_batch(self, batch):
        """
        Run a single training iteration.
        :param batch: the current `data.Batch` instance to process

        :returns: A tuple of the loss and part thereof that is coverage loss
        """
        self.optimizer.zero_grad()
        loss, cov_loss, _output = self.model(batch)
        loss.mean().backward()
        clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        return loss.mean().item(), cov_loss.mean().item()


def initialize_optimizer(params, cfg):
    """
    Create an optimizer for the given params based on the given cfg.

    :param params: The parameters of the model we optimize.
    :params cfg: The config from which we configure the optimizer.

    :returns: An optimizer for given `params` based on the `cfg`.
    """
    optimizer = cfg.optimizer.lower()
    assert optimizer in ["adam", "adadelta", "adamax", "rmsprop", "adagrad"]
    if optimizer == "adam":
        return Adam(params, lr=cfg.learning_rate)
    if optimizer == "adadelta":
        return Adadelta(params, lr=cfg.learning_rate)
    if optimizer == "adamax":
        return Adamax(params, lr=cfg.learning_rate)
    if optimizer == "rmsprop":
        return RMSprop(params, lr=cfg.learning_rate)
    if optimizer == "adagrad":
        return Adagrad(
            params, lr=cfg.learning_rate, initial_accumulator_value=cfg.adagrad_init_acc
        )


def convert_to_coverage(model, vocab, cfg):
    """
    Convert an existing model to a model with coverage enabled.
    This is done by creating a new model with coverage, and loading
    in the parameters from the given model.

    :param model: The model to be converted to coverage
    :param vocab: The vocabulary used for initializing the new model
    :param cfg:
        The cfg associated with model. This config will have
        coverage set to `True`.
    :returns: Returns a triple of (model, optimizier, cfg),
        where the returned model has coverage enabled. A new
        optimizer for the new model is also created, with its
        parameters and learning rate set per given `cfg`.
        The output_file of the config will be updated with
        a '_cov' suffix.
    """
    print("Converting model to coverage...")
    old_state = model.state_dict()
    cfg.coverage = True
    cfg.output_file = "_cov".join(os.path.splitext(cfg.output_file))
    new_model = Seq2Seq(vocab, cfg, initialize=False)

    if isinstance(model, nn.DataParallel):
        new_model = nn.DataParallel(new_model)

    new_state = new_model.state_dict()

    for k in filter(lambda k: k in old_state, new_state.keys()):
        new_state[k] = old_state[k]

    new_model.load_state_dict(new_state)
    new_model.to(DEVICE)

    optimizer = initialize_optimizer(new_model.parameters(), cfg)
    print("New output_file is", cfg.output_file)
    return new_model, optimizer, cfg


def load_model(model_path, extra_args=None):
    """
    Load and return a model from the given `model_path`.

    :param model_path: The path of the checkpoint to load from
    :param extra_args: Optional extra args to be loaded into the cfg
    :returns: (model, optimizer, vocab, stats, cfg)
    """
    print("Loading model from %s..." % model_path)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    cfg = checkpoint["config"]
    # If we renamed the model file, we use the new name
    setattr(cfg, "output_file", model_path)

    if extra_args is not None:
        cfg.update(extra_args)

    vocab = checkpoint["vocab"]
    stats = checkpoint["stats"]

    # Warn if loading model with different git revision from current
    current_commit = get_model_identifier()
    identifier = stats["model_identifier"]
    if identifier != current_commit:
        print(
            f"[WARN] Loaded model identifier is {identifier} (current is {current_commit})"
        )

    model = Seq2Seq(vocab, cfg, initialize=False)
    state_dict = checkpoint["model_state_dict"]

    # Compatibility with older models, saved as DataParallel.
    # For these we just remove leading 'module.' from the keys.
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if torch.cuda.device_count() > 1:
        print(
            "%d GPUs available - wrapping in DataParallel" % torch.cuda.device_count()
        )
        model = nn.DataParallel(model)

    model.to(DEVICE)

    optimizer = initialize_optimizer(model.parameters(), cfg)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, vocab, stats, cfg


def initialize_model(cfg):
    """
    Initialize a new model using the given config.

    :param cfg: the `Config` instance used to create the model
    :returns: (model, optimizer, vocab, stats, cfg)
    """

    print("Initializing a new model...")
    vocab = load_vocabulary(cfg.vocab_file, cfg.vocab_size)
    model = Seq2Seq(vocab, cfg)

    if torch.cuda.device_count() > 1:
        print(
            "%d GPUs available - wrapping in DataParallel" % torch.cuda.device_count()
        )
        model = nn.DataParallel(model)

    model.to(DEVICE)

    optimizer = initialize_optimizer(model.parameters(), cfg)
    stats = defaultdict(int)
    stats["model_identifier"] = get_model_identifier()
    return model, optimizer, vocab, stats, cfg


def prepare_arg_parser():
    """Create arg parser handling input/output and training conditions"""
    arg_parser = argparse.ArgumentParser(
        description="Trains an existing or new summarization model. "
        + "If no args are given, a new model will be created and saved "
        + "to the default or given save path."
    )
    arg_parser.add_argument(
        "-r",
        "--resume_from",
        metavar="resume-from-path",
        default=argparse.SUPPRESS,
        help="loads the given existing model for training if it exists",
    )
    arg_parser.add_argument(
        "-cfg",
        "--config",
        metavar="config-file-path",
        default=argparse.SUPPRESS,
        help="load config params from yaml file - only usable for new models",
    )
    arg_parser.add_argument(
        "-c",
        "--convert_to_coverage",
        action="store_true",
        help="convert the model to coverage - should be used with --resume_from",
    )
    arg_parser.add_argument(
        "-e",
        "--eval",
        nargs="?",
        metavar="test-file",
        const="data/cnndm_abisee_test.tsv",
        default=argparse.SUPPRESS,
        help=".tsv-file with test pairs to run evaluation on when training finishes",
    )

    return arg_parser


def main():
    """Build dataset according to args and train model"""
    args, unknown_args = prepare_arg_parser().parse_known_args()

    if "resume_from" in args:
        model, optimizer, vocab, stats, cfg = load_model(args.resume_from, unknown_args)
    else:
        cfg = Config()
        if "config" in args:
            cfg.load(args.config)
        if unknown_args:
            cfg.update(unknown_args)

        model, optimizer, vocab, stats, cfg = initialize_model(cfg)

    if args.convert_to_coverage:
        assert not cfg.coverage, "Given model already has coverage enabled!"
        model, optimizer, cfg = convert_to_coverage(model, vocab, cfg)

    with Trainer(model, optimizer, vocab, cfg, stats) as trainer:
        trainer.train()

    if "eval" in args:
        # if there is a '_best' model we use that
        test_model = "_best".join(os.path.splitext(cfg.output_file))
        test_model = test_model if os.path.isfile(test_model) else cfg.output_file
        os.system(f"python evaluate.py {test_model} {args.eval} -s")


if __name__ == "__main__":
    main()
