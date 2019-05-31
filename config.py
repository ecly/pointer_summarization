"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Store full Network + Inference configuration,
supporting processing of unknown args from argparse module
"""
import yaml


class Config:
    """Configuration and hyper-parameters of a Seq2Seq model"""

    limit: int = float("inf")  # limit dataset size
    epochs: int = 30  # maximum epochs
    iterations: int = float("inf")  # limit iterations (training batches)
    batch_size: int = 16

    train_file: str = "data/cnndm_abisee_train.tsv"
    valid_file: str = "data/cnndm_abisee_dev.tsv"
    vocab_file: str = "data/cnndm_abisee.vocab"
    output_file: str = "log/summarization.tar"
    save_every: int = 5000  # save checkpoint this often
    early_stopping: bool = False
    patience: int = float("inf")  # patience for early stopping
    validator: str = "rouge"  # rouge | loss
    validate_every: int = 0  # validate every X iterations (0 = every epoch)
    validation_size: int = 13368  # instances to sample for rouge validation

    # Data filtering.
    ## Instances outside these intervals are ignored if enabled.
    filter_instances: bool = False
    min_article_length: int = 50
    max_article_length: int = 4000
    min_summary_length: int = 5
    max_summary_length: int = 200
    min_compression_ratio: float = 2.0

    # Evaluation parameters (for beam search)
    beam_size: int = 4
    min_decode_steps: int = 35
    max_decode_steps: int = 120  # limit generated summaries to this many

    ## Length normalization during beam search
    length_normalize: str = "avg"  # avg | wu
    length_normalize_alpha: float = 2.2  # alpha for wu length penalty

    ## Coverage penalty during Beam Search from Wu et al. 2016
    coverage_penalty: bool = False
    coverage_penalty_beta: float = 5.0

    ## Block ngram repetition of size X - 0 is no blocking
    block_ngram_repeat: int = 0
    ## Block the model from producing <UNK> during beam search
    block_unknown: bool = False
    ## Prevent the model from repeating the same word twice in a row
    block_repeat: bool = False

    # Hyper-parameters
    ## Optimizer (fine tuning of specific optimizer not supported through config.py)
    optimizer: str = "adam"  # adam | adadelta | adam | adagrad | adamax | rmsprop
    adagrad_init_acc: float = 0.1  # adagrad specific (for See et al. 2017)
    learning_rate: float = 0.001
    learning_rate_decay: float = 1.0  # 1.0 is no decay, 0.5 is halving
    learning_rate_patience: int = 0  # how many receding validations before decay lr
    max_grad_norm: float = 2.0  # gradient clipping
    ## Normally we compute masked 2d loss, over an entire batch.
    ## This intuitively gives longer sequence a larger impact on the loss.
    ## With size_average=True, we first average the loss for each instance, prior
    ## to averaging for the batch, intuitively giving each instance equal say despite size.
    size_average: bool = False  # Size average loss for each instance in batch if True

    # Model
    rnn_cell: str = "gru"  # gru | lstm
    embed_file: str = None  # initialize with pretrained weights from w2v file
    vocab_size: int = 50000
    truncate_article: int = 400  # truncate articles during training
    truncate_summary: int = 100  # truncate summaries during training
    embed_size: int = 128
    hidden_size: int = 256
    attn_feature_size: int = 512  # 2xhidden_size in See et al. (2017)
    encoder_layers: int = 1
    decoder_layers: int = 1
    ##  Dropout configuration
    embedding_dropout: float = 0.0  # dropout for embeddings
    output_dropout: float = 0.0  # dropout before prediction layer
    encoder_dropout: float = 0.0  # only if encoder_layers > 1
    decoder_dropout: float = 0.0  # only if decoder_layers > 1

    ## attn_model:
    ## bahdanau | dot | scaled_dot | dot_coverage | general |
    ## general_coverage | bilinear_coverage | bilinear
    attn_model: str = "bahdanau"
    pointer: bool = True  # true if allow pointing, otherwise false
    coverage: bool = True  # penalize repeated attention, and make attn coverage aware
    coverage_loss_weight: float = 1.0  # reweigh coverage loss $(\lambda)$
    # reduce coverage loss weight every epoch with decreasing validation (scheduled $(\lambda)$
    coverage_loss_weight_decay: float = 1.0
    coverage_func: str = "sum"  # sum | max - note that we have only ever used sum
    # non-linear between last output layers (None | tanh | relu | sigmoid)
    output_activation: str = None

    ## Unknown management
    penalize_unknown: bool = False  # If `True`, UNK probabilities are added to total loss
    sample_when_unknown: bool = False  # If `True` we never feed UNK, and use prev. pred instead
    ignore_unknown_loss: bool = False  # If `True` don't let UNKs in target contribute to total loss

    def _set_param(self, param, value):
        assert param in self.__annotations__, "Unknown argument: {}".format(param)
        old = getattr(self, param)
        setattr(self, param, value)
        print("Hyper-parameter %s = %s (was %s)" % (param, value, old))

    def as_dict(self):
        """Utility to get a `Config` instance as dictionary"""
        cfg = {}
        for a in self.__annotations__:
            cfg[a] = getattr(self, a)

        return cfg

    def load(self, file):
        """
        Update the config from a yaml file with parameters
        :param file: A yaml file with corresponding attributes
        """
        print("Updating cfg from %s..." % file)
        with open(file, "r") as f:
            loaded = yaml.safe_load(f)
            for k, v in loaded.items():
                self._set_param(k, v)

    def update(self, args):
        """
        Update configuration by a list of command line arguments.
        Supports flags, and both (=| ) as key value delimiter.
        """
        # Handle arguments given in format key=value
        args = [i for arg in args for i in arg.split("=")]

        # Handle case of flag given as last argument
        if args and args[-1].startswith("--"):
            args.append("true")

        normalized = []
        ptr = 0
        while ptr < len(args):
            assert args[ptr].startswith("--")
            arg, nxt = args[ptr][2:], args[ptr + 1]
            assert hasattr(self, arg), "Unknown argument: {}".format(arg)
            typ = self.__annotations__[arg]
            if nxt.startswith("--"):
                assert typ == bool, "Only booleans can be given in flag format"
                normalized.append((arg, True))
                ptr += 1
                continue
            else:
                val = nxt.lower() == "true" if typ == bool else typ(nxt)
                normalized.append((arg, val))
                ptr += 2

        for k, v in normalized:
            self._set_param(k, v)
