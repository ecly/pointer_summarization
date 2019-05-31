"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Implementation of Pointer Generator with `EncoderRNN`, `DecoderRNN` and `Attention` model.
Also has ReduceState module, for creating an initial decoder state, from the last encoder state.
Provides a `Seq2Seq` wrapper, taking care of running an entire
execution step by step, as well as loss calculation.

Network supports a wide variety of configurations, which can be made using the `config` module.
"""
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from gensim.models import KeyedVectors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-31
SEED = 123

# For replication of results, we manually seed
random.seed(SEED)
torch.manual_seed(SEED)


class EncoderRNN(nn.Module):
    """Encoder RNN computing hidden representations from word embeddings"""

    def __init__(self, cfg):
        super(EncoderRNN, self).__init__()
        self.layers = cfg.encoder_layers
        self.hidden_size = cfg.hidden_size
        self.rnn_cell = cfg.rnn_cell.lower()
        self.rnn = (
            nn.LSTM(
                cfg.embed_size,
                cfg.hidden_size,
                cfg.encoder_layers,
                bidirectional=True,
                batch_first=True,
                dropout=cfg.encoder_dropout,
            )
            if self.rnn_cell == "lstm"
            else nn.GRU(
                cfg.embed_size,
                cfg.hidden_size,
                cfg.encoder_layers,
                bidirectional=True,
                batch_first=True,
                dropout=cfg.encoder_dropout,
            )
        )

    def forward(self, embedded, input_lengths):
        """
        :param embedded: Full embedded input (batch_size, token_count, embed_size)
        :param input_lengths:
            tensor of batch_size with non-padded length of each instance in embedded
        :returns:
            A tuple of (outputs, last_layer_state)
            where outputs: (batch_size, token_count, hidden_size * 2)
            and last_layer_state: (2, batch_size, hidden_size) if GRU
            and a tuple of size 2 thereof if LSTM
        """
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, state = self.rnn(packed)
        outputs, _output_lengths = pad_packed_sequence(outputs, batch_first=True)

        batch_size = embedded.size(0)
        if self.rnn_cell == "lstm":
            hidden, cell_state = state

            # separate layers so we can take last for multi-layered encoder
            hidden_states = hidden.view(self.layers, 2, batch_size, self.hidden_size)
            cell_states = cell_state.view(self.layers, 2, batch_size, self.hidden_size)

            return outputs, (hidden_states[-1], cell_states[-1])

        # separate layers so we can take last for multi-layered encoder
        states = state.view(self.layers, 2, batch_size, self.hidden_size)
        return outputs, states[-1]


class ReduceState(nn.Module):
    """
    Module for reducing Encoder's bidrectional RNN hidden state for the decoder
    Also adjusts state for the number of layers in the decoder.
    """

    def __init__(self, cfg):
        super(ReduceState, self).__init__()
        self.hidden_size = cfg.hidden_size
        self.rnn_cell = cfg.rnn_cell.lower()
        self.decoder_layers = cfg.decoder_layers
        self.reduce_hidden = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size * cfg.decoder_layers),
            nn.ReLU(),
        )
        if self.rnn_cell == "lstm":
            self.reduce_cell = nn.Sequential(
                nn.Linear(cfg.hidden_size * 2, cfg.hidden_size * cfg.decoder_layers),
                nn.ReLU(),
            )

    def forward(self, state):
        """
        Reduce `state` from EncoderRNN depending on what rnn_cell is used (GRU or LSTM).
        Also adjusts the given `state` to the number of layers in the decoder.

        :param state:
            The hidden state from the `EncoderRNN`
            GRU: (2, batch_size, hidden_size)
            LSTM: Tuple of ((2, batch_size, hidden_size), (2, batch_size, hidden_size))

        :returns:
            A reduced hidden state of size (1, batch_size, hidden_size) if GRU
            else ((1, batch_size, hidden_size), (1, batch_size, hidden_size))
        """

        if self.rnn_cell == "lstm":
            hidden, cell = state

            hidden = hidden.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2)
            hidden = self.reduce_hidden(hidden).unsqueeze(0)
            hidden = hidden.view(self.decoder_layers, -1, self.hidden_size)

            cell = cell.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2)
            cell = self.reduce_cell(cell).unsqueeze(0)
            cell = cell.view(self.decoder_layers, -1, self.hidden_size)
            return (hidden, cell)

        hidden = state.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2)
        hidden = self.reduce_hidden(hidden).unsqueeze(0)
        hidden = hidden.view(self.decoder_layers, -1, self.hidden_size)
        return hidden


class Attention(nn.Module):
    """Attention module supporting various attention methods"""

    def __init__(self, cfg):
        super(Attention, self).__init__()

        self.method = cfg.attn_model.lower()
        self.hidden_size = cfg.hidden_size
        self.state_size = (
            cfg.hidden_size * 2 if cfg.rnn_cell.lower() == "lstm" else cfg.hidden_size
        )
        self.feature_size = cfg.attn_feature_size
        self.coverage = cfg.coverage
        self.coverage_func = cfg.coverage_func.lower()

        if self.method.startswith("dot"):
            assert cfg.rnn_cell.lower() == "lstm", "dot currently only works with lstm"
            self._attention_function = self._dot
            if self.method.endswith("_coverage"):
                self.coverage_projection = nn.Linear(1, 1)
                self._attention_function = self._dot_coverage
        elif self.method == "scaled_dot":
            assert cfg.rnn_cell.lower() == "lstm", "dot currently only works with lstm"
            self._attention_function = self._scaled_dot
            self.scale = math.sqrt(cfg.hidden_size * 2)
        elif self.method.startswith("general"):
            self._attention_function = self._general
            self.hidden_projection = nn.Linear(
                self.state_size, self.hidden_size * 2, bias=False
            )
            if self.method.endswith("_coverage"):
                self.coverage_projection = nn.Linear(1, 1)
                self._attention_function = self._general_coverage

        elif self.method == "bilinear":
            self._attention_function = self._bilinear
            self.attn_bilinear = nn.Bilinear(self.state_size, self.hidden_size * 2, 1)
            if cfg.coverage:
                self.coverage_projection = nn.Linear(1, 1)
                self.coverage_bilinear = nn.Bilinear(1, 1, 1)

        elif self.method == "bilinear_coverage":
            assert self.coverage, "bilinear_coverage only applicable with coverage=True"
            self._attention_function = self._bilinear_coverage
            self.hidden_projection = nn.Linear(
                self.state_size, self.feature_size, bias=False
            )
            self.coverage_projection = nn.Linear(1, 1, bias=False)  # W_c

            self.v = nn.Bilinear(self.feature_size, 1, 1)
        elif self.method == "bahdanau":
            self._attention_function = self._bahdanau
            # Note, that we remove bias from everything but v,
            # as Eq. 11 from See et al. (2017) only has bias (b_attn)
            # The encoder features are given as arguments, to ensure
            # they are only computes once for every instance
            self.v = nn.Linear(self.feature_size, 1)
            self.hidden_projection = nn.Linear(
                self.state_size, self.feature_size, bias=False
            )  # W_s
            if cfg.coverage:
                self.coverage_projection = nn.Linear(
                    1, self.feature_size, bias=False
                )  # W_c

        else:
            raise ValueError("Attention not implemented for %s" % self.method)

    def _get_coverage(self, coverage, attn_weights):
        """
        Combines the past attention weights into one vector

        :param coverage: The existing coverage (batch_size, token_count)
        :param attn_weights: List of past attention (batch_size, token_count)

        :returns: A single tensor representing coverage (batch_size, token_count)
        """
        if self.coverage_func == "sum":
            return coverage + attn_weights
        if self.coverage_func == "max":
            stacked = torch.stack((coverage, attn_weights), dim=2)
            new_coverage, _ = torch.max(stacked, dim=2)
            return new_coverage

        raise ValueError("Coverage function %s not supported" % self.coverage_func)

    def _bahdanau(self, hidden, _encoder_outputs, **kwargs):
        """Bahdanau et al. (2015) + coverage from See et al. (2017)"""
        encoder_features = kwargs["encoder_features"]
        hidden_features = self.hidden_projection(hidden)
        hidden_features = (
            hidden_features.unsqueeze(1).expand_as(encoder_features).contiguous()
        )
        attn_features = hidden_features + encoder_features
        if self.coverage:
            coverage = kwargs["coverage"].unsqueeze(2)
            coverage_features = self.coverage_projection(coverage)
            attn_features = attn_features + coverage_features

        # v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn) if coverage
        # v^T tanh(W_h h_i + W_s s_t + b_attn) if not coverage
        # (batch_size, token_count, 1)
        energies = self.v(torch.tanh(attn_features))
        return energies.squeeze(2)

    # pylint: disable=no-self-use
    def _dot(self, hidden, encoder_outputs, **_kwargs):
        """Dot attention described in Luong et al. (2015)"""
        energies = torch.bmm(hidden.unsqueeze(1), encoder_outputs.transpose(1, 2))
        return energies.squeeze(1)

    def _dot_coverage(self, hidden, encoder_outputs, **kwargs):
        """Dot attention biased on weighted representation of coverage"""
        energies = self._dot(hidden, encoder_outputs, **kwargs)

        if self.coverage:
            coverage = kwargs["coverage"].unsqueeze(2)
            coverage_feature = self.coverage_projection(coverage)
            energies = torch.add(energies, coverage_feature.squeeze(2))

        return energies

    def _scaled_dot(self, hidden, encoder_outputs, **kwargs):
        """Scaled dot from Vaswani (2017)"""
        energies = self._dot(hidden, encoder_outputs, **kwargs)
        return energies.mul(self.scale)

    def _general(self, hidden, encoder_outputs, **kwargs):
        """General attention described in Luong et al. (2015)"""
        hidden_features = self.hidden_projection(hidden)
        return self._dot(hidden_features, encoder_outputs, **kwargs)

    def _general_coverage(self, hidden, encoder_outputs, **kwargs):
        """General attention biased on ewighted representation of coverage"""
        hidden_features = self.hidden_projection(hidden)
        return self._dot_coverage(hidden_features, encoder_outputs, **kwargs)

    def _bilinear_coverage(self, hidden, _encoder_outputs, **kwargs):
        """Similar to Bahdanau et al. (2015) except coverage features applied bilinearly"""
        encoder_features = kwargs["encoder_features"]
        hidden_features = self.hidden_projection(hidden)
        hidden_features = (
            hidden_features.unsqueeze(1).expand_as(encoder_features).contiguous()
        )
        attn_features = hidden_features + encoder_features
        coverage_features = self.coverage_projection(kwargs["coverage"].unsqueeze(2))
        energies = self.v(torch.tanh(attn_features), torch.tanh(coverage_features))
        return energies.squeeze(2)

    def _bilinear(self, hidden, encoder_outputs, **kwargs):
        """Bilinear attention - note that this is very slow and requires lots (V)RAM"""
        batch_size, token_count, _ = encoder_outputs.shape
        hidden_features = (
            hidden.unsqueeze(1).expand(batch_size, token_count, -1).contiguous()
        )
        attn = self.attn_bilinear(hidden_features, encoder_outputs.contiguous())

        if self.coverage:
            coverage = kwargs["coverage"].unsqueeze(2)
            coverage_features = self.coverage_projection(coverage)
            attn = self.coverage_bilinear(attn, coverage_features)

        return attn.squeeze(2)

    def forward(self, hidden, encoder_outputs, encoder_pad_mask, **kwargs):
        """
        Compute attention using configured model.
        Computed for given decoder `hidden` on given `encoder_outputs`.

        :param hidden: (batch_size, state_size)
        :param encoder_outputs: (batch_size, token_count, hidden_size)
        :param encoder_pad_mask: Paddings to normalize attention (batch_size, token_count)
        :param **kwargs:
            coverage: coverage tensor if enabled (batch_size, token_count)
            encoder_features: features for bahdanau (batch_size, token_count, hidden_size * 2)

        :returns: A tuple consisting of
            1. Attention energies (batch_size, token_count)
            2. New coverage_vector (batch_size, token_count) if coverage, otherwise None
        """
        attn = self._attention_function(hidden, encoder_outputs, **kwargs)
        attn = F.softmax(attn, dim=1)
        attn = attn * encoder_pad_mask
        normalization_factor = torch.sum(attn, dim=1, keepdim=True)  # (batch_size, 1)
        attn = attn / normalization_factor
        new_coverage = (
            self._get_coverage(kwargs["coverage"], attn) if self.coverage else None
        )
        return attn, new_coverage


class AttnDecoderRNN(nn.Module):
    """
    Decoder RNN computing word probabilities given encoder outputs
    and previous decoder hidden state. Uses attention mechanism from `Attention` module.
    """

    def __init__(self, cfg):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = cfg.attn_model
        self.embed_size = cfg.embed_size
        self.output_size = cfg.vocab_size
        self.pointer = cfg.pointer
        self.hidden_size = cfg.hidden_size
        self.rnn_cell = cfg.rnn_cell.lower()
        # the joint state we compute for LSTM is both cell_state and hidden making it x2
        self.state_size = (
            cfg.hidden_size * 2 if self.rnn_cell == "lstm" else cfg.hidden_size
        )
        self.coverage = cfg.coverage

        # Define layers
        self.attn = Attention(cfg)
        self.rnn = (
            nn.LSTM(
                cfg.embed_size,
                cfg.hidden_size,
                cfg.decoder_layers,
                batch_first=True,
                dropout=cfg.decoder_dropout,
            )
            if self.rnn_cell == "lstm"
            else nn.GRU(
                cfg.embed_size,
                cfg.hidden_size,
                cfg.decoder_layers,
                batch_first=True,
                dropout=cfg.decoder_dropout,
            )
        )

        self.context_projection = nn.Linear(
            cfg.hidden_size * 2 + cfg.embed_size, cfg.embed_size
        )

        self.out = nn.Sequential(
            nn.Dropout(p=cfg.output_dropout),
            nn.Linear(cfg.hidden_size * 3, cfg.hidden_size),
            self._get_activation(cfg.output_activation),
            nn.Linear(cfg.hidden_size, cfg.vocab_size),
            nn.Softmax(dim=1),
        )
        if self.pointer:
            self.ptr = nn.Sequential(
                nn.Linear(cfg.hidden_size * 2 + self.state_size + cfg.embed_size, 1),
                nn.Sigmoid(),
            )

    @staticmethod
    def _get_activation(activation):
        if activation is None:
            return nn.Sequential()  # identity
        if activation.lower() == "relu":
            return nn.ReLU()
        if activation.lower() == "tanh":
            return nn.Tanh()
        if activation.lower() == "sigmoid":
            return nn.Sigmoid()

        raise ValueError("Activation (%s) not supported" % activation)

    def forward(
        self,
        embedded,
        last_state,
        last_context,
        encoder_outputs,
        encoder_pad_mask,
        encoder_word_ids,
        ext_vocab_size=None,
        coverage=None,
        encoder_features=None,
    ):
        """
        :param embedded: (batch_size, token_count, embed_size)
        :param last_state:
            Will be (1, batch_size, hidden_size) if rnn_cell is GRU
            else a tuple of size 2 thereof, if rnn_cell is LSTM
        :param last_context: (batch_size, hidden_size * 2)
        :param encoder_outputs: (batch_size, token_count, hidden_size * 2)
        :param encoder_pad_mask: pad mask for attention (batch_size, token_count)
        :param encoder_word_ids: (batch_size, token_count), with extended vocab ids
        :param ext_vocab_size: The size of the `ExtendedVocabulary` for the current batch
        :param coverage: (batch_size, token_count)
        :param encoder_features:
            encoder attention features, used for some attn_models
            (batch_size, token_count, hidden_size * 2)

        :returns: A tuple containing:
            1. Word probablities of size (batch_size, vocab_size) if not `self.pointer`
               otherwise of size (batch_size, ext_vocab_size)
            2. New hidden state of the decoder
            3. Attention weights (batch_size, token_count)
            4. New coverage (batch_size, token_count) if `self.coverage` otherwise None
        """
        # Combine context and embedded (batch_size, embed_size)
        input_ = self.context_projection(torch.cat((embedded, last_context), dim=1))

        # Get current hidden state from input word and last hidden state
        # (1, batch_size, hidden_size), (1, batch_size, hidden_size)
        rnn_output, rnn_state = self.rnn(input_.unsqueeze(1), last_state)

        # `hidden_hat`, is the state of the layer last layer of the RNN.
        # For LSTMs we concatenate the hidden state and the cell state
        # For GRUs, the hidden_hat just the last layer's hidden state
        if self.rnn_cell == "lstm":
            last_layer_hidden = rnn_state[0][-1, :, :]
            last_layer_cell = rnn_state[1][-1, :, :]
            hidden_hat = torch.cat((last_layer_hidden, last_layer_cell), dim=1)
        else:
            hidden_hat = rnn_state[-1, :, :]

        # ((batch_size, token_count), (batch_size, token_count))
        attn_weights, new_coverage = self.attn(
            hidden_hat,
            encoder_outputs,
            encoder_pad_mask,
            coverage=coverage,
            encoder_features=encoder_features,
        )

        # Adjust for batched mm: (batch_size, 1, token_count)
        attn_weights_ = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights_, encoder_outputs)

        # Prepare for and create [s_t, h^*_t]
        context = context.squeeze(1)
        rnn_output = rnn_output.squeeze(1)
        # (batch_size, hidden_size * 3)
        concat_input = torch.cat((rnn_output, context), dim=1)
        # V'(V([s_t, h^*_t] + b) + b')
        output = self.out(concat_input)
        if self.pointer:
            # (batch_size, ext_vocab_size)
            ext_output = torch.zeros(output.size(0), ext_vocab_size, device=DEVICE)
            # (batch_size, 3 * hidden_size + embed_size)
            p_gen_input = torch.cat((context, hidden_hat, input_), dim=1)
            # (batch size, 1)
            p_gen = self.ptr(p_gen_input)
            p_ptr = 1 - p_gen

            # add generator probabilities to output
            ext_output[:, : self.output_size] = p_gen * output

            # (batch_size, token_count)
            copy_attn = p_ptr * attn_weights
            ext_output.scatter_add_(1, encoder_word_ids, copy_attn)
            output = ext_output

        return output, rnn_state, context, attn_weights, new_coverage


class Seq2Seq(nn.Module):
    """
    Wrapper for Encoder/Decoder with Attention to simplify iterating
    through timesteps during training. Also provides beam search implementation.
    """

    def __init__(self, vocab, cfg, initialize=True):
        """
        Create a complete Seq2Seq model, with Encoder/Decoder automatically created.
        Underlying modules will be configured based on the given `cfg`.

        :param vocab: Vocabulary for vocab_size and padding id
        :param cfg: `Config` instance for model hyper-parameters
        :param initialize:
            If False, we expect weights to be loaded in manually,
            and therefore will load no pretrained embeddings, despite configuration.

        """
        super(Seq2Seq, self).__init__()
        self.rnn_cell = cfg.rnn_cell.lower()
        self.hidden_size = cfg.hidden_size
        self.penalize_unknown = cfg.penalize_unknown
        self.sample_when_unknown = cfg.sample_when_unknown
        self.ignore_unknown_loss = cfg.ignore_unknown_loss
        self.vocab = vocab
        self.vocab_size = cfg.vocab_size
        self.coverage = cfg.coverage
        self.coverage_loss_weight = cfg.coverage_loss_weight
        self.size_average = cfg.size_average
        self.criterion = (
            nn.NLLLoss(ignore_index=vocab.PAD, reduction="none")
            if self.size_average
            else nn.NLLLoss(ignore_index=vocab.PAD, reduction="mean")
        )

        if initialize and cfg.embed_file is not None:
            self.embedding = self.load_pretrained_embeddings(cfg.embed_file, vocab)
            # override whatever value embed_size was set to with actual size
            cfg.embed_size = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(
                cfg.vocab_size, cfg.embed_size, padding_idx=vocab.PAD
            )

        self.embedding = nn.Sequential(
            self.embedding, nn.Dropout(p=cfg.embedding_dropout)
        )
        self.encoder = EncoderRNN(cfg)

        # To avoid producing encoder_features for the attnetion multiple times in
        # the attention module, which is called every time step, we compute it here instead.
        self.encoder_projection = (
            nn.Linear(cfg.hidden_size * 2, cfg.attn_feature_size, bias=False)  # W_h
            if cfg.attn_model.lower() in ["bahdanau", "bilinear_coverage"]
            else nn.Sequential()
        )
        self.reduce_state = ReduceState(cfg)
        self.decoder = AttnDecoderRNN(cfg)

    @staticmethod
    def load_pretrained_embeddings(embed_file, vocab):
        """
        Create an embedding layer using the given w2v file and `vocab`.
        Words from vocab with no embedding from w2v will be randomly initialized.
        The embedding layer returned will NOT be frozen.
        Padding embedding is a zero vector.

        :param embed_file:
            Path to non-binary word2vec file
            See tools.glove2word2vec module for creating embed_file
        :param vocab: `Vocab` instance to get indices and num_embeddings from

        :returns: An instance of `nn.Embedding` with preloaded weights for known words
        """

        print("Loading embeddings from %s..." % embed_file)
        w2v = KeyedVectors.load_word2vec_format(embed_file, binary=False)
        embed_size = w2v.vector_size
        vocab_size = len(vocab)
        weights = torch.randn(vocab_size, embed_size)
        found = 0
        for word, idx in vocab.w2i.items():
            if word in w2v:
                weights[idx] = torch.from_numpy(w2v[word])
                found += 1

        # If we use CoreNLP's special tokens for brackets etc.
        # we need to load embeddings for the original token.
        specials = [
            ("-lrb-", "("),
            ("-rrb-", ")"),
            ("-lsb-", "["),
            ("-rsb-", "]"),
            ("-lcb-", "{"),
            ("-rcb-", "}"),
        ]

        for (special, token) in specials:
            if special in vocab:
                idx = vocab[special]
                weights[idx] = torch.from_numpy(w2v[token])
                found += 1

        # Ensure that padding is a zero vector
        weights[vocab.PAD] = torch.zeros(embed_size)
        embed_layer = nn.Embedding.from_pretrained(weights, freeze=False)
        embed_layer.padding_idx = vocab.PAD
        print("Loaded embeddings for %d/%d words" % (found, vocab_size))
        return embed_layer

    @staticmethod
    def _mask_and_avg(values, mask):
        """
        Mask and average a tensor of values, based on the given
        `padding_mask` of same shape as `values`. Used for calculating
        masked, averaged coverage loss.

        :param values:
            The values to be masked and averaged of shape (batch_size, T)
        :param padding_mask:
            The mask to be applied of same shape as `values`: (batch_size, T)

        :returns: A single valued tensor, with the average across the given `values`.
        """
        normalization_factor = torch.sum(mask, dim=1)
        masked = values * mask
        normalized = torch.sum(masked, dim=1) / normalization_factor
        return torch.mean(normalized)

    def _calc_loss(self, y_pred, batch, cov_losses=None):
        """Calculate loss given prediction, batch and optional cov_losses"""
        y = batch.targets

        if self.ignore_unknown_loss:
            # If ignoring loss for UNK, substitute UNK with PAD in target
            y[y == batch.vocab.UNK] = batch.vocab.PAD

        loss = self.criterion(torch.log(y_pred + EPSILON), y)
        if self.size_average:
            loss = torch.mean(loss.sum(dim=1) / batch.target_lengths.float().to(DEVICE))

        cov_loss = torch.zeros(1, device=DEVICE)  # dummy

        if self.coverage:
            cov_loss = self._mask_and_avg(cov_losses, batch.target_mask)
            cov_loss = cov_loss * self.coverage_loss_weight
            loss = loss + cov_loss

        if self.penalize_unknown:
            unk_penalty = torch.sum(y_pred[:, batch.vocab.UNK, :])
            loss = loss + unk_penalty

        return loss, cov_loss

    def forward(self, batch):
        """
        Computes the NLLLoss for each timestep, and returns the mean.

        :param batch: An instance of `data.Batch`

        :returns:
            Returns the loss, the part thereof that is coverage and the output.
            Output is of size (batch_size, max_target_length, (ext_)vocab_size)
            Returned as a triple (loss, cov_loss, output)
        """
        # Ensure contiguous memory. Necessary when running multi-GPU.
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
        batch.to(DEVICE)
        batch_size = len(batch)
        vocab = batch.vocab
        vocab_size = len(vocab)

        # Get word embeddings for encoder
        embedded = self.embedding(vocab.filter_oov(batch.inputs))
        # Run embeddings through encoder.
        enc_outputs, enc_state = self.encoder(embedded, batch.input_lengths)
        # Calculate encoder attention features (relevant only for some attn_models)
        enc_features = self.encoder_projection(enc_outputs)
        # Prepare input for decoder
        dec_input = torch.LongTensor([vocab.SOS] * batch_size).to(DEVICE)
        # Use last (forward) hidden state from encoder
        dec_state = self.reduce_state(enc_state)
        dec_context = torch.zeros(batch_size, self.hidden_size * 2, device=DEVICE)

        # Prepare tensor to store outputs
        max_target_length = batch.target_lengths.max()
        outputs = torch.zeros(batch_size, vocab_size, max_target_length, device=DEVICE)
        # Prepare tensors to store stepwise coverage loss
        step_cov_losses = torch.zeros(batch_size, max_target_length, device=DEVICE)

        coverage = (
            torch.zeros(batch_size, max(batch.input_lengths), device=DEVICE)
            if self.coverage
            else None
        )

        # Run through decoder one time step at a time
        for t in range(max_target_length):
            embedded = self.embedding(dec_input)
            dec_output, dec_state, dec_context, attn, new_coverage = self.decoder(
                embedded,
                dec_state,
                dec_context,
                enc_outputs,
                batch.input_mask,
                batch.inputs,
                vocab_size,
                coverage,
                enc_features,
            )

            if self.coverage:
                step_cov_loss = torch.sum(torch.min(coverage, attn), dim=1)
                step_cov_losses[:, t] = step_cov_loss
                coverage = new_coverage

            outputs[:, :, t] = dec_output

            # Next input is current target (teacher forcing)
            dec_input = batch.targets[:, t].clone()

            if self.sample_when_unknown:
                # sub UNKs in teacher forced input, if didn't predict OOV
                for i in range(batch_size):
                    if dec_input[i].item() == vocab.UNK:
                        pred = dec_output[i].argmax()
                        dec_input[i] = pred

            # Note that we do in place filter since we already cloned
            vocab.filter_oov_(dec_input)

        loss, cov_loss = self._calc_loss(outputs, batch, step_cov_losses)
        return (loss, cov_loss, outputs)
