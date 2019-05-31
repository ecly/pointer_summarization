"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Module with a BeamSearch implementation, that is fully coupled
to the `Seq2Seq` implementation from `model.py`.
Currently only support batches of size=1.
"""
import torch
import torch.nn as nn
from model import Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Beam:
    """
    Class representing a single `Beam` or `Hypothesis`
    instance during beam search evaluation.
    """

    def __init__(self, tokens, log_prob, dec_state, dec_context, coverage):
        """
        :param tokens: A list of previously generated token ids
        :param log_prob: The beam's current total log_prob
        :param dec_state:
            The decoder hidden state at the latest step of size (1, hidden_size)
        :param dec_context:
            The decoder context at the latest step of size (1, hidden_size)
        :param coverage:
            The most recent attention coverage of size (token_count, 1)
        """
        self.tokens = tokens
        self.log_prob = log_prob
        self.dec_state = dec_state
        self.dec_context = dec_context
        self.coverage = coverage

    @property
    def latest_token(self):
        """Return the most recently generated token"""
        return self.tokens[-1]

    def __len__(self):
        """Minus 1, since the first token is SOS"""
        return len(self.tokens) - 1

    def extend(self, token, log_prob, dec_state, dec_context, coverage):
        """
        Advance the current beam with the following `token`, `log_prob`,
        `dec_hidden`, `dec_context` and `coverage` and return
        the newly created Beam.
        """
        return Beam(
            self.tokens + [token],
            self.log_prob + log_prob,
            dec_state,
            dec_context,
            coverage,
        )


class BeamSearch:
    """
    A BeamSearch class able to perform beam search decoding on
    `Seq2Seq` models. Fully dependent on `Seq2Seq` fields and underlying
    modules, and am primarily separated into its own class to avoid
    cluttering the `Seq2Seq` implementation.
    """

    def __init__(self, model, beam_size=4, min_steps=35, max_steps=120, cfg=None):
        """
        :param model: The `Seq2Seq` model used for beam search decoding
        :param beam_size: The size of the beam during decoding (k)
        :param min_steps: The minimum length of a result
        :param max_steps: The maximum length of a result
        :param cfg:
            Optional `Config` instance from which parameters will be read from if given
        """
        if isinstance(model, Seq2Seq):
            self.model = model
        elif isinstance(model, nn.DataParallel):
            self.model = model.module
        else:
            raise ValueError("Model must be Seq2Seq or Seq2Seq wrapped in DataParallel")

        if cfg is None:
            self.beam_size = beam_size
            self.min_steps = min_steps
            self.max_steps = max_steps
            self.length_normalize = "avg"
            self._alpha = 0.0
            self.coverage_penalty = False
            self._beta = 0.0
            self.block_ngram_repeat = 0
            self.block_unknown = False
            self.block_repeat = False
        else:
            self.beam_size = cfg.beam_size
            self.min_steps = cfg.min_decode_steps
            self.max_steps = cfg.max_decode_steps
            self.length_normalize = cfg.length_normalize.lower()
            self._alpha = cfg.length_normalize_alpha
            self.coverage_penalty = cfg.coverage_penalty
            self._beta = cfg.coverage_penalty_beta
            self.block_ngram_repeat = cfg.block_ngram_repeat
            self.block_unknown = cfg.block_unknown
            self.block_repeat = cfg.block_repeat

    @staticmethod
    def length_normalize_wu(length, alpha=0.0):
        """Compute length penalty as Wu et al. 2017"""
        return ((5.0 + length) / 6.0) ** alpha

    @staticmethod
    def calc_coverage_penalty(cov, beta=0.0):
        """
        Compute summarization coverage penalty as Wu et al. 2017,
        as defined in Gehrmann et al. (2018). Penalizes coverage>1
        for any token.
        """
        n = cov.size(0)
        return beta * (-n + cov.clamp_min(1).sum().item())

    def score_beam(self, beam):
        """Compute the score of a beam, including penalties per configuration"""
        score = beam.log_prob
        if self.length_normalize == "wu":
            score /= self.length_normalize_wu(len(beam), self._alpha)
        elif self.length_normalize == "avg":
            score /= len(beam)
        if self.coverage_penalty:
            score += self.calc_coverage_penalty(beam.coverage, self._beta)

        return score

    def sort_beams(self, beams):
        """Sort beams in descending likelihood"""
        return sorted(beams, key=self.score_beam, reverse=True)

    @staticmethod
    def block_unknown_(outputs, unk_idx):
        """Prevent the model from producing unknown"""
        for i in range(outputs.size(0)):
            outputs[i, unk_idx] = -10e20

    @staticmethod
    def block_repeat_(beams, outputs):
        """Prevent the prediction of a word two times in a row"""
        assert len(beams) == outputs.size(0)
        for idx, b in enumerate(beams):
            outputs[idx, b.latest_token] = -10e20

    def block_ngrams_(self, beams, outputs):
        """
        Block ngrams in place for the given outputs based on given beams.
        We expect the beams and outputs to be ordered by correspondance.
        """
        assert len(beams) == outputs.size(0)
        block = self.block_ngram_repeat
        if len(beams[0]) < (block):
            return

        for idx, b in enumerate(beams):
            sub_gram = tuple(b.tokens[-(block - 1) :])
            seen = set()
            output = outputs[idx]
            for i in range(len(b.tokens) - block + 1):
                seen.add(tuple(b.tokens[i : i + block]))
            while True:
                next_token = output.argmax().item()
                next_gram = sub_gram + (next_token,)
                if next_gram in seen:
                    output[next_token] = -10e20
                else:
                    break

    def get_state(self, beams):
        """Get the hidden state for a batch of beams"""
        if self.model.rnn_cell == "lstm":
            hidden = list(map(lambda b: b.dec_state[0], beams))
            hidden = torch.cat(hidden, dim=1).to(DEVICE)

            cell_state = list(map(lambda b: b.dec_state[1], beams))
            cell_state = torch.cat(cell_state, dim=1).to(DEVICE)
            return (hidden, cell_state)

        state = list(map(lambda b: b.dec_state, beams))
        state = torch.cat(state, dim=1).to(DEVICE)
        return state

    def search(self, batch):
        """
        Find most probable output for the given batch using beam search.

        :param batch: An instance of `Batch`. The batch's `targets` are ignored.
        :param beam_size: The size of the beam for the beam search.
        :param min_steps:
            The minimum length of a summary produced. If EOS is
            produced prior to this threshold, the summary is discarded.
        :param max_steps: The maximum length of the summaries produced.
        """
        m = self.model  # alias to shorten code
        input_seqs = batch.inputs.to(DEVICE)
        input_mask = batch.input_mask.to(DEVICE)
        assert len(batch) == 1, "beam search currently only supports batch size 1"
        vocab = batch.vocab

        # Get word embeddings.
        embedded = m.embedding(vocab.filter_oov(input_seqs))
        # Run embeddings through encoder.
        enc_outputs, enc_hidden = m.encoder(embedded, batch.input_lengths)
        # Expand enc_outputs, and features to beam_size
        enc_outputs = enc_outputs.expand(self.beam_size, -1, -1)
        enc_features = m.encoder_projection(enc_outputs)

        # Prepare input for decoder
        dec_state = m.reduce_state(enc_hidden)
        dec_context = torch.zeros(1, m.hidden_size * 2, device=DEVICE)
        dec_input = torch.LongTensor([vocab.SOS]).to(DEVICE)

        beams = [
            Beam(
                [vocab.SOS],
                0.0,
                dec_state,
                dec_context,
                torch.zeros(max(batch.input_lengths), device=DEVICE)
                if m.coverage
                else None,
            )
        ] * self.beam_size
        results = []
        step = 0
        while len(results) < self.beam_size and step < self.max_steps:
            latest_tokens = list(map(lambda b: b.latest_token, beams))
            dec_input = torch.LongTensor(latest_tokens).to(DEVICE)
            dec_input = m.embedding(vocab.filter_oov(dec_input))

            dec_state = self.get_state(beams)

            dec_context = list(map(lambda b: b.dec_context, beams))
            dec_context = torch.cat(dec_context, dim=0).to(DEVICE)

            coverage = None
            if m.coverage:
                coverage = list(map(lambda b: b.coverage, beams))
                coverage = torch.stack(coverage, dim=0).to(DEVICE)

            dec_output, dec_state, dec_context, _attn, coverage = m.decoder(
                dec_input,
                dec_state,
                dec_context,
                enc_outputs,
                input_mask,
                input_seqs,
                len(vocab),
                coverage,
                enc_features,
            )

            log_probs = torch.log(dec_output)
            # If block ngram repetitions, we zero out probabilities of
            # reptitions of the configured size. Done inplace on outputs.
            if self.block_ngram_repeat > 0:
                self.block_ngrams_(beams, log_probs)
            if self.block_unknown:
                self.block_unknown_(log_probs, vocab.UNK)
            if self.block_repeat:
                self.block_repeat_(beams, log_probs)

            # If first decoder step, our decoder might produce EOS
            # as one of the topk. This would result in only 3 possible
            # continuations, unless we make sure to take k+1
            k = self.beam_size + 1 if step == 0 else self.beam_size
            top_probs, top_idx = log_probs.data.topk(k)
            new_beams = []
            # if first iteration, we only care about the first as they are all identical
            for i in range(1 if step == 0 else len(beams)):
                beam = beams[i]
                state = (
                    (
                        dec_state[0][:, i, :].unsqueeze(1),
                        dec_state[1][:, i, :].unsqueeze(1),
                    )
                    if m.rnn_cell == "lstm"
                    else dec_state[:, i, :].unsqueeze(1)
                )

                coverage_ = coverage[i, :] if m.coverage else None
                context = dec_context[i].unsqueeze(0)
                for j in range(k):
                    new_beams.append(
                        beam.extend(
                            token=top_idx[i, j].item(),
                            log_prob=top_probs[i, j].item(),
                            dec_state=state,
                            dec_context=context,
                            coverage=coverage_,
                        )
                    )

            beams = []  # empty list and fill with new candidates
            for beam in self.sort_beams(new_beams):
                if beam.latest_token == vocab.EOS:
                    if step >= self.min_steps:
                        results.append(beam)
                else:
                    beams.append(beam)
                if len(beams) == self.beam_size or len(results) == self.beam_size:
                    break

            step += 1

        if not results:
            results = beams

        return [self.sort_beams(results)[0].tokens]
