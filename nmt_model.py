#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings

Hypothesis = namedtuple("Hypothesis", ["value", "score"])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, embed_size, hidden_size, num_classes):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size=embed_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.encoder = nn.LSTM(
            embed_size, hidden_size, num_layers=1, bidirectional=True
        )
        self.sentiment_projection = nn.Linear(2 * hidden_size, num_classes, bias=True)

    def forward(self, sents: List[List[str]], sentiments: List[int]) -> torch.Tensor:
        """ Take a mini-batch of sentences along with the associated sentiments. Outputs
        log-likelihood of each (sentence, sentiment) pair.

        @param sents (List[List[str]]): sentences of size (batch_size, XXXXXXX)
        @param sentiments (List[int]): sentiments of size (batch_size)

        @returns scores (Tensor): a variable/tensor of shape (batch_size, ) representing the
                                    log-likelihood of generating the correct sentiment
        """
        probs = self.step(sents)
        scores = probs[range(len(sentiments)), sentiments].log()
        return scores

    def step(self, sents):
        """ compute log probabilities of each output class for each input """

        sents_embedded, sents_length = self.model_embeddings.embed_sentence(
            sents, self.device
        )
        sents_packed = pack_padded_sequence(
            sents_embedded, sents_length, enforce_sorted=False
        )

        # pass through bi-directional LSTM
        _, (last_hidden, _) = self.encoder(sents_packed)
        # last_hidden is already h^forward_t and h^backward_0, so simply concatinating the two directions
        # gives the desired tensor to be projected into h_0^dec
        last_hidden = torch.cat((last_hidden[0], last_hidden[1]), -1)

        # pass through FC layer
        fc_output = self.sentiment_projection(
            last_hidden
        )  # size should be (batch_size, num_classes)
        # fc_output = nn.Dropout(0.6)(fc_output)
        probs = nn.Softmax(dim=-1)(fc_output)
        return probs

    def predict(self, sents):
        probs = self.step(sents)
        predictions = torch.argmax(probs, dim=-1)
        return predictions

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        # return self.model_embeddings.source.weight.device
        # return torch.device("cpu")
        return self.sentiment_projection.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params["args"]
        model = NMT(**args)
        model.load_state_dict(params["state_dict"])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print("save model parameters to [%s]" % path, file=sys.stderr)

        params = {
            "args": dict(
                embed_size=self.embed_size,
                hidden_size=self.hidden_size,
                num_classes=self.num_classes,
            ),
            "state_dict": self.state_dict(),
        }

        torch.save(params, path)
