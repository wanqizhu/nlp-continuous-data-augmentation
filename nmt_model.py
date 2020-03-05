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
import numpy as np

from model_embeddings import ModelEmbeddings

Hypothesis = namedtuple("Hypothesis", ["value", "score"])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, embed_size, hidden_size, num_classes, dropout_rate=0.3,
                 data_augmenter=None):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param num_classes (int): Num classes in classification task
        @param dropout_rate (float): Dropout probability, for attention
        @param data_augmenter (DataAugmenter instance or None): type of data augmentation to use
        """
        super(NMT, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.data_augmenter = data_augmenter

        self.encoder = nn.LSTM(
            embed_size, hidden_size, num_layers=1, bidirectional=True
        )
        self.sentiment_projection = nn.Linear(2 * hidden_size, num_classes, bias=True)
        self.dropout = nn.Dropout(dropout_rate)


    def pad_and_pack_data(self, sentences):
        '''
        @param sentences: List[embedded_vectors], each is a numpy array
        '''
        
        sentences_length = list(map(len, sentences))
        max_len = max(sentences_length)

        sentences_padded = [np.pad(s, ((0, max_len - len(s)), (0, 0))) for s in sentences]
        sentences_padded = torch.tensor(sentences_padded, device=self.device).permute(1, 0, 2)

        sentences_packed = pack_padded_sequence(
            sentences_padded, sentences_length, enforce_sorted=False
        )

        return sentences_packed


    def forward(self, sents: List[np.ndarray], sentiments: List[int]) -> torch.Tensor:
        """ Take a mini-batch of sentences along with the associated sentiments. Outputs
        log-likelihood of each (sentence, sentiment) pair.

        @param sents (List[List[str]]): sentences of size (batch_size, XXXXXXX)
        @param sentiments (List[int]): sentiments of size (batch_size)

        @returns scores (Tensor): a variable/tensor of shape (batch_size, ) representing the
                                    log-likelihood of generating the correct sentiment
        """
        sentences_packed = self.pad_and_pack_data(sents)
        probs = self.step(sentences_packed)
        scores = probs[range(len(sentiments)), sentiments].log()
        return scores

    def step(self, sentences_packed):
        """ compute log probabilities of each output class for each input """
        # pass through bi-directional LSTM
        _, (last_hidden, _) = self.encoder(sentences_packed)
        # last_hidden is already h^forward_t and h^backward_0, so simply concatinating the two directions
        # gives the desired tensor to be projected into h_0^dec
        last_hidden = torch.cat((last_hidden[0], last_hidden[1]), -1)

        # pass through FC layer
        fc_output = self.sentiment_projection(
            last_hidden
        )  # size should be (batch_size, num_classes)
        fc_output = self.dropout(fc_output)
        probs = nn.Softmax(dim=-1)(fc_output)
        return probs

    def predict(self, sentences):
        # DO NOT augment data here
        sentences_packed = self.pad_and_pack_data(sentences)
        probs = self.step(sentences_packed)
        predictions = torch.argmax(probs, dim=-1)
        return predictions


    def compute_accuracy(self, sentences, sentiments):
        '''
        Return % accuracy of prediction compared to true labels
        '''
        predictions = self.predict(sentences)
        sentiments = torch.tensor(sentiments, device=self.device)
        return (predictions == sentiments).sum().item() / len(sentences)


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
                dropout_rate=self.dropout_rate,
                data_augmenter=self.data_augmenter
            ),
            "state_dict": self.state_dict(),
        }

        torch.save(params, path)
