#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from utils import pad_sents


class ModelEmbeddings:
    """
    Class that converts input words to their embeddings.
    """

    def __init__(self, embed_file="glove.6B.50d.txt.40k", embed_size=50):
        """
        Init the Embedding layers.
        """

        self.word_to_idx = {"<pad>": 0}

        embeddings = [[0] * embed_size]  # pad maps to all zeros

        with open(embed_file, "r") as f:
            cnt = 1
            for line in f:
                line = line.strip().split(" ")
                word = line[0]
                emb = line[1:]
                if len(emb) != embed_size:
                    continue

                self.word_to_idx[word] = cnt
                cnt += 1
                embeddings.append(emb)

        self.embeddings = np.array(embeddings, dtype=float)
        print("embeddings shape", self.embeddings.shape)

    def embed_sentence(self, sents, device):
        """
        sents: List[List[str]] of length batch_size, and variable sentence length inside

        @returns: tensor of (max_sentence_length, batch_size, embed_size)
        @returns: list of (batch_size)
        """

        # print(sents)
        # print(device)

        sentsIdx = [
            [self.word_to_idx[wd] if wd in self.word_to_idx else 0 for wd in sent]
            for sent in sents
        ]
        sents_padded = pad_sents(sentsIdx, 0)
        sents_padded = np.array(sents_padded)

        sent_lengths = [len(s) for s in sents]
        sents_embedded = self.embeddings[sents_padded.T]
        # print(sents_embedded.shape)
        return (
            torch.tensor(sents_embedded, dtype=torch.float, device=device),
            sent_lengths,
        )


# m = ModelEmbeddings()
# embed_out = m.embed_sentence([["hi", "there"], ["wow"], ["huh", "a", "but", "long"]],
#                  device=torch.device("cpu"))

# print(embed_out.shape)

