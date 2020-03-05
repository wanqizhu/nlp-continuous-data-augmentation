#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytreebank
from model_embeddings import ModelEmbeddings

data = pytreebank.load_sst()

    

def load_train_data(embed_size=50, perct=1.):
    '''
    labeledTree.to_labeled_lines()[0] gives you a single sentence and its labeling

    we split it into X = list of words, Y = sentence's labeling

    By default, Y falls into [0, 1, 2, 3, 4]

    @returns: train, dev
        train: List[(List[words], sentiment)] for each sentence in dataset
        dev: ~
    '''
    M = ModelEmbeddings(embed_size=embed_size)
    X = [labeledTree.to_labeled_lines()[0][1].split(" ") for labeledTree in data['train']]
    Y = [labeledTree.to_labeled_lines()[0][0] for labeledTree in data['train']]
    
    train_size = int(len(X) * perct)
    X = X[:train_size]
    Y = Y[:train_size]
    X = M.embed_sentence(X)

    return (X, Y)


def load_dev_data(embed_size=50, dev_perct=1.):
    M = ModelEmbeddings(embed_size=embed_size)
    X_dev = [labeledTree.to_labeled_lines()[0][1].split(" ") for labeledTree in data['dev']]
    Y_dev = [labeledTree.to_labeled_lines()[0][0] for labeledTree in data['dev']]
    
    dev_size = int(len(X_dev) * dev_perct)
    X_dev = X_dev[:dev_size]
    Y_dev = Y_dev[:dev_size]
    X_dev = M.embed_sentence(X_dev)

    # dev data doesn't need to be augmented, hence it's already zipped and 
    # ready to be passed into model.forward()
    return list(zip(X_dev, Y_dev))  


def load_test_data(embed_size=50, perct=1.):
    '''
    labeledTree.to_labeled_lines()[0] gives you a single sentence and its labeling

    we split it into X = list of words, Y = sentence's labeling

    By default, Y falls into [0, 1, 2, 3, 4]

    @returns: test
        test: List[(List[words], sentiment)] for each sentence in dataset
    '''
    M = ModelEmbeddings(embed_size=embed_size)
    X = [labeledTree.to_labeled_lines()[0][1].split(" ") for labeledTree in data['test']]
    Y = [labeledTree.to_labeled_lines()[0][0] for labeledTree in data['test']]

    test_size = int(len(X) * perct)
    X = X[:test_size]
    Y = Y[:test_size]
    X = M.embed_sentence(X)

    return list(zip(X, Y))


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (sentences, sentiments)): list of tuples containing sentence & sentiment labels
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)  # e = one (sentence, sentiment) sorting by len(sentences)
        sentences = [e[0] for e in examples]
        sentiments = [e[1] for e in examples]

        yield sentences, sentiments

