#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 4
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>

Usage:
    run.py train [options]
    run.py test [options] MODEL_PATH

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --data-aug=<string>                     data augmentation method [default: "None"]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --num-classes=<int>                     num classes in sentiment prediction [default: 5]
    --embed-size=<int>                      embedding size [default: 50]
    --hidden-size=<int>                     hidden size [default: 10]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 50]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --train-perct=<float>                   % of training data to use [default: 1.0]
    --dev-perct=<float>                     % of dev data to use [default: 1.0]
"""
import math
import sys
import pickle
import time


from docopt import docopt
from nmt_model import NMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import batch_iter, load_train_data, load_dev_data, load_test_data
from collections import defaultdict
from data_augmenter import BaseDataAugmenter, GaussianNoiseDataAugmenter

import torch
import torch.nn.utils

# TODO sample random predictions?
def evaluate_dev(model, dev_data, batch_size):
    """
        higher is betterrrrrrrr
    """
    was_training = model.training
    model.eval()

    cum_score = 0.0
    cum_correct = 0

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for sentences, sentiments in batch_iter(dev_data, batch_size):
            score = model(sentences, sentiments).sum()
            cum_score += score.item()
            cum_correct += model.compute_accuracy(sentences, sentiments) * len(
                sentences
            )

    if was_training:
        model.train()

    # return: loss, accuracy
    return cum_score / len(dev_data), cum_correct / len(dev_data)


# TODO
def test(args):
    print("load model from {}".format(args["MODEL_PATH"]), file=sys.stderr)
    model = NMT.load(args["MODEL_PATH"])

    if args["--cuda"]:
        model = model.to(torch.device("cuda:0"))

    test_data = load_test_data()
    batch_size = int(args["--batch-size"])

    cum_correct = 0
    cum_score = 0

    with torch.no_grad():
        for sentences, sentiments in batch_iter(test_data, batch_size):
            correct = model.compute_accuracy(sentences, sentiments) * len(sentences)
            cum_correct += correct
            score = model(sentences, sentiments).sum()
            cum_score += score

    print("test dataset size: %d" % len(test_data))
    print("accuracy: %f" % (cum_correct / len(test_data)))
    print("loss: %f" % (cum_score / len(test_data)))


def print_and_write(s, f):
    print(s)
    f.write(s + "\n")


def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    long_logfile = "long_logfiles/" + str(time.time()) + "long.txt"
    train_logfile = "train_logfiles/" + str(time.time()) + "train.txt"
    dev_logfile = "dev_logfiles/" + str(time.time()) + "dev.txt"
    f_long = open(long_logfile, "w")
    f_train = open(train_logfile, "w")
    # TODO: add hyperparameters
    f_train.write("#args: %s" % args)
    f_train.write("#epoch, train iter, train score\n")
    f_dev = open(dev_logfile, "w")
    f_dev.write("#epoch, train iter, dev score, dev accuracy\n")

    train_data = load_train_data(perct=float(args["--train-perct"]))
    dev_data = load_dev_data(dev_perct=float(args["--dev-perct"]))

    train_batch_size = int(args["--batch-size"])
    clip_grad = float(args["--clip-grad"])
    valid_niter = int(args["--valid-niter"])
    log_every = int(args["--log-every"])
    model_save_path = args["--save-to"]

    embed_size = int(args["--embed-size"])

    # TODO: load train data_augmenter based on args
    data_augmenter = str(args["--data-aug"])
    print_and_write("Using data augmentation method: %s" % data_augmenter, f_long)
    if data_augmenter == "gaussian":
        data_augmenter = GaussianNoiseDataAugmenter()
    else:
        data_augmenter = BaseDataAugmenter()

    # perform augmentation
    train_data_aug = data_augmenter.augment(train_data)
    print_and_write(
        "train size: %d, after aug %d" % (len(train_data[0]), len(train_data_aug)),
        f_long,
    )

    model = NMT(
        embed_size=embed_size,
        hidden_size=int(args["--hidden-size"]),
        num_classes=int(args["--num-classes"]),
        dropout_rate=float(args["--dropout"])
    )
    model.train()

    uniform_init = float(args["--uniform-init"])
    if np.abs(uniform_init) > 0.0:
        print_and_write(
            "uniformly initialize parameters [-%f, +%f]" % (uniform_init, uniform_init),
            f_long,
        )
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    device = torch.device("cuda:0" if args["--cuda"] else "cpu")
    print_and_write("use device: %s" % device, f_long)
    model = model.to(device)
    print_and_write("confirming model device %s" % model.device, f_long)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args["--lr"]))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print_and_write("begin Maximum Likelihood training", f_long)

    while True:
        epoch += 1

        for sentences, sentiments in batch_iter(
            train_data_aug, batch_size=train_batch_size, shuffle=True
        ):
            train_iter += 1

            optimizer.zero_grad()

            example_losses = -model(sentences, sentiments)  # (batch_size,)
            batch_size = len(example_losses)  # in case data augmentation makes returned
            # number of examples > input batch size
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                # train_accuracy = model.compute_accuracy(sentences, sentiments)
                print_and_write(
                    "epoch %d, iter %d, avg. loss %.2f, "
                    "cum. examples %d, time elapsed %.2f sec"
                    % (
                        epoch,
                        train_iter,
                        report_loss / report_examples,
                        cum_examples,
                        time.time() - begin_time,
                    ),
                    f_long,
                )
                f_train.write(
                    "%d, %d, %.2f\n"
                    % (epoch, train_iter, report_loss / report_examples)
                )

                train_time = time.time()
                report_loss = report_examples = 0.0

            # perform validation
            if train_iter % valid_niter == 0:
                cum_loss = cum_examples = 0.0
                valid_num += 1

                print_and_write("begin validation ...", f_long)

                # compute dev
                dev_score, dev_accuracy = evaluate_dev(
                    model, dev_data, batch_size=5000
                )  # dev batch size can be a bit larger
                valid_metric = dev_score  # maybe use accuracy instead?

                print_and_write(
                    "validation: iter %d, dev. score %f, dev. accuracy %f"
                    % (train_iter, dev_score, dev_accuracy),
                    f_long,
                )
                f_dev.write(
                    "%d, %d, %f, %f\n" % (epoch, train_iter, dev_score, dev_accuracy)
                )

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(
                    hist_valid_scores
                )
                hist_valid_scores.append(valid_metric)

                # train_score = evaluate_dev(model, train_data, batch_size=100000)

                # see some trainig examples
                # with torch.no_grad():
                #     print("[training] sample predictions")
                #     # print("sent\t true sentiment\t predicted sentiment")
                #     sents = sentences[:5]
                #     sentis = sentiments[:5]
                #     predictions = model.predict(sents)
                #     probabilities = model.step(sents)

                #     for i in range(5):
                #         print(
                #             # " ".join(sents[i]),
                #             sentis[i],
                #             predictions[i],
                #             probabilities[i],
                #         )

                #     counts = defaultdict(int)
                #     for pred in predictions:
                #         counts[int(pred)] += 1
                #     print(counts)

                if is_better:
                    patience = 0
                    print_and_write(
                        "save currently the best model to [%s]" % model_save_path,
                        f_long,
                    )
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + ".optim")
                elif patience < int(args["--patience"]):
                    patience += 1
                    print_and_write("hit patience %d" % patience, f_long)

                    if patience == int(args["--patience"]):
                        num_trial += 1
                        print_and_write("hit #%d trial" % num_trial, f_long)
                        if num_trial == int(args["--max-num-trial"]):
                            print_and_write("early stop!", f_long)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]["lr"] * float(args["--lr-decay"])
                        print_and_write(
                            "load previously best model and decay learning rate to %f"
                            % lr,
                            f_long,
                        )

                        # load model
                        params = torch.load(
                            model_save_path, map_location=lambda storage, loc: storage
                        )
                        model.load_state_dict(params["state_dict"])
                        model = model.to(device)

                        print_and_write("restore parameters of the optimizers", f_long)
                        optimizer.load_state_dict(
                            torch.load(model_save_path + ".optim")
                        )

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args["--max-epoch"]):
                    print_and_write("reached maximum number of epochs!", f_long)
                    exit(0)


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check pytorch version
    assert (
        torch.__version__ >= "1.0.0"
    ), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(
        torch.__version__
    )

    # seed the random number generators
    seed = int(args["--seed"])
    torch.manual_seed(seed)
    if args["--cuda"]:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args["train"]:
        train(args)
    elif args["test"]:
        test(args)
    else:
        raise RuntimeError("invalid run mode")


if __name__ == "__main__":
    main()
