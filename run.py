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
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --num-classes=<int>                     num classes in sentiment prediction [default: 5]
    --embed-size=<int>                      embedding size [default: 50]
    --hidden-size=<int>                     hidden size [default: 100]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
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
from utils import batch_iter, load_training_data

import torch
import torch.nn.utils

# TODO sample random predictions?
def evaluate_dev(model, dev_data, batch_size):
    '''
        higher is betterrrrrrrr
    '''
    was_training = model.training
    model.eval()

    cum_score = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for sentences, sentiments in batch_iter(dev_data, batch_size):
            score = model(sentences, sentiments).sum()
            cum_score += score.item()
            

        print('sample predictions')
        print('sent\t true sentiment\t predicted sentiment')
        sents = sentences[:10]
        sentis = sentiments[:10]
        predictions = model.predict(sents)
        probabilities = model.step(sents)

        for i in range(10):
            print(" ".join(sents[i]), sentis[i], predictions[i], probabilities[i])
        print('\n\n')

    if was_training:
        model.train()

    return cum_score / len(dev_data)



# TODO
def predict(args):
    # TODO
    pass

    # # model, test_data, batch_size):
    # cum_correct = 0

    # with torch.no_grad():
    #     for sentences, sentiments in batch_iter(test_data, batch_size):
    #         predictions = model.predict(sentences)
    #         accuracy = (sentences == sentiments).sum()
    #         cum_correct += accuracy.item()


    # return cum_correct / len(test_data)


def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """

    # TODO
    train_data, dev_data = load_training_data(size=500, dev_size=100)

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                num_classes=int(args['--num-classes']))
    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)
    print('confirming model device', model.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for sentences, sentiments in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            example_losses = -model(sentences, sentiments) # (batch_size,)
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
                print('epoch %d, iter %d, avg. loss %.2f' \
                      'cum. examples %d, time elapsed %.2f sec' % (epoch, train_iter,
                                                                 report_loss / report_examples,
                                                                 cum_examples,
                                                                 time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_examples = 0.


            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev
                dev_score = evaluate_dev(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_score

                print('validation: iter %d, dev. score %f' % (train_iter, dev_score), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)


                # see some trainig examples
                with torch.no_grad():
                    print('[training] sample predictions')
                    print('sent\t true sentiment\t predicted sentiment')
                    sents = sentences[:5]
                    sentis = sentiments[:5]
                    predictions = model.predict(sents)
                    probabilities = model.step(sents)

                    for i in range(5):
                        print(" ".join(sents[i]), sentis[i], predictions[i], probabilities[i])


                # if is_better:
                #     patience = 0
                #     print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                #     model.save(model_save_path)

                #     # also save the optimizers' state
                #     torch.save(optimizer.state_dict(), model_save_path + '.optim')
                # elif patience < int(args['--patience']):
                #     patience += 1
                #     print('hit patience %d' % patience, file=sys.stderr)

                #     if patience == int(args['--patience']):
                #         num_trial += 1
                #         print('hit #%d trial' % num_trial, file=sys.stderr)
                #         if num_trial == int(args['--max-num-trial']):
                #             print('early stop!', file=sys.stderr)
                #             exit(0)

                #         # decay lr, and restore from previously best checkpoint
                #         lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                #         print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                #         # load model
                #         params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                #         model.load_state_dict(params['state_dict'])
                #         model = model.to(device)

                #         print('restore parameters of the optimizers', file=sys.stderr)
                #         optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                #         # set new lr
                #         for param_group in optimizer.param_groups:
                #             param_group['lr'] = lr

                #         # reset patience
                #         patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)



def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check pytorch version
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['test']:
        test(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
