import matplotlib.pyplot as plt
import numpy as np
import sys
import os

fig, axs = plt.subplots(2, sharex=True)

if len(sys.argv) == 1:  # no args
    filenames = ['train_logfiles/' + sorted(os.listdir('train_logfiles'))[-1],
                 'dev_logfiles/' + sorted(os.listdir('dev_logfiles'))[-1]]
else:
    filenames = sys.argv[1:]


for filename in filenames:
    f = np.loadtxt(filename, delimiter=',')

    iter_num = f[:, 1]
    loss = f[:, 2]
    axs[0].plot(iter_num, loss, label='loss_' + filename)
    if f.shape[1] > 3:
        accuracy = f[:, 3]
        axs[1].plot(iter_num, accuracy, label='accuracy_' + filename)


fig.legend()
fig.savefig('graphs/graph_out.png')