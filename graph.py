import matplotlib.pyplot as plt
import numpy as np
import sys
import os


if len(sys.argv) == 1:  # no args
    filenames = ['train_logfiles/' + sorted(os.listdir('train_logfiles'))[-1],
                 'dev_logfiles/' + sorted(os.listdir('dev_logfiles'))[-1]]
else:
    filenames = sys.argv[1:]


for filename in filenames:
    f = np.loadtxt(filename, delimiter=',')

    iter_num = f[:, 1]
    loss = f[:, 2]
    plt.plot(iter_num, loss, label='loss_' + filename)
    if f.shape[1] > 3:
        accuracy = f[:, 3]
        plt.plot(iter_num, accuracy, label='accuracy_' + filename)


plt.legend()
plt.show()