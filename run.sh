# #!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --hidden-size 100 --log-every 100 --max-epoch 1000 --cuda
elif [ "$1" = "test" ]; then
  CUDA_VISIBLE_DEVICES=0 python3 run.py test model.bin --cuda
elif [ "$1" = "train_local" ]; then
	python3 run.py train --train-perct 0.2 --valid-niter 100
elif [ "$1" = "test_local" ]; then
  python3 run.py test model.bin
elif [ "$1" = "clean_logs" ]; then
  rm *_logfiles/*
else
	echo "Invalid Option Selected"
fi
