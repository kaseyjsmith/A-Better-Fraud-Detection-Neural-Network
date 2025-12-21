#!/bin/bash

python3 -m scripts.train_architecture --arch baseline --epochs 50 --learning_rate 0.001
python3 -m scripts.train_architecture --arch wide --epochs 50 --learning_rate 0.001
python3 -m scripts.train_architecture --arch deep --epochs 5 --learning_rate 0.001
python3 -m scripts.train_architecture --arch resnet --epochs 5 --learning_rate 0.001
python3 -m scripts.train_architecture --arch batchnorm --epochs 50 --learning_rate 0.001

