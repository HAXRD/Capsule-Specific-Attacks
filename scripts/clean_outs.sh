#!/bin/bash

# clean outs folder
rm -rf ./scripts/*/*/outs

# caps_full
mkdir -p ./scripts/caps_full/evaluate/outs
mkdir -p ./scripts/caps_full/max_caps_dim_diff/outs
mkdir -p ./scripts/caps_full/max_norm_diff/outs
mkdir -p ./scripts/caps_full/naive_max_caps_dim/outs
mkdir -p ./scripts/caps_full/naive_max_norm/outs
mkdir -p ./scripts/caps_full/train/outs

# cnn
mkdir -p ./scripts/cnn/evaluate/outs
mkdir -p ./scripts/cnn/max_norm_diff/outs
mkdir -p ./scripts/cnn/naive_max_norm/outs
mkdir -p ./scripts/cnn/train/outs