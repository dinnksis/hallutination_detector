#!/bin/bash
set -e

python src/data/dataset_builder.py
echo "dataset in data/train.csv"