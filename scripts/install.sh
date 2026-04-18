#!/bin/bash
set -e

echo "istalling"
pip install --upgrade pip
pip install torch transformers pandas numpy scikit-learn tqdm huggingface_hub

echo "sucsessful"