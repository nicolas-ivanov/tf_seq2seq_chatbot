#!/usr/bin/env bash

# create directories to store results locally
dir='/var/lib/try_seq2seq'
sudo mkdir -p $dir'/corpora_raw/'
sudo mkdir -p $dir'/corpora_processed/'
sudo mkdir -p $dir'/words_index/'
sudo mkdir -p $dir'/w2v_models/'
sudo mkdir -p $dir'/nn_models/'
sudo mkdir -p $dir'/results/'
sudo chown -R "$USER" $dir

# install required packages
pip install -r requirements.txt

# build and install seq2seq package
cd $dir
git clone git@github.com:LeavesBreathe/Seq2Seq_Upgrade_TensorFlow.git
cd Seq2Seq_Upgrade_TensorFlow
sudo python setup.py build
sudo python setup.py install