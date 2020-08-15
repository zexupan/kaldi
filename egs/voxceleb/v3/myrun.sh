#!/usr/bin/env bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

main_path=pretrain
audio_from=Audio_2mix_min
source_data_path="/home/panzexu/datasets/LRS2/audio/${audio_from}"
xvector_npy_path="/home/panzexu/datasets/LRS2/xvector_2mix_min/"

main_data_path=exp/${audio_from}/${main_path}/data
mfcc_path=exp/${audio_from}/${main_path}/MFCC
vad_path=exp/${audio_from}/${main_path}/VAD
vector_path=exp/${audio_from}/${main_path}/Xvector

stage=0

if [ $stage -le 0 ]; then
  local/make_lrs2.pl $source_data_path ${main_path} $main_data_path
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    $main_data_path exp1/make_mfcc $mfcc_path
  utils/fix_data_dir.sh $main_data_path
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    $main_data_path exp1/make_vad $vad_path
  utils/fix_data_dir.sh $main_data_path
fi

if [ $stage -le 2 ]; then
  # Extract x-vectors for centering, LDA, and PLDA training.
  nnet_dir=model/exp/xvector_nnet_1a
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
    $nnet_dir $main_data_path \
    $vector_path
fi

if [ $stage -le 3 ]; then
  # Extract x-vectors for centering, LDA, and PLDA training.
  python x-vector.py \
  --xvector_path $vector_path \
  --xvector_npy_path $xvector_npy_path \
  --main_folder $main_path
fi
