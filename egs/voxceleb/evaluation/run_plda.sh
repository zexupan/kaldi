
. ./cmd.sh
. ./path.sh

set -e

id=qkv_gau_train

voxceleb2_root=/home/panzexu/Download/eer/runs/${id}/

voxceleb1_trials=/home/panzexu/workspace/kaldi/egs/voxceleb/evaluation/feat/new_trials
voxceleb2_train_save_root=/home/panzexu/workspace/kaldi/egs/voxceleb/evaluation/feat/data_train
voxceleb2_test_save_root=/home/panzexu/workspace/kaldi/egs/voxceleb/evaluation/feat/data_test

stage=0

if [ $stage -le 0 ]; then
  local/make_voxceleb2.pl $voxceleb2_root train $voxceleb2_train_save_root
  local/make_voxceleb2.pl $voxceleb2_root test $voxceleb2_test_save_root
fi

if [ $stage -le 1 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd feat/log/${id}_train_compute_mean.log \
    ivector-mean scp:feat/${id}_train_feat.scp \
    feat/exp/${id}_train_mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd feat/log/${id}_train_lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:feat/${id}_train_feat.scp ark:- |" \
    ark:feat/data_train/utt2spk feat/exp/${id}_train_transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd feat/log/${id}_train_plda.log \
    ivector-compute-plda ark:feat/data_train/spk2utt \
    "ark:ivector-subtract-global-mean scp:feat/${id}_train_feat.scp ark:- | transform-vec feat/exp/${id}_train_transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    feat/exp/${id}_plda || exit 1;
fi

if [ $stage -le 2 ]; then
  $train_cmd feat/log/${id}_test_score.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 feat/exp/${id}_plda - |" \
    "ark:ivector-subtract-global-mean feat/exp/${id}_train_mean.vec scp:feat/${id}_test_feat.scp ark:- | transform-vec feat/exp/${id}_train_transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean feat/exp/${id}_train_mean.vec scp:feat/${id}_test_feat.scp ark:- | transform-vec feat/exp/${id}_train_transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" feat/exp/scores_${id}_test || exit 1;
fi

if [ $stage -le 3 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials feat/exp/scores_${id}_test) 2> /dev/null`
  # mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_kaldi_test $voxceleb1_trials 2> /dev/null`
  # mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_kaldi_test $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  # echo "minDCF(p-target=0.01): $mindcf1"
  # echo "minDCF(p-target=0.001): $mindcf2"
  # EER: 3.128%
  # minDCF(p-target=0.01): 0.3258
  # minDCF(p-target=0.001): 0.5003
  #
  # For reference, here's the ivector system from ../v1:
  # EER: 5.329%
  # minDCF(p-target=0.01): 0.4933
  # minDCF(p-target=0.001): 0.6168
fi
