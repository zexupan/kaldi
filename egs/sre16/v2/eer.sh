. ./path.sh
. ./cmd.sh

score=/home/panzexu/workspace/avss_speaker_embedding/src/evaluation/egs1/score_1
trials=/home/panzexu/workspace/avss_speaker_embedding/src/evaluation/egs1/trials

pooled_eer=$(paste $trials $score | awk '{print $6, $3}' | compute-eer - )
