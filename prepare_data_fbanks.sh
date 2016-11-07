#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. path.sh

stage=2

# Set paths to various datasets
swbd="/mingback/backup/caiwch_dir/LDC/LDC97S62"
fisher_dirs="/mingback/backup/caiwch_dir/LDC/LDC2004T19/fe_03_p1_tran/  /mingback/backup/caiwch_dir/LDC/LDC2005T19/fe_03_p2_tran/" # Set to "" if you don't have the fisher corpus
eval2000_dirs="/mingback/backup/caiwch_dir/LDC/LDC2002S09/hub5e_00 /mingback/backup/caiwch_dir/LDC/LDC2002T43"
conv_tab_dir="/home/caiwch/eesen/asr_egs/swbd/v1/conv.tab"

. parse_options.sh

if [ $stage -le 1 ]; then
  echo =====================================================================
  echo "             Data Preparation and FST Construction                 "
  echo =====================================================================
  # Use the same data preparation script from Kaldi
  local/swbd1_data_prep.sh $swbd  $conv_tab_dir || exit 1;

  # Construct the phoneme-based lexicon
  local/swbd1_prepare_phn_dict.sh || exit 1;

  # Compile the lexicon and token FSTs
  utils/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;

  # Train and compile LMs.
  local/swbd1_train_lms.sh data/local/train/text data/local/dict_phn/lexicon.txt data/local/lm $fisher_dirs || exit 1;

  # Compile the language-model FST and the final decoding graph TLG.fst
  local/swbd1_decode_graph.sh data/lang_phn data/local/dict_phn/lexicon.txt || exit 1;

  # Data preparation for the eval2000 set
  local/eval2000_data_prep.sh $eval2000_dirs
fi

if [ $stage -le 2 ]; then
  echo =====================================================================
  echo "                    FBank Feature Generation                       "
  echo =====================================================================
  fbankdir=fbank

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
#  steps/make_fbank.sh --cmd "$train_cmd" --nj 32 data/train exp/make_fbank/train $fbankdir || exit 1;
#  utils/fix_data_dir.sh data/train || exit;
#  steps/compute_cmvn_stats.sh data/train exp/make_fbank/train $fbankdir || exit 1;

  steps/make_fbank.sh --cmd "$train_cmd" --nj 10 data/eval2000 exp/make_fbank/eval2000 $fbankdir || exit 1;
  utils/fix_data_dir.sh data/eval2000 || exit;
  steps/compute_cmvn_stats.sh data/eval2000 exp/make_fbank/eval2000 $fbankdir || exit 1;

  # Use the first 4k sentences as dev set, around 5 hours
  utils/subset_data_dir.sh --first data/train 4000 data/train_dev
  n=$[`cat data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last data/train $n data/train_nodev

  # Create a smaller training set by selecting the first 100k utterances, around 110 hours
  utils/subset_data_dir.sh --first data/train_nodev 100000 data/train_100k
  local/remove_dup_utts.sh 200 data/train_100k data/train_100k_nodup

  # Finally the full training set, around 286 hours
  local/remove_dup_utts.sh 300 data/train_nodev data/train_nodup
fi
