#!/bin/bash

TRANS_DIR='./swb_ms98_transcriptions'
word_trans_list=`find ${TRANS_DIR} -name "*word*" | sort`
all_words='all_words_trans.txt'

for ind in ${word_trans_list[@]}
do
    echo "${ind}"
    cat ${ind} >> ${all_words}
done
