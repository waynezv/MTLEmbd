#!/bin/bash

ark_files=`find ./ -name 'raw_fbank*ark'`
for file in ${ark_files}
do
		save_name="${file%'.ark'}"".txt"
		echo "Converting ${file} to ${save_name}"
	    copy-feats ark:${file} ark,t:${save_name}
done
