from __future__ import print_function
import os
import csv
import re
import argparse as agp
import numpy as np

# FBANK_TXT_DIR = '/home/caiwch/eesen/asr_egs/swbd/v1/fbank/fbank_txt'
FBANK_TXT_DIR = '.'

class AlignWordFbank(object):
    def __init__(self):
        self.word = {}

    def split_fbank(self, FBANK_TXT_DIR):
        with open(os.path.join(FBANK_TXT_DIR, 'raw_fbank_train.1.txt'), 'r') as f:
            pat1 = re.compile('.+\[\n')
            pat2 = re.compile('.+\]\n')
            fbank = []
            for line in f:
                if pat1.match(line):
                    print(line.split(' ')[0])
                    fbank_id = line.split('_')[0]
                    utt_id = fbank_id.split('-')[0]
                    chn_id = fbank_id.split('-')[1]
                    self.word['utter_id'] = utt_id
                    self.word['channel'] = chn_id
                    self.word['speaker_id'] = '00001'
                    self.word['word'] = 'test'
                    self.word['position'] = 1
                    self.word['fbank'] = []
                else:
                    if line.split()[-1] == ']':
                        tmp = line.split()[:-1]
                        tmp = np.array(tmp).astype(float)
                        fbank.append(tmp)
                    else:
                        tmp = line.split()
                        tmp = np.array(tmp).astype(float)
                        fbank.append(tmp)
            self.word['fbank'] = np.array(fbank).reshape(-1,40)
            print(self.word['fbank'])
            print(self.word['fbank'].shape)


    def write_to_file(self):
        save_path = os.path.join(self.word['utter_id'], \
                                 '_', self.word['channel'], \
                                 '_', self.word['position'], \
                                 '.txt')
        with open(save_path, 'w') as f:
            f.write('test')



if __name__ == '__main__':
    word = AlignWordFbank()
    word.split_fbank(FBANK_TXT_DIR)
    #word.write_to_file()

