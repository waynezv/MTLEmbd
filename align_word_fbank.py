from __future__ import print_function
import os
import csv
import argparse as agp
import numpy as np

FBANK_TXT_DIR = '/home/caiwch/eesen/asr_egs/swbd/v1/fbank/fbank_txt'

class AlignWordFbank(object):
    def __init__(self):
        self.word = {}

    def split_fbank(self, FBANK_TXT_DIR):
        with open(os.path.join(FBANK_TXT_DIR, 'raw_fbank_train.1.txt'), 'r') as f:
            line = f.readline()
            fbank_id = line.split(' ')[0]
            utt_id = line.split('_')[0]
            chn_id = utt_id.split('-')[1]
            tm_frame = [line.split('_')[1].split('-')[0], \
                        line.split('_')[1].split('-')[1]]
            self.word['word'] = 'test'
            self.word['fbank_id'] = fbank_id
            self.word['utter_id'] = utt_id
            self.word['channel'] = chn_id
            self.word['time_frame'] = tm_frame
            self.word['position'] = 1
            print(self.word)
            tmp = []
            for nxt_line in f:
				tmp.append(tmp)
                tmp = np.array(tmp)
			print(tmp)
			print(tmp.shape)
            #next_line = f.readline()
            #next_line = np.array(next_line)
            #print(next_line.shape)

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

