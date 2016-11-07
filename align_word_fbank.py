from __future__ import print_function
import os
import csv
import re
import argparse as agp
import numpy as np

# FBANK_TXT_DIR = '/home/caiwch/eesen/asr_egs/swbd/v1/fbank/fbank_txt'
FBANK_TXT_DIR = '.'
SAVE_DIR = './tmp'

class AlignWordFbank(object):
    def __init__(self):
        self.word = {}
        self.conv_info = {}
        self.trans_info = []

    def load_conv_info(self):
        conv_info = [s.strip().split(',') for s in open('conv_tab.csv')]
        for conv in conv_info:
            conv_id = conv[0] # conversation id
            call_id = conv[2] # caller id of side A/channel A
            recv_id = conv[3] # receiver id of side B/channel B
            self.conv_info[conv_id] = {'A': call_id, 'B': recv_id}

    def load_word_trans(self):
        word_trans = [s for s in open('all_words_trans.txt')]
        pos = 0 # word position in utterance
        utt_id = 2 # begining utterance index
        for entry in word_trans:
            parts = entry.strip().split() # swxxxx(A|B)-ms98-a-xxxx x.xxxxxx x.xxxxxx word
            if parts[3] == '[silence]': # skip silence
                continue
            word = parts[3]
            start_time = float(parts[1])
            end_time = float(parts[2])
            p1 = parts[0].split('-')
            conv_id = p1[0][2:-1]
            utt_id_new = int(p1[3]) # the utterance index in conversation
            if utt_id_new == utt_id:
                pos += 1
            else:
                utt_id = utt_id_new
                pos = 1
            self.trans_info.append({'word': word, \
                                    'utt_id': utt_id, 'conv_id': conv_id, \
                                    'start_time': start_time, 'end_time': end_time, \
                                    'position': pos})

    def split_fbank(self):
        fbank_flist = [s.strip() for s in open('raw_fbank_train.list')]
        word_ind = 0 # index for word in self.trans_info
        for fbank_file in fbank_flist:
            print(fbank_file)
            with open(fbank_file, 'r') as f:
                pat1 = re.compile('.+\[\n')
                pat2 = re.compile('.+\]\n')
                fbank = []
                for line in f:
                    if pat1.match(line):
                        fbank_id = line.split('_')[0]
                        conv_id = fbank_id.split('-')[0].lstrip('sw0')
                        chn_id = fbank_id.split('-')[1]
                        self.word['conv_id'] = conv_id # 2001 - 4940
                        self.word['channel'] = chn_id # A, B
                        self.word['speaker_id'] = self.conv_info[conv_id][chn_id] # 1000 - 1703

                        self.word['word'] = self.trans_info[word_ind]['word']
                        self.word['utt_id'] = self.trans_info[word_ind]['utt_id']
                        self.word['position'] = self.trans_info[word_ind]['position']
                        st_t = self.trans_info[word_ind]['start_time']
                        nd_t = self.trans_info[word_ind]['end_time']
                        dur = nd_t - st_t
                        num_frames = np.ceil(dur/0.020)
                        frm_ind = 1
                        self.word['fbank'] = []
                    else:
                        if line.split()[-1] == ']':
                            tmp = line.split()[:-1]
                            tmp = np.array(tmp).astype(float)
                            fbank.append(tmp)
                            frm_ind += 1
                            if frm_ind >= num_frames:
                                self.word['fbank'] = np.array(fbank).reshape(-1,40)
                                self.save_word_fbank()
                                fbank = []
                                word_ind += 1
                                self.word['word'] = self.trans_info[word_ind]['word']
                                self.word['utt_id'] = self.trans_info[word_ind]['utt_id']
                                self.word['position'] = self.trans_info[word_ind]['position']
                                st_t = self.trans_info[word_ind]['start_time']
                                nd_t = self.trans_info[word_ind]['end_time']
                                dur = nd_t - st_t
                                num_frames = np.ceil(dur/0.020)
                                frm_ind = 1
                                self.word['fbank'] = []

                        else:
                            tmp = line.split()
                            tmp = np.array(tmp).astype(float)
                            fbank.append(tmp)
                            frm_ind += 1
                            if frm_ind >= num_frames:
                                self.word['fbank'] = np.array(fbank).reshape(-1,40)
                                self.save_word_fbank()
                                fbank = []
                                word_ind += 1
                                self.word['word'] = self.trans_info[word_ind]['word']
                                self.word['utt_id'] = self.trans_info[word_ind]['utt_id']
                                self.word['position'] = self.trans_info[word_ind]['position']
                                st_t = self.trans_info[word_ind]['start_time']
                                nd_t = self.trans_info[word_ind]['end_time']
                                dur = nd_t - st_t
                                num_frames = np.ceil(dur/0.020)
                                frm_ind = 1
                                self.word['fbank'] = []

    def save_word_fbank(self):
        save_f = os.path.join(SAVE_DIR, ''.join(str(self.word['conv_id']) \
                                 + '_' + str(self.word['channel']) \
                                 + '_' + str(self.word['utt_id']) \
                                 + '_' + str(self.word['position']) \
                                 + '.txt'))
        print(save_f)
        out = ""
        with open(save_f, 'w') as f:
            out = self.word['word'] + '\n' \
                + self.word['speaker_id'] + '\n'
            for fline in self.word['fbank']:
                for val in fline:
                    out = out + str(val) + ' '
                out = out + '\n'
            f.write(out)
        print('Saved ', save_f)

if __name__ == '__main__':
    word = AlignWordFbank()
    word.load_conv_info()
    word.load_word_trans()
    word.split_fbank()
