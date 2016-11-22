from __future__ import print_function
import os
import csv
import re
import argparse as agp
import numpy as np

# FBANK_TXT_DIR = '/home/caiwch/eesen/asr_egs/swbd/v1/fbank/fbank_txt'
FBANK_TXT_DIR = '.'
SAVE_DIR = './tmp2_11_19'

T_WINDOW = 0.02 # sec, NOTE: not sure
T_WINDOW_OVERLAP = 0.01 # sec
SAMPLE_FREQ = 8000 # Hz

class AlignWordFbank(object):
    def __init__(self):
        self.word = {}
        self.conv_info = {}
        self.trans_info = []
        self.fnum_frame = [] # list of num. frames in each utterance for one fbank file
        self.anum_word = [] # list of num. words in each utterance for all fbank files
        self.word_dict = {}

    def load_conv_info(self):
        conv_info = [s.strip().split(',') for s in open('conv_tab.csv')]
        for conv in conv_info:
            conv_id = conv[0] # conversation id
            call_id = conv[2] # caller id of side A/channel A
            recv_id = conv[3] # receiver id of side B/channel B
            self.conv_info[conv_id] = {'A': call_id, 'B': recv_id}

    def load_word_trans(self):
        """
        Load word info. and build word dictionry.
        """
        word_trans = [s for s in open('all_words_trans.txt')]
        pos = 0 # word position in utterance
        utt_id = 2 # begining utterance index
        word_cnt = 0 # word counter
        for entry in word_trans:
            parts = entry.strip().split() # swxxxx(A|B)-ms98-a-xxxx x.xxxxxx x.xxxxxx word
            if parts[3] == '[silence]': # skip silence
                continue
            word = parts[3]
            if word not in self.word_dict:
                self.word_dict[word] = len(self.word_dict)
            start_time = float(parts[1])
            end_time = float(parts[2])
            p1 = parts[0].split('-')
            conv_id = p1[0][2:-1]
            utt_id_new = int(p1[3]) # the utterance index in conversation
            if utt_id_new == utt_id:
                pos += 1
            else:
                utt_id = utt_id_new
                word_cnt = pos
                self.anum_word.append(word_cnt)
                pos = 1
            self.trans_info.append({'word': word, \
                                    'utt_id': utt_id, 'conv_id': conv_id, \
                                    'start_time': start_time, 'end_time': end_time, \
                                    'position': pos})
# Save word dictionary
        out = ""
        with open('word_dict.txt', 'w') as f:
            for word in self.word_dict:
                out += word + ',' + str(self.word_dict[word]) + '\n'
            f.write(out)

    def _split_list(self, l, n):
        # Split a list l to almostly even n parts
        return np.array_split(l,n)

    def pair_word_fbank(self):
        fbank_flist = [s.strip() for s in open('raw_fbank_train_1.list')]
        word_ind = 0 # index for word in self.trans_info
        utt_ind = 0 # index for utt in self.anum_word
        for fbank_file in fbank_flist:
            # First find how many frames in one fbank file
            tot_frm = 0
            with open(fbank_file, 'r') as f:
                for line in f:
                    tot_frm += 1
                    if line.split()[-1] == ']': # end of utt
                        self.fnum_frame.append(tot_frm-1)
                        tot_frm = 0

            frm_ind = 0 # index of self.fnum_frame
            # Split fbanks to match words
            with open(fbank_file, 'r') as f:
                pat1 = re.compile('.+\[\n') # begin
                pat2 = re.compile('.+\]\n') # end
                for line in f:
                    if pat1.match(line):
                        num_frames = self.fnum_frame[frm_ind] # num. frames in one utt
                        num_words = self.anum_word[utt_ind] # num. words in one utt
                        seglist = self._split_list(xrange(1, num_frames+1), num_words)

                        fbank_id = line.split('_')[0]
                        conv_id = fbank_id.split('-')[0].lstrip('sw0')
                        chn_id = fbank_id.split('-')[1]
                        self.word['conv_id'] = conv_id # 2001 - 4940
                        self.word['channel'] = chn_id # A, B
                        self.word['speaker_id'] = self.conv_info[conv_id][chn_id] # 1000 - 1703
                        self.word['word'] = self.trans_info[word_ind]['word']
                        self.word['utt_id'] = self.trans_info[word_ind]['utt_id']
                        self.word['position'] = self.trans_info[word_ind]['position']
                        self.word['fbank'] = []

                        fbank = []
                        frm_cnt = 0
                        wrd_cnt = 0

                        frm_ind += 1
                        utt_ind += 1

                    else:
                        if line.split()[-1] == ']':
                            tmp = line.split()[:-1]
                        else:
                            tmp = line.split()
                        tmp = np.array(tmp).astype(float)
                        fbank.append(tmp)
                        frm_cnt += 1
                        print('num_words', num_words, ' wrd_cnt ', wrd_cnt, \
										'seglistlastind', seglist[wrd_cnt][-1])
                        if frm_cnt == seglist[wrd_cnt][-1]: # last ind for this word
                            self.word['fbank'] = np.array(fbank).reshape(-1,40)
                            self.save_word_fbank()

                            fbank = []
                            wrd_cnt += 1
                            word_ind += 1

                            self.word['word'] = self.trans_info[word_ind]['word']
                            self.word['utt_id'] = self.trans_info[word_ind]['utt_id']
                            self.word['position'] = self.trans_info[word_ind]['position']
                            self.word['fbank'] = []

    def split_fbank(self):
        fbank_flist = [s.strip() for s in open('raw_fbank_train.list')]
        word_ind = 0 # index for word in self.trans_info
        for fbank_file in fbank_flist:
            # First find how many frames in one fbank file
            ftot_frm = []
            tot_frm = 0
            with open(fbank_file, 'r') as f:
                for line in f:
                    tot_frm += 1
                    if line.split()[-1] == ']':
                        ftot_frm.append(tot_frm-1)
                        tot_frm = 0

            with open(fbank_file, 'r') as f:
                pat1 = re.compile('.+\[\n') # begin
                pat2 = re.compile('.+\]\n') # end
                ftot_frm_ind = 0
                for line in f:
                    if pat1.match(line):
                        fbank_id = line.split('_')[0]
                        utt_tfrm = line.split('_')[1].split()[0]
                        utt_st_t = int(utt_tfrm.split('-')[0]) / 100.0
                        utt_nd_t = int(utt_tfrm.split('-')[1]) / 100.0
                        utt_len = utt_nd_t - utt_st_t
                        conv_id = fbank_id.split('-')[0].lstrip('sw0')
                        chn_id = fbank_id.split('-')[1]
                        self.word['conv_id'] = conv_id # 2001 - 4940
                        self.word['channel'] = chn_id # A, B
                        self.word['speaker_id'] = self.conv_info[conv_id][chn_id] # 1000 - 1703
                        self.word['word'] = self.trans_info[word_ind]['word']
                        self.word['utt_id'] = self.trans_info[word_ind]['utt_id']
                        self.word['position'] = self.trans_info[word_ind]['position']
                        w_st_t = self.trans_info[word_ind]['start_time']
                        w_nd_t = self.trans_info[word_ind]['end_time']
                        w_len = w_nd_t - w_st_t
                        num_frames = ((w_len-T_WINDOW_OVERLAP) \
                                           / (utt_len-T_WINDOW_OVERLAP)) \
                                         * ftot_frm[ftot_frm_ind]
                        self.word['fbank'] = []

                        frm_cnt = 0
                        fbank = []

                        ftot_frm_ind += 1

                    else:
                        if line.split()[-1] == ']':
                            tmp = line.split()[:-1]
                        else:
                            tmp = line.split()
                        tmp = np.array(tmp).astype(float)
                        fbank.append(tmp)
                        frm_cnt += 1
                        if frm_cnt >= num_frames:
                            self.word['fbank'] = np.array(fbank).reshape(-1,40)
                            self.save_word_fbank()

                            frm_cnt = 0
                            word_ind += 1
                            fbank = []

                            self.word['word'] = self.trans_info[word_ind]['word']
                            self.word['utt_id'] = self.trans_info[word_ind]['utt_id']
                            self.word['position'] = self.trans_info[word_ind]['position']
                            w_st_t = self.trans_info[word_ind]['start_time']
                            w_nd_t = self.trans_info[word_ind]['end_time']
                            w_len = w_nd_t - w_st_t
                            num_frames = ((w_len-T_WINDOW_OVERLAP) \
                                           / (utt_len-T_WINDOW_OVERLAP)) \
                                         * ftot_frm[ftot_frm_ind]
                            self.word['fbank'] = []

    def save_word_fbank(self):
        save_f = os.path.join(SAVE_DIR, ''.join(str(self.word['conv_id']) \
                                 + '_' + str(self.word['channel']) \
                                 + '_' + str(self.word['utt_id']) \
                                 + '_' + str(self.word['position']) \
                                 + '.txt'))
        out = ""
        with open(save_f, 'w') as f:
            out = self.word['word'] + '\n' \
                + self.word['speaker_id'] + '\n'
            for fline in self.word['fbank']:
                for val in fline:
                    out = out + str(val) + ' '
                out = out + '\n'
            f.write(out)

        print(save_f)

if __name__ == '__main__':
    word = AlignWordFbank()
    word.load_conv_info()
    word.load_word_trans()
    # word.split_fbank()
#    word.pair_word_fbank()
