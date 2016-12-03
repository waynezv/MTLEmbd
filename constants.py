import numpy as np

TASKS = ["word", "sem_similarity", "speaker_id", "gender", "age", "education", "dialect"]
FRAMES_PER_WORD = 100
FRAME_SIZE = 40

# Network architecture parameters
CONV1_FILTER_SIZE = (10,1,40,9)
CONV1_BIAS_SIZE = (10,)
CONV1_STRIDE = (2,2)

MAXPOOL_SHAPE = (3,3)
MAXPOOL_STRIDE = (1,1)
MAXPOOL_PAD = (1,1)

MEAN_KERNEL = (3,3)

CONV2_FILTER_SIZE = (5,10,20,5)
CONV2_BIAS_SIZE = (5,)

FORWARD1_FILTER_SIZE = (1024,720) #TODO:
FORWARD1_BIAS_SIZE = (1024,)

SHARED_REPRESENTATION_SIZE = 1024

# Training parameters
BATCH_SIZE = 20
LEARNING_RATE = 0.1

# Training miscellaneous
# DATA_PATH = ''.join('/mingback/715Proj/small_dataset_hard_align') # on server 252
DATA_PATH = ''.join('/home/caiwch/eesen/asr_egs/swbd/v1/fbank/fbank_txt/tmp') # on server 222
# DATA_PATH = ''.join('./data/small_dataset_hard_align') # on local
