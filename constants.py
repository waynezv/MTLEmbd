import numpy as np

TASKS = ["word", "speaker_id", "gender", "age", "education", "dialect"]
FRAMES_PER_WORD = 200

CONV1_PARAM_SIZE = (1, 1, 8, 8)

CONV1_PARAM_BOUND = np.sqrt(1*1*64)
CONV1_BIAS_SIZE = (8,)

MAXPOOL_SHAPE = (4,4)
MAXPOOL_STRIDE = (2,2)

MEAN_KERNEL = (3,3)

CONV2_PARAM_SIZE = (1, 1, 8, 8)

CONV2_PARAM_BOUND = np.sqrt(1*1*64)
CONV2_BIAS_SIZE = (8,)

FORWARD1_PARAM_SIZE = (1024, 1024)
FORWARD1_BIAS_SIZE = (1024)
