import numpy as np

TASKS = ["word", "speaker_id", "gender", "age", "education", "dialect"]
FRAMES_PER_WORD = 200
FRAME_SIZE = 40

# Network architecture parameters
CONV1_FILTER_SIZE = (1,1,8,8)
CONV1_BIAS_SIZE = (1,)
CONV1_STRIDE = (8,1)

MAXPOOL_SHAPE = (4,4)
MAXPOOL_STRIDE = (2,2)

MEAN_KERNEL = (3,3)

CONV2_FILTER_SIZE = (1,1,8,8)
CONV2_BIAS_SIZE = (8,)

FORWARD1_FILTER_SIZE = (1024, 1024)
FORWARD1_BIAS_SIZE = (1024,)

# Training parameters
BATCH_SIZE = 20
IMPROVEMENT_THRESHOLD = 0.995
VALIDATION_FREQUENCY = 10
