import constants
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import sys

rng = np.random.RandomState(93492019)
inp = T.tensor4(name = 'input')

class CallerInfo:
    def __init__(self, userid, gender = 0, age = 0, education = 0, dialect = 0):
        self.userid = userid
        self.gender = gender #0 female 1 male
        self.age = age
        self.education = education
        self.dialect = dialect

caller_info_dic = dict() #a dictionary of conversation ids to callers [a,b]

def load_caller_info():
    callers = [s.strip().split(',') for s in open('caller_tab.csv')]
    education_dict = {}

    for caller in callers:
        call_id = int(caller[0])
        gender = caller[3]
        if gender == "FEMALE":
            gender = 0
        else:
            gender = 1
        age = 1997 - int(callers[4])
        dialect = callers[5]
        education = callers[6]
        if education in education_dict:
            education = education_dict[education]
        else:
            education_dict[education] = len(education_dict) # ???
        caller_info_dic[call_id] = CallerInfo(userid, gender, age, education, dialect)


class Instance:
    def __init__(self, filename, multitask_flag):
        self.input_file = filename  # Name of file containing data for the instance
        self.vec = []               # Vector representation for network input
        self.multitask_flag = 0     # Flag for whether to train on multiple tasks
        self.task_labels = dict()   # Labels for each task for the instance

    def read_vec():
        # TODO: Wenbo Read self.input_file, assign values to vec and task_labels
        pass

class ConvolutionBuilder:
    def __init__(self, param_size, param_bound, bias_size, name):
        self.W = theano.shared(np.asarray(
            rng.uniform(
                low = -1.0 / param_bound,
                high = 1 / param_bound,
                size = param_size
                ),
            dtype = inp.dtype), name = name + 'W')

        self.b = theano.shared(np.asarray(
            rng.uniform(
                low = -0.5,
                high = 0.5,
                size = bias_size
                ),
            dtype = inp.dtype), name = name + 'b')

        self.out = conv2d(inp, self.W, subsample = (9,1))

        self.output = T.nnet.relu(self.out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.f = theano.function([inp], self.output)


class Maxpool:
    def __init__(self, shape, stride):
        self.output = pool.pool_2d(inp, shape, st = stride, ignore_border = True)
        self.f = theano.function([inp], self.output)

class MeanSubtract:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.f = theano.function([inp], )
        self.filter_shape = (1, 1, self.kernel_size, self.kernel_size)
        self.filters = mean_filter(self.kernel_size).reshape(filter_shape)
        self.filters = shared(_asarray(filters, dtype=floatX), borrow=True)

        self.mean = conv2d(inp, filters=filters, filter_shape=filter_shape,
                        border_mode='full')
        self.mid = int(floor(kernel_size/2.))
        self.output = inp - mean[:,:,mid:-mid,mid:-mid]
        self.f = theano.function([inp], self.output)


    def mean_filter(self):
        s = self.kernel_size**2
        x = repeat(1./s, s).reshape((self.kernel_size, self.kernel_size))
        return x

class ForwardLayer:
    def __init__(self, param_size, bias_size):
        self.W =

class MultitaskNetwork:
    def __init__(self, config):
        self.config = config
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]

        # TODO: Read self.input_file, assign values to vec and task_labels
        self.conv1 = ConvolutionBuilder(constants.CONV1_PARAM_SIZE, constants.CONV1_PARAM_BOUND, constants.CONV1_BIAS_SIZE, 'conv1')
        self.maxpool = Maxpool(constants.MAXPOOL_SHAPE, constants.MAXPOOL_STRIDE)
        self.mean = MeanSubtract(constants.MEAN_KERNEL)
        self.conv2 = ConvolutionBuilder(constants.CONV1_PARAM_SIZE, constants.CONV1_PARAM_BOUND, constants.CONV1_BIAS_SIZE, 'conv2')
        self.forward = ForwardLayer(FORWARD1_SIZE, )
