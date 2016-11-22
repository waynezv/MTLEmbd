from __future__ import print_function
import sys
import os
import re
import argparse as agp

import constants
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.nnet import softmax
from theano.tensor.signal import pool

rng = np.random.RandomState(93492019)
caller_info_dic = dict() #a dictionary of conversation ids to callers [a,b]
DIM_TASKS = {"word": 100, "sem_similarity": 50, "gender": 2, "age": 1}


class CallerInfo:
    def __init__(self, userid, gender = 0, age = 0, education = 0, dialect = 0):
        self.userid = userid
        self.gender = gender #0 female 1 male
        self.age = age
        self.education = education
        self.dialect = dialect


def load_caller_info():
    callers = [s.strip().split(',') for s in open('caller_tab.csv')]
    education_dict = {}
    max_dialect = 0

    for caller in callers:
        call_id = int(caller[0])
        gender = caller[3]
        if gender == "FEMALE":
            gender = 0
        else:
            gender = 1
        age = 1997 - int(callers[4])
        dialect = callers[5]
        if dialect > max_dialect:
            max_dialect = dialect

        education = callers[6]
        if education in education_dict:
            education = education_dict[education]
        else:
            education_dict[education] = len(education_dict) # ???
        caller_info_dic[call_id] = CallerInfo(call_id, gender, age, education, dialect)

    DIM_TASKS["dialect"] = max_dialect
    DIM_TASKS["education"] = len(education_dict)
    DIM_TASKS["speaker_id"] = len(caller_info_dic)


class Instance:
    def __init__(self, filename, multitask_flag):
        """
        Representation of an instance for our model
        """
        self.input_file = filename  # Name of file containing data for the instance
        self.vec = []               # Vector representation for network input
        self.multitask_flag = 0     # Flag for whether to train on multiple tasks
        self.task_labels = dict()   # Labels for each task for the instance
        self.vec = self.read_vec()

    def read_vec():
        # TODO: Wenbo Read self.input_file, assign values to vec and task_labels
        with open(self.input_file, 'r') as f:
            word = f.readline()
            self.task_labels['word'] = word
            spk_id = f.readline()
            self.task_labels['speaker_id'] = spk_id
            fbank = np.array(f.readlines()).astype(float)
            assert fbank.shape[1] == 40, 'Wrong fbank dimension! \
                Expect 40, got {:d}'.format(fbank.shape[1])
            self.vec = fbank


class ConvolutionBuilder:
    def __init__(self, inp, filter_size, bias_size, stride, name):
        """
        Allocate a convolutional layer with shared variable internal parameters

        :type inp: theano.tensor.dtensor4
        :param inp: symbolic word filter bank representation tensor of shape
            (batch size, num input feature maps, input height, input width)

        :type filter_size: tuple or list of length 4
        :param filter_size: (number of filters, num input feature maps,
            filter height, filter width)

        :type bias_size: tuple or list of length 2
        :param bias_size: number of biases -- one bias per output feature map

        :type stride: tuple or list of length 2
        :param stride: (stride for height, stride for width)

        :type name: string
        :param name: name of the convolution
        """

        fan_in = np.prod(filter_size[1:])
        fan_out = (filter_size[0] * np.prod(filter_size[2:]))
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        self.W = theano.shared(np.asarray(
            rng.uniform(
                low = -W_bound,
                high = W_bound,
                size = filter_size
                ),
            dtype = inp.dtype), name = name + '_W', borrow=True)

        self.b = theano.shared(np.asarray(
            rng.uniform(
                low = -0.5,
                high = 0.5,
                size = bias_size
                ),
            dtype = inp.dtype), name = name + '_b', borrow=True)

        self.out = conv2d(input=inp, filters=self.W, subsample=stride)
        self.output = T.nnet.relu(self.out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.f = theano.function([inp], self.output)
        self.params = [self.W, self.b]


class Maxpool:
    def __init__(self, inp, shape, stride):
        self.output = pool.pool_2d(input=inp, ds=shape, st=stride, ignore_border=True)
        self.f = theano.function([inp], self.output)


class MeanSubtract:
    def __init__(self, inp, kernel_size):
        self.kernel_size = kernel_size
        self.filter_shape = (1, 1, self.kernel_size, self.kernel_size)
        self.filters = self.mean_filter(self.kernel_size).reshape(filter_shape)
        self.filters = theano.shared(_asarray(filters, dtype=floatX), borrow=True)

        self.mean = conv2d(input=inp, filters=filters, filter_shape=filter_shape,
                        border_mode='full')
        self.mid = int(floor(kernel_size/2.))
        self.output = inp - mean[:,:,mid:-mid,mid:-mid]
        self.f = theano.function([inp], self.output)

    def mean_filter(self):
        s = self.kernel_size**2
        x = repeat(1./s, s).reshape((self.kernel_size, self.kernel_size))
        return x


class ForwardLayer:
    def __init__(self, inp, filter_size, bias_size, name, activation=T.tanh):
        """
        Allocate a fully-connected feed forward layer with nonlinearity

        :type inp: theano.tensor.dmatrix
        :param inp: a symbolic tensor of shape (number of examples, input dimension)

        :type filter_size: tuple or list of length 2
        :param filter_size: (input dimension, output dimension)

        :type bias_size: tuple or list of length 2
        :param bias_size: size of bias

        :type name: string
        :param name: name of the feed forward layer
        """

        input_size = np.prod(filter_size)
        output_size = np.prod(bias_size)

        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (input_size + output_size)),
                high=numpy.sqrt(6. / (input_size + output_size)),
                size=filter_size
            ),
            dtype=theano.config.floatX
        )

        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4 # ???

        self.W = theano.shared(value=W_values, name=name+'_W', borrow=True)

        b_values = numpy.zeros(bias_size, dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name=name+'_b', borrow=True)

        self.output = activation(T.dot(inp, self.W) + self.b)
        self.f = theano.function([inp], self.output)
        self.params = [self.W, self.b]


class MultitaskNetwork:
    def __init__(self, batch_size, X, y, **kwargs):
        """
        :type batch_size:
        :para batch_size:

        :type X: ftensor4
        :para X: word feature of size (num_words, 1, number of fbanks, fbank dim 40)

        :type y: dict
        :para y: task specific labels ('taskname': labels)
        """
        self.multitask_flag = kwargs['multitask_flag']
        self.batch_size = batch_size

        # Reshape matrix of size (batch size, frames per word x frame size)
        # TODO: truncate or pad frames
        inp = X

        self.conv1 = ConvolutionBuilder(inp, constants.CONV1_FILTER_SIZE,
            constants.CONV1_BIAS_SIZE, constants.CONV1_STRIDE, 'conv1')
        self.maxpool = Maxpool(self.conv1.output, constants.MAXPOOL_SHAPE, constants.MAXPOOL_STRIDE)
        self.mean = MeanSubtract(self.maxpool.output, constants.MEAN_KERNEL)
        self.conv2 = ConvolutionBuilder(self.mean.output, constants.CONV1_FILTER_SIZE,
            constants.CONV1_FILTER_BOUND, constants.CONV1_BIAS_SIZE, 'conv2')
        self.forward = ForwardLayer(self.conv2.output.flatten(2), constants.FORWARD1_FILTER_SIZE,
            constants.FORWARD1_BIAS_SIZE, "forward")

        self.params = self.conv1.params + self.conv2.params + self.forward.params

        if  self.multitask_flag == 0:
            task = kwargs['task']
            self.task_specific_components = dict()
            self.task_specific_loss = dict()
            self.task_specific_grad = dict()
            self.task_specific_components[task] = ForwardLayer(self.forward.output,
                                                               (constants.SHARED_REPRESENTATION_SIZE, DIM_TASKS[task]), DIM_TASKS[task], task)
            self.params += self.task_specific_components[task].params

            # Loss and gradient
            loss = self.negative_log_likelihood(y[task], task)
            grad = T.grad(loss, self.params)

            self.task_specific_loss[task] = loss
            self.task_specific_grad[task] = grad

        else:
            self.task_specific_components = dict()
            self.task_specific_loss = dict()
            self.task_specific_grad = dict()
            for task in constants.TASKS:
                self.task_specific_components[task] = ForwardLayer(self.forward.output,
                    (constants.SHARED_REPRESENTATION_SIZE, DIM_TASKS[task]), DIM_TASKS[task], task)
                self.params += self.task_specific_components[task].params

                # Loss and gradient
                loss = self.negative_log_likelihood(y[task], task)
                grad = T.grad(loss, self.params)

                self.task_specific_loss[task] = loss
                self.task_specific_grad[task] = grad

    def negative_log_likelihood(self, y, task):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        p_y_given_x = T.nnet.softmax(self.task_specific_components[task].output)
        return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

    def mse(self, y, task):
        """
        Return the mean squared error of the prediction

        :type y: theano.tensor.TensorType
        :param y: corresponds to a matrix that gives for each example the
                  output to be matched
        """
        output = self.task_specific_components[task].output
        return T.mean((output - y)**2)



def test_network():
    # Model training code
    print('... training')

    # Prepare data
    task_flag = 0
    single_task = 'word'
    load_caller_info()
    TASK_DIM = DIM_TASKS[task]
    train_input = []
    train_label = []

    word_feat_filelist = [s.strip() for s in open('word_feat.filelist')]
    for i in xrange(TASK_DIM):
        fbank_file = os.path.join(constants.data_path, word_feat_filelist[i])
        instance = Instance(fbank_file, task_flag)
        word = instance.task_labels['word']
        spk_id = instance.task_labels['speaker_id']
        gender = caller_info_dic[spk_id].gender
        age = caller_info_dic[spk_id].age
        education = caller_info_dic[spk_id].education
        dialect = caller_info_dic[spk_id].dialect
        label = {'word':word, 'speaker_id':spk_id, 'gender':gender,
                    'age':age, 'education':education, 'dialect':dialect}

        train_input.append(instance.vec)
        train_label.append(label)

    train_input_shared = T.shared(np.asarray(train_input,
                                        dtype=T.config.floatX),
                            borrow=True)
    train_label_shared = T.cast(T.shared(np.asarray(train_label,
                                    dtype=T.config.floatX),
                            borrow=True),
                        'int32')

    # Allocate symbolic variables for data
    X = T.ftensor4('X')
    y = T.ivector('y') #TODO: y type not agree
    batch_index = T.lscalar()  # index to a [mini]batch

    #TODO: input size not equal
    network_input = X.reshape((constants.BATCH_SIZE, 1, constants.FRAMES_PER_WORD, constants.FRAME_SIZE))

    model = MultitaskNetwork(constants.BATCH_SIZE, network_input, multitask_flag=task_flag, task=single_task)

    updates = [(param_i, param_i - learning_rate * grad_i)
               for param_i, grad_i in zip(model.params, model.grad)]

    train_net = T.function(
        [batch_index],
        # model.task_specific_loss,
        model.negative_log_likelihood(y[single_task], single_task),
        updates = updates,
        givens = {
            X: train_input_shared[batch_index*model.batch_size : (batch_index+1)*model.batch_size],
            y: train_label_shared[batch_index*model.batch_size : (batch_index+1)*model.batch_size]
        }
    )

    num_train_batches = train_input_shared.get_value(borrow=True).shape[0] // model.batch_size
    for batch_ind in xrange(num_train_batches):
        loss = train_net(batch_ind)
        print(loss)
