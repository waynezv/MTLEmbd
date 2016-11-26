from __future__ import print_function
import sys
import os
import re
import argparse as agp

import constants
import numpy as np

import theano
import theano.tensor as T
from theano import shared, config, _asarray, function
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
        age = 1997 - int(caller[4])
        dialect = callers[5]
        if dialect > max_dialect:
            max_dialect = dialect

        education = caller[6]
        if education in education_dict:
            education = education_dict[education]
        else:
            education_dict[education] = len(education_dict)
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
        self.multitask_flag = multitask_flag     # Flag for whether to train on multiple tasks
        self.task_labels = dict()   # Labels for each task for the instance
        self.vec = self.read_vec()

    def read_vec(self):
        with open(self.input_file, 'r') as f:
            word = f.readline()
            self.task_labels['word'] = word
            spk_id = int(f.readline().strip())
            self.task_labels['speaker_id'] = spk_id
            fbank = []
            for line in f:
                fbank.append(np.array(line.split()).astype(float))
            fbank = np.array(fbank)
            assert fbank.shape[1] == 40, 'Wrong fbank dimension! \
                Expect 40, got {:d}'.format(fbank.shape[1])
            return fbank


class ConvolutionBuilder:
    def __init__(self, inp, filter_size, bias_size, name, **kwargs):
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

        if 'stride' in kwargs:
            stride = kwargs['stride']
        else:
            stride = (1,1)
        #self.out = conv2d(input=inp, filters=self.W, subsample=stride, border_mode='full')
        self.out = conv2d(input=inp, filters=self.W, subsample=stride)
        self.output = T.nnet.relu(self.out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.f = theano.function([inp], self.output)
        self.params = [self.W, self.b]

        tmp = np.random.rand(20,1,200,40)
        t_inp = theano.tensor.dtensor4()
        out = conv2d(input=t_inp, filters=self.W, subsample=stride)
        #print(t_inp.shape.eval({t_inp: tmp}))
        print('conv:', theano.function([t_inp], out.shape)(tmp))


class Maxpool:
    def __init__(self, inp, shape, stride):
        self.output = pool.pool_2d(input=inp, ds=shape, st=stride, ignore_border=True)
        self.f = theano.function([inp], self.output)

        tmp = np.random.rand(20,1,25,33)
        t_inp = theano.tensor.dtensor4()
        out = pool.pool_2d(input=t_inp, ds=shape, st=stride, ignore_border=True)
        print(t_inp.shape.eval({t_inp: tmp}))
        print('max:', theano.function([t_inp], out.shape)(tmp))

class MeanSubtract:
    def __init__(self, inp, kernel_size):
        """
        :type kernel_size: tuple or list of length 2
        :para kernel_size:
        """
        self.kernel_size = kernel_size
        #inp_shape = (inp.shape[0], 1, inp.shape[1], inp.shape[2])
        #inp = inp.reshape(inp_shape).astype(theano.config.floatX)

        self.filter_shape = (1, 1, self.kernel_size[0], self.kernel_size[1])
        self.filters = self.mean_filter().reshape(self.filter_shape)
        #self.filters = theano.shared(_asarray(self.filters, dtype=theano.config.floatX),
        #        borrow=True)
        self.filters = theano.shared(np.asarray(self.filters,
                                            dtype='float64'),
                                borrow=True)

        self.mean = conv2d(input=inp, filters=self.filters,
                #input_shape=inp_shape,
                filter_shape=self.filter_shape,
                border_mode='full') # TODO: might have bug
        mid = int(np.floor(self.kernel_size[0]/2.))
        new_inp = inp - self.mean[:,:,mid:-mid,mid:-mid]

        self.output = new_inp
        self.f = theano.function([inp], self.output)

    def mean_filter(self):
        s = np.power(self.kernel_size[0], 2)
        x = np.repeat(1./s, s).reshape((self.kernel_size[0], self.kernel_size[1]))
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

        W_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (input_size + output_size)),
                high=np.sqrt(6. / (input_size + output_size)),
                size=filter_size
            ),
            dtype=theano.config.floatX
        )

        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4 # ???

        self.W = theano.shared(value=W_values, name=name+'_W', borrow=True)

        b_values = np.zeros(bias_size, dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name=name+'_b', borrow=True)

        self.output = activation(T.dot(inp, self.W.T) + self.b)
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
        inp = X

        self.conv1 = ConvolutionBuilder(inp, constants.CONV1_FILTER_SIZE,
            constants.CONV1_BIAS_SIZE, 'conv1', stride=constants.CONV1_STRIDE)
        self.maxpool = Maxpool(self.conv1.output, constants.MAXPOOL_SHAPE, constants.MAXPOOL_STRIDE)
        self.mean = MeanSubtract(self.maxpool.output, constants.MEAN_KERNEL)
        self.conv2 = ConvolutionBuilder(self.mean.output, constants.CONV1_FILTER_SIZE,
                constants.CONV1_BIAS_SIZE, 'conv2')
        self.forward = ForwardLayer(self.conv2.output.flatten(2), constants.FORWARD1_FILTER_SIZE,
            constants.FORWARD1_BIAS_SIZE, "forward")

        t_inp = theano.tensor.dtensor4()
        conv1 = ConvolutionBuilder(t_inp, constants.CONV1_FILTER_SIZE,
            constants.CONV1_BIAS_SIZE, 'conv1', stride=constants.CONV1_STRIDE)
        maxpool = Maxpool(conv1.output, constants.MAXPOOL_SHAPE, constants.MAXPOOL_STRIDE)
        mean = MeanSubtract(maxpool.output, constants.MEAN_KERNEL)
        conv2 = ConvolutionBuilder(mean.output, constants.CONV1_FILTER_SIZE,
                constants.CONV1_BIAS_SIZE, 'conv2')
        forward = ForwardLayer(conv2.output.flatten(2), constants.FORWARD1_FILTER_SIZE,
            constants.FORWARD1_BIAS_SIZE, "forward")
        tmp = np.random.rand(20,1,200,40)
        print(t_inp.shape.eval({t_inp: tmp}))
        print('conv1:', theano.function([t_inp], conv1.output.shape)(tmp))
        print('max:', theano.function([t_inp], maxpool.output.shape)(tmp))
        print('mean:', theano.function([t_inp], mean.output.shape)(tmp))
        print('conv2:', theano.function([t_inp], conv2.output.shape)(tmp))
        print('forward:', theano.function([t_inp], forward.output.shape)(tmp))

        self.params = self.forward.params + self.conv2.params + self.conv1.params
        if  self.multitask_flag == 0:
            task = kwargs['task']
            self.task_specific_components = dict()
            self.task_specific_loss = dict()
            self.task_specific_grad = dict()
            self.task_specific_components[task] = ForwardLayer(self.forward.output,
                    (1, constants.SHARED_REPRESENTATION_SIZE),
                    1, task)

            forward_word = ForwardLayer(forward.output,
                    (1, constants.SHARED_REPRESENTATION_SIZE),
                    1, task)
            print('forward_word:', theano.function([t_inp], forward_word.output.shape)(tmp))

            self.params = self.task_specific_components[task].params + self.params

            # Loss and gradient
            self.loss = self.negative_log_likelihood(y, task)
            self.grads = T.grad(cost=self.loss, wrt=self.params)

            self.task_specific_loss[task] = self.loss
            self.task_specific_grad[task] = self.grads


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
        #return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
        return -T.mean(T.log(p_y_given_x)[y])

    def mse(self, y, task):
        """
        Return the mean squared error of the prediction

        :type y: theano.tensor.TensorType
        :param y: corresponds to a matrix that gives for each example the
                  output to be matched
        """
        output = self.task_specific_components[task].output
        return T.mean((output - y)**2)


def load_word_dict(file='word_dict.txt'):
    word_dict = {}
    for w in open(file, 'r'):
        t = w.strip().split(',')
        word_dict[t[0]] =  int(t[1])
    return word_dict


def test_network():
    # Model training code
    print('... training')

    # Prepare data
    task_flag = 0
    single_task = 'word'
    load_caller_info()
    TASK_DIM = DIM_TASKS[single_task]
    train_input = []
    train_label = []

    word_dict = load_word_dict()
    word_feat_filelist = [s.strip() for s in open('word_feat.filelist')]
    for i in xrange(TASK_DIM):
        fbank_file = os.path.join(constants.DATA_PATH, word_feat_filelist[i])
        instance = Instance(fbank_file, task_flag)
        word = instance.task_labels['word'].strip()
        spk_id = instance.task_labels['speaker_id']
        gender = caller_info_dic[spk_id].gender
        age = caller_info_dic[spk_id].age
        education = caller_info_dic[spk_id].education
        dialect = caller_info_dic[spk_id].dialect
        label = {'word':word_dict[word], 'speaker_id':spk_id, 'gender':gender,
                    'age':age, 'education':education, 'dialect':dialect}

        # truncate or pad word feat to FRAMES_PER_WORD = 200
        tmp_vec = np.zeros((constants.FRAMES_PER_WORD, constants.FRAME_SIZE))
        vec = np.array(instance.vec)
        ins_vec_dim = vec.shape[0]
        if ins_vec_dim > constants.FRAMES_PER_WORD:
            ins_vec_dim = constants.FRAMES_PER_WORD
        tmp_vec[:ins_vec_dim,:] = vec[:ins_vec_dim,:]
        # TODO: Single task for now
        task_label = label[single_task]
        train_input.append(tmp_vec)
        train_label.append(task_label)


    train_input_shared = theano.shared(np.asarray(train_input,
                                        dtype=theano.config.floatX),
                            borrow=True)
    train_label_shared = T.cast(theano.shared(np.asarray(train_label,
                                    dtype=theano.config.floatX),
                            borrow=True),
                        'int32')

    # Allocate symbolic variables for data
    X = T.dtensor4('X')
    y = T.lvector('y') #TODO: y type not agree, make it single task for now
    batch_index = T.lscalar()  # index to a [mini]batch

    network_input = X.reshape((constants.BATCH_SIZE, 1, constants.FRAMES_PER_WORD, constants.FRAME_SIZE))
    network_label = y

    model = MultitaskNetwork(constants.BATCH_SIZE, network_input, network_label,
            multitask_flag=task_flag, task=single_task)

    updates = [(param_i, param_i - constants.LEARNING_RATE * grad_i)
            for param_i, grad_i in zip(model.params, model.task_specific_grad[single_task])]

# Single batch train
    train_net = theano.function(
            [network_input, network_label],
            model.loss,
            updates = updates
            )

    train_input = np.array(train_input).reshape((-1,1,constants.FRAMES_PER_WORD, constants.FRAME_SIZE))
    train_label = np.array(train_label)
    loss = train_net(train_input[:20,:,:,:], train_label[:20])
    print(loss)

#Batch train
    #train_net = theano.function(
    #    [batch_index],
        # model.task_specific_loss,
    #    model.loss,
    #    updates = updates,
    #    givens = {
    #        network_input: train_input_shared[batch_index*model.batch_size : (batch_index+1)*model.batch_size],
    #        network_label: train_label_shared[batch_index*model.batch_size : (batch_index+1)*model.batch_size]
    #    }
    #)

    #num_train_batches = train_input_shared.get_value(borrow=True).shape[0] // model.batch_size
    #for batch_ind in xrange(num_train_batches):
    #    loss = train_net(batch_ind)
    #    print(loss)

if __name__ == '__main__':
    test_network()
