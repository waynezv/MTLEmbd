from __future__ import print_function
import sys
import os
import re
import argparse as agp
import six.moves.cPickle as pickle
import numpy as np
import theano
import theano.tensor as T
from theano import shared, config, _asarray, function
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.nnet import softmax
from theano.tensor.signal import pool

import constants

rng = np.random.RandomState(93492019)
caller_info_dic = dict() # dictionary of callers id to callers info
DIM_TASKS = {"word": 100, "sem_similarity": 50, "gender": 2, "age": 1} # dimension of tasks

class CallerInfo:
    def __init__(self, userid, gender = 0, age = 0, education = 0, dialect = 0):
        self.userid = userid
        self.gender = gender
        self.age = age
        self.education = education
        self.dialect = dialect


def load_caller_info():
    callers = [s.strip().split(',') for s in open('caller_tab.csv')]
    education_dict = dict()
    dialect_dict = dict()

    for caller in callers:
        call_id = int(caller[0])
        gender = caller[3]
        if gender == "FEMALE":
            gender = 0
        else:
            gender = 1
        age = 1997 - int(caller[4])
        dialect = str(caller[5])
        if not dialect:
            dialect = 'UNK'
        if dialect in dialect_dict:
            dialect = dialect_dict[dialect]
        else:
            dialect_dict[dialect] = len(dialect_dict)
        education = int(caller[6])
        if education in education_dict:
            education = education_dict[education]
        else:
            education_dict[education] = len(education_dict)
        caller_info_dic[call_id] = CallerInfo(call_id, gender, age, education, dialect)

    DIM_TASKS["dialect"] = len(dialect_dict)
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
        self.out = conv2d(input=inp, filters=self.W, subsample=stride)
        self.output = T.nnet.relu(self.out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.f = theano.function([inp], self.output)
        self.params = [self.W, self.b]


class Maxpool:
    def __init__(self, inp, shape, stride, pad):
        self.output = pool.pool_2d(input=inp, ds=shape, st=stride,
                ignore_border=True, padding=pad)
        self.f = theano.function([inp], self.output)


class MeanSubtract:
    def __init__(self, inp, kernel_size):
        """
        :type kernel_size: tuple or list of length 2
        :para kernel_size:
        """
        self.kernel_size = kernel_size
        self.filter_shape = (10, 10, self.kernel_size[0], self.kernel_size[1])
        self.filters = self.mean_filter().reshape(self.filter_shape)
        self.filters = theano.shared(np.asarray(self.filters,
                                            dtype=inp.dtype),
                                borrow=True)

        self.mean = conv2d(input=inp, filters=self.filters,
                filter_shape=self.filter_shape,
                border_mode='half')
        # TODO: problematic
        #mid = int(np.floor(self.kernel_size[0]/2.))
        #new_inp = inp - self.mean[:,:,mid:-mid,mid:-mid]
        new_inp = inp - self.mean

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
        :param filter_size: (output dimension, input dimension)

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
            dtype=inp.dtype
        )

        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(value=W_values, name=name+'_W', borrow=True)

        b_values = np.zeros(bias_size, dtype=inp.dtype)
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

        inp = X

        self.conv1 = ConvolutionBuilder(inp, constants.CONV1_FILTER_SIZE,
            constants.CONV1_BIAS_SIZE, 'conv1', stride=constants.CONV1_STRIDE)
        self.maxpool = Maxpool(self.conv1.output, constants.MAXPOOL_SHAPE,
                constants.MAXPOOL_STRIDE, constants.MAXPOOL_PAD)
        #self.mean = MeanSubtract(self.maxpool.output, constants.MEAN_KERNEL)
        self.conv2 = ConvolutionBuilder(self.maxpool.output, constants.CONV2_FILTER_SIZE,
                constants.CONV2_BIAS_SIZE, 'conv2')
        self.flattened = self.conv2.output.flatten(2)
        self.forward = ForwardLayer(self.flattened,
                constants.FORWARD1_FILTER_SIZE,
                constants.FORWARD1_BIAS_SIZE, "forward")

#IF DEBUG
        t_inp = theano.tensor.dtensor4()
        conv1 = ConvolutionBuilder(t_inp, constants.CONV1_FILTER_SIZE,
            constants.CONV1_BIAS_SIZE, 'conv1', stride=constants.CONV1_STRIDE)
        maxpool = Maxpool(conv1.output, constants.MAXPOOL_SHAPE,
                constants.MAXPOOL_STRIDE, constants.MAXPOOL_PAD)
        #mean = MeanSubtract(maxpool.output, constants.MEAN_KERNEL)
        conv2 = ConvolutionBuilder(maxpool.output, constants.CONV2_FILTER_SIZE,
                constants.CONV2_BIAS_SIZE, 'conv2')
        flattened = conv2.output.flatten(2)
        forward = ForwardLayer(flattened,
                constants.FORWARD1_FILTER_SIZE,
                constants.FORWARD1_BIAS_SIZE, "forward")
        tmp = np.random.rand(constants.BATCH_SIZE,1,100,40)
        print('input:', t_inp.shape.eval({t_inp: tmp}))
        print('conv1:', theano.function([t_inp], conv1.output.shape)(tmp))
        print('max:', theano.function([t_inp], maxpool.output.shape)(tmp))
        #print('mean:', theano.function([t_inp], mean.output.shape)(tmp))
        print('conv2:', theano.function([t_inp], conv2.output.shape)(tmp))
        print('flat:', theano.function([t_inp], flattened.shape)(tmp))
        print('forward:', theano.function([t_inp], forward.output.shape)(tmp))
#ENDIF

        self.params = self.forward.params + self.conv2.params + self.conv1.params
        if  self.multitask_flag == 0: # single task
            task = kwargs['task']
            self.task_specific_components = dict()
            self.task_specific_loss = dict()
            self.task_specific_grad = dict()
            self.task_specific_components[task] = ForwardLayer(self.forward.output,
                    (DIM_TASKS[task], constants.SHARED_REPRESENTATION_SIZE),
                    DIM_TASKS[task], task)
            self.params = self.task_specific_components[task].params + self.params

#IF DEBUG
            forward_word = ForwardLayer(forward.output,
                    (DIM_TASKS[task], constants.SHARED_REPRESENTATION_SIZE),
                    DIM_TASKS[task], task)
            print('forward_word:', theano.function([t_inp], forward_word.output.shape)(tmp))
#ENDIF

            # Loss and gradient
            if task in ['word', 'speaker_id', 'gender', 'education', 'dialect']:
                self.p_y_given_x = T.nnet.softmax(self.task_specific_components[task].output)
                self.loss = self.negative_log_likelihood(y, task)
                #self.y_pred = T.argmax(self.p_y_given_x, axis=1)
                #self.predict_error = T.mean(T.neq(self.y_pred, y))
            else:
                self.loss = self.mse(y, task)

            self.grads = T.grad(cost=self.loss, wrt=self.params)
            self.task_specific_loss[task] = self.loss
            self.task_specific_grad[task] = self.grads


        else: # multitask
            self.task_specific_components = dict()
            self.task_specific_loss = dict()
            self.task_specific_grad = dict()
            for task in constants.TASKS:
                self.task_specific_components[task] = ForwardLayer(self.forward.output,
                    (DIM_TASKS[task], constants.SHARED_REPRESENTATION_SIZE),
                    DIM_TASKS[task], task)
                self.params += self.task_specific_components[task].params

                # Loss and gradient
                if task in ['word', 'speaker_id', 'gender', 'education', 'dialect']:
                    loss = self.negative_log_likelihood(y[task], task)
                else:
                    loss = self.mse(y[task], task)
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
        #return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        Lp = T.log(self.p_y_given_x)
        return -T.mean([ Lp[i, y[i]]
            for i in xrange(constants.BATCH_SIZE) ])

    def mse(self, y, task):
        """
        Return the mean squared error of the prediction

        :type y: theano.tensor.TensorType
        :param y: corresponds to a matrix that gives for each example the
                  output to be matched
        """
        output = self.task_specific_components[task].output
        return T.mean(np.power(output.reshape((constants.BATCH_SIZE,))
            - y.reshape((constants.BATCH_SIZE,)) ,2))


def load_word_dict(file='word_dict.txt'):
    word_dict = {}
    for w in open(file, 'r'):
        t = w.strip().split(',')
        word_dict[t[0]] =  int(t[1])
    return word_dict


def test_network():
    # Prepare data
    task_flag = 0
    #single_task = 'word'
    # single_task = 'sem_similarity'
    #single_task = 'speaker_id'
    #single_task = 'gender'
    #single_task = 'age'
    single_task = 'education'
    #single_task = 'dialect'
    load_caller_info()
    tot_input = []
    tot_label = []
    train_input = []
    train_label = []
    val_input = []
    val_label = []
    test_input = []
    test_label = []

    word_dict = load_word_dict()
    DIM_TASKS['word'] = len(word_dict)
    word_feat_filelist = [s.strip() for s in open('word_feat.filelist')]
    TOT_DATA_SIZE = len(word_feat_filelist)
    TRAIN_SIZE = int(np.floor(TOT_DATA_SIZE * 0.8))
    VAL_SIZE = int(np.floor(TOT_DATA_SIZE * 0.1))
    TEST_SIZE = int(TOT_DATA_SIZE - TRAIN_SIZE - VAL_SIZE)
    SHUFFLE_LIST = np.random.permutation(xrange(TOT_DATA_SIZE))
    TRAIN_LIST = SHUFFLE_LIST[: TRAIN_SIZE]
    VAL_LIST = SHUFFLE_LIST[TRAIN_SIZE : TRAIN_SIZE+VAL_SIZE]
    TEST_LIST = SHUFFLE_LIST[TRAIN_SIZE+VAL_SIZE : TOT_DATA_SIZE]
    for i in xrange(TOT_DATA_SIZE):
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

        # Truncate or pad word feat to FRAMES_PER_WORD
        tmp_vec = np.zeros((constants.FRAMES_PER_WORD, constants.FRAME_SIZE))
        vec = np.array(instance.vec)
        ins_vec_dim = vec.shape[0]
        if ins_vec_dim > constants.FRAMES_PER_WORD:
            ins_vec_dim = constants.FRAMES_PER_WORD
        tmp_vec[:ins_vec_dim,:] = vec[:ins_vec_dim,:]

        # TODO: Single task for now
        task_label = label[single_task]
        tot_input.append(tmp_vec)
        tot_label.append(task_label)

    tot_input = np.array(tot_input).reshape((-1, 1,
            constants.FRAMES_PER_WORD, constants.FRAME_SIZE))
    tot_label = np.array(tot_label).reshape(-1)
    train_input = tot_input[TRAIN_LIST]
    train_label = tot_label[TRAIN_LIST]
    val_input = tot_input[VAL_LIST]
    val_label = tot_label[VAL_LIST]
    test_input = tot_input[TEST_LIST]
    test_label = tot_label[TEST_LIST]
    print(tot_label.shape)
    print(train_label.shape)
    print(val_label.shape)
    print(test_label.shape)
    print(test_label)

    # Define network
    # Allocate symbolic variables for data
    network_input = T.tensor4('X', dtype=theano.config.floatX)
    network_label = T.ivector('y')
    network_input = network_input.reshape((constants.BATCH_SIZE, 1, constants.FRAMES_PER_WORD, constants.FRAME_SIZE))
    network_label = network_label

    # Model
    model = MultitaskNetwork(constants.BATCH_SIZE, network_input, network_label,
            multitask_flag=task_flag, task=single_task)

    # Gradients
    updates = [(param_i, param_i - constants.LEARNING_RATE * grad_i)
            for param_i, grad_i in zip(model.params, model.task_specific_grad[single_task])]

    # Objectives
    train_net = theano.function(
        inputs = [network_input, network_label],
        outputs = model.loss,
        updates = updates,
        allow_input_downcast=True
    )

    validate_net = theano.function(
        inputs = [network_input, network_label],
        outputs = model.loss,
        allow_input_downcast=True
    )

    test_net = theano.function(
        inputs = [network_input, network_label],
        outputs = model.loss,
        allow_input_downcast=True
    )

    #Batch train
    print('... training ...')
    num_train_batches = train_input.shape[0] // constants.BATCH_SIZE
    num_val_batches = val_input.shape[0] // constants.BATCH_SIZE
    num_test_batches = test_input.shape[0] // constants.BATCH_SIZE
    num_epochs = 1000
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(num_train_batches, patience//2)
    best_val_loss = np.inf
    test_score = 0.
    done_looping = False
    epoch = 0
    while (epoch < num_epochs) and (not done_looping):
        epoch += 1
        for tb_ind in xrange(num_train_batches):
            train_loss = train_net(
                train_input[tb_ind*constants.BATCH_SIZE : (tb_ind+1)*constants.BATCH_SIZE],
                train_label[tb_ind*constants.BATCH_SIZE : (tb_ind+1)*constants.BATCH_SIZE]
            )
            iter = (epoch - 1) * num_train_batches + tb_ind
            #if (iter + 1) % validation_frequency == 0: # validation
            if (iter + 1) % 1000 == 0: # validation
                val_losses = [validate_net(
                    val_input[vb_ind*constants.BATCH_SIZE : (vb_ind+1)*constants.BATCH_SIZE],
                    val_label[vb_ind*constants.BATCH_SIZE : (vb_ind+1)*constants.BATCH_SIZE]
                ) for vb_ind in xrange(num_val_batches)]
                this_val_loss = np.mean(val_losses)
                print(
                        'epoch %i, minibatch %i/%i,loss %f, validation error %f %%' %
                    (
                        epoch,
                        tb_ind + 1,
                        num_train_batches,
                        train_loss,
                        this_val_loss * 100.
                    )
                )

                if this_val_loss < best_val_loss:
                    # improve patience if loss improvement is good enough
                    #if this_val_loss < best_val_loss * improvement_threshold:
                    #    patience = max(patience, iter * patience_increase)
                    best_val_loss = this_val_loss

                    # Test
                    test_losses = [test_net(
                        test_input[i*constants.BATCH_SIZE : (i+1)*constants.BATCH_SIZE],
                        test_label[i*constants.BATCH_SIZE : (i+1)*constants.BATCH_SIZE]
                    ) for i in xrange(num_test_batches)]
                    test_score = np.mean(test_losses)
                    print(
                        ( '     epoch %i, minibatch %i/%i, test error of'
                         ' best model %f %%'
                        ) %
                        (
                            epoch,
                            tb_ind + 1,
                            num_train_batches,
                            test_score * 100.
                         )
                    )

                    # Save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(model, f)

            #if patience <= iter:
            #    done_looping = True
            #    break

    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        ) %
        (best_val_loss * 100., test_score * 100.)
    )

    # TODO: Multitask train


if __name__ == '__main__':
    test_network()
