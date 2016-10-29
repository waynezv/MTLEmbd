import constants
import numpy as np
import theano
import theano.tensor as T

class Instance:
    def __init__(self, filename, multitask_flag):
        self.input_file = filename  # Name of file containing data for the instance
        self.vec = []               # Vector representation for network input
        self.multitask_flag = 0     # Flag for whether to train on multiple tasks 
        self.task_labels = dict()   # Labels for each task for the instance

    def read_vec():
        # TODO: Read self.input_file, assign values to vec and task_labels

class MultitaskNetwork:
    def __init__(self, config):
        self.config = config
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]

        
