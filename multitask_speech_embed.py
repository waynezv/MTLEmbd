import constants
import numpy as np
import theano
import theano.tensor as T
import sys


class CallerInfo:
	def __init__(self, userid, gender = 0, age = 0, education = 0, dialect = 0):
		self.userid = userid
		self.gender = gender #0 female 1 male
		self.age = age
		self.education = education
		self.dialect = dialect

caller_info_dic = dict() #a dictionary of conversation ids to callers [a,b]

def load_caller_info():
	callers = [s.strip().split(',') for s in open('conv_tab.csv')]
	education_dict = {}

	for caller in callers:
		call_id = int(caller[0])
		gender = caller[3]
		if gender == "Female":
			gender = 0
		else:
			gender = 1
		age = 1997 - int(callers[4])
		dialect = callers[5]
		education = callers[6]
		if education in education_dict:
			education = education_dict[education]
		else:
			education_dict[education] = len(education_dict)
		caller_info_dic[call_id] = CallerInfo(userid, gender, age, education, dialect)


class Instance:
    def __init__(self, filename, multitask_flag):
        self.input_file = filename  # Name of file containing data for the instance
        self.vec = []               # Vector representation for network input
        self.multitask_flag = 0     # Flag for whether to train on multiple tasks 
        self.task_labels = dict()   # Labels for each task for the instance

    def read_vec():
        # TODO: Read self.input_file, assign values to vec and task_labels
        pass

class MultitaskNetwork:
    def __init__(self, config):
        self.config = config
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]

        # TODO: Read self.input_file, assign values to vec and task_labels
