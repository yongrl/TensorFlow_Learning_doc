from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import urllib
import pandas as pd
import os


# Necessary Flags
tf.app.flags.DEFINE_string('train_path',os.path.dirname(os.path.abspath(__file__)) + '/train_logs',
                           'Directory where event logs are written to.')

tf.app.flags.DEFINE_string('checkpoint_path',
                           os.path.dirname(os.path.abspath(__file__))+'/checkpoints',
                           'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer('max_num_checkpoint',10,
                            'Maximum number of checkpoints that tensorflow will keep.')

tf.app.flags.DEFINE_integer('num_classes',2,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('batch_size',np.power(2,9),
                            'batch size of dataset')

tf.app.flags.DEFINE_integer('num_epochs',10,'Number of epochs for training.')

# learning rate flags
tf.app.flags.DEFINE_float('initial_learning rate',0.001,'Initial learning rate')

tf.app.flags.DEFINE_float('learning_rate_decay_factor',0.95,'learning rate decay factor.')

tf.app.flags.DEFINE_float('num_epochs_per_decay',1,'Number of epoch pass to decay learning rate.')


'''
status flags
'''
tf.app.flags.DEFINE_boolean('is_training',False,'Training/Testing.')
tf.app.flags.DEFINE_boolean('fine_tuning',False,'Fine tuning is desired or not?.')
tf.app.flags.DEFINE_boolean('online_test',True,'online_test')
tf.app.flags.DEFINE_boolean('allow_soft_placement',True,'Automatically put the variables on CPU if there is no GPU support.')
tf.app.flags.DEFINE_boolean('log_device_placement',False,'Demonstrate which variables are on what device.')

# Store all elements in FLAG structure!
FLAGS = tf.app.flags.FLAGS

# handling errors!
if not os.path.isabs(FLAGS.train_path):
    raise ValueError('You must assign absolute path for --train_path')

if not os.path.isabs(FLAGS.checkpoint_path):
    raise ValueError('You must assign absolute path for --checkpoint_path')

mnist = input_data.read_data_sets("MNIST_data/",reshape=True,one_hot=False)

# data processing
data = {}

data['train_image'] = mnist.train.images
data['train_label'] = mnist.train.labels
data['test_image'] = mnist.test.images
data['test_label'] = mnist.test.labels

def extract_samples_Fn(data):
    index_list = []
    for sample_index in range(data.shape[0]):
        label = data[sample_index]
        if label == 1 or label ==0:
            index_list.append(sample_index)
    return index_list


# Get only the samples with zero and one label for training.
index_list_train = extract_samples_Fn(data['train_label'])

# Get only the samples with zero and one label for test set
index_list_test = extract_samples_Fn(data['test_label'])

# Reform the test data structure
data['train_image'] = mnist.train.images[index_list_train]
data['train_label'] = mnist.train.labels[index_list_train]

data['test_image'] = mnist.test.images[index_list_test]
data['test_label'] = mnist.test.labels[index_list_test]

# Dimentionality of train
dimensionality_train = data['train_image'].shape

# dimensions
num_train_samples = dimensionality_train[0]
num_features = dimensionality_train[1]

# defining graph
graph = tf.Graph()
with graph.as_default():

    # global step
    global_step = tf.Variable(0,name='global_step',trainable=False)

    # learning rate policy
    decay_steps = int(num_train_samples/FLAGS.batch_size*FLAGS.num_epoch_per_decay)
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,global_step,decay_steps,
                                               FLAGS.learning_rate_decay_factor,staircase=True,
                                               name='exponential_decay_learning_rate')

    #defining place holders
    image_place = tf.placeholder(tf.float32,shape=([None,num_features]),name='image')
    label_place = tf.placeholder(tf.int32,shape=([None,]),name='gt')
    label_one_hot = tf.one_hot(label_place,depth=FLAGS.num_classes,axis=-1)
    dropout_param = tf.placeholder(tf.float32)























