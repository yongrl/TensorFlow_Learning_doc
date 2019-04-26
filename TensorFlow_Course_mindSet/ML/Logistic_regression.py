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

tf.app.flags.DEFINE_integer('batch_size',128,'batch size of dataset')

tf.app.flags.DEFINE_integer('num_epochs',10,'Number of epochs for training.')

# learning rate flags
tf.app.flags.DEFINE_float('initial_learning_rate',0.001,'Initial learning rate')

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
    decay_steps = int(num_train_samples/FLAGS.batch_size*FLAGS.num_epochs_per_decay)
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,global_step,decay_steps,
                                               FLAGS.learning_rate_decay_factor,staircase=True,
                                               name='exponential_decay_learning_rate')

    #defining place holders
    image_place = tf.placeholder(tf.float32,shape=([None,num_features]),name='image')
    label_place = tf.placeholder(tf.int32,shape=([None,]),name='gt')
    label_one_hot = tf.one_hot(label_place,depth=FLAGS.num_classes,axis=-1)
    dropout_param = tf.placeholder(tf.float32)


    # model + loss + accuracy
    # A simple fully connected with two class and a softmax is equivalent to logistic regression
    logits = tf.contrib.layers.fully_connected(inputs = image_place,num_outputs = FLAGS.num_classes,scope='fc')

    # Define loss
    with tf.name_scope('loss'):
        loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label_one_hot))

    # Accurary
    # Evaluate the model
    prediction_correct = tf.equal(tf.argmax(logits,1),tf.argmax(label_one_hot,1))

    # Accuracy calculation
    accuracy = tf.reduce_mean(tf.cast(prediction_correct,tf.float32))


    # training operation

    # define optimizer by its default values
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # 'train_op' is a operation that is run for gradient update on parameters
    # Each execution of 'train_op' is a training step
    # by passing 'global_step' to the optimizer,each time that the 'train_op' is run, tensorflow
    # update the 'global_step' and increment it by one

    # gradient update
    with tf.name_scope('train_op'):
        gradients_and_variables = optimizer.compute_gradients(loss_tensor)
        train_op = optimizer.apply_gradients(gradients_and_variables,global_step = global_step)

    # run the session
    session_conf = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
        log_device_placement = FLAGS.log_device_placement)

    sess = tf.Session(graph=graph,config=session_conf)

    with sess.as_default():

        # The saver op
        saver = tf.train.Saver()

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # The prefix for checkpoint files
        checkpoint_prefix = 'model'

        # if fine-tuning flag in 'True' the model will be restored.
        if FLAGS.fine_tuning:
            saver.restore(sess,os.path.join(FLAGS.checkpoint_path,checkpoint_prefix))
            print("Model restored for fine-tuning...")

        # Run the training the loop over the batches
        # go through the batches
        test_accuary = 0
        for epoch in range(FLAGS.num_epochs):
            total_batch_training = int(data['train_image'].shape[0]/FLAGS.batch_size)

            #go through the batches
            for batch_num in range(total_batch_training):
                '''
                get the training batches
                '''
                start_idx = batch_num * FLAGS.batch_size
                end_idx = (batch_num+1)*FLAGS.batch_size

                # Fit training using batch data
                train_batch_data,train_batch_label = data['train_image'][start_idx:end_idx],\
                                                    data['train_label'][start_idx:end_idx]

                # run the session
                # Run optimization op(backprop) and calculate batch loss and accuracy
                # When the tensor tensors['global_step'] is evaluated, it will be incremented by one
                batch_loss,_,training_step = sess.run(
                    [loss_tensor,train_op,global_step],
                    feed_dict={image_place: train_batch_data,
                               label_place:train_batch_label,
                               dropout_param:0.5}
                )

            # write summaries
            # plot the progressive bar
            print("Epoch " + str(epoch+1)+", Training Loss = "+\
                  "{:.5f}".format(batch_loss))

        # Saving the model checkpoint

        # the model will be saved when the training is done
        # create the path for saving the checkpoints.
        if not os.path.exists(FLAGS.checkpoint_path):
            os.makedirs(FLAGS.checkpoint_path)

        # save the model
        save_path = saver.save(sess,os.path.join(FLAGS.checkpoint_path,checkpoint_prefix))
        print("Model saved in file:%s"%save_path)

        # run the session for evaluation on the test data
        # The prefix for checkpoint files
        checkpoint_prefix = 'model'

        # Restoring the saved weights
        saver.restore(sess,os.path.join(FLAGS.checkpoint_path,checkpoint_prefix))
        print("Model restored...")

        # Evaluation of the model
        test_accuary = 100*sess.run(accuracy,feed_dict={
            image_place:data['test_image'],
            label_place:data['test_label'],
            dropout_param:1.
        })

        print("Final Test Accuracy is %% %.2f"%test_accuary)






































