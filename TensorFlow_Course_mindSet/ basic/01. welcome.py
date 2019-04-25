import tensorflow as tf
import os


# DuplicateFlagError: The flag 'log_dir' is defined twice. absl.logging has defined log_dir flag
# abslpy is a  Python library code for building Python applications. The code is collected from Google's own Python code base,
# and has been extensively tested and used in production.
tf.app.flags.DEFINE_string(
    'log_dir_1',os.path.dirname(os.path.abspath(__file__))+'/logs',
    'Directory where event logs are written to.'
)


# os.path.dirname(os.path.abspath(__file__)): gets the directory name of the current
# python file.


# Store all elements in Flag structure
FLAGS = tf.app.flags.FLAGS
# FLAGS indicator points to all defined flags

# only work with absolute path and raise error if the path is not absolute
# os.path.expanduser is leveraged to transform '~' sign to the corresponding path indicator.
#       Example: '~/logs' equals to '/home/username/logs'
# if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
#     raise ValueError('You must assign absolute path for --log_dir')
#
# Defining some sentence!
welcome = tf.constant('Welcome to TensorFlow world!')

a = tf.constant(5.0,name='a')
b = tf.constant(10.0,name='b')

x= tf.add(a,b,name='add')
y = tf.add(a,b,name='divide')

# Run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir_1), sess.graph)
    print("output: ", sess.run([welcome,a,b,x,y]))

# Closing the writer.
writer.close()
sess.close()
# tensorboard 1.13 may be cause problem and transfer to 1.12.1 version would fix
# this problem


