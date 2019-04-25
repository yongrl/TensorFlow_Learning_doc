import tensorflow as tf
from tensorflow.python.framework import ops

# defining variables
# create three variables with some default values
weights = tf.Variable(tf.random_normal([2,3],stddev=0.1),name="weights")
bias = tf.Variable(tf.zeros([3]),name="bias")
custom_variable = tf.Variable(tf.zeros([3]),name='custom')

# Get all the variables' tensors and store them in a list
all_variable_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

# The initializer
# "Variable_list_custom" is the list of variables that we want to initialize
variable_list_custom = [weights,custom_variable]
init_custom_op = tf.variables_initializer(var_list=variable_list_custom)

# Global variable initialization
# All variables can be initialized at once using the tf.global_variables_initializer()
# This op must be run after the model constructed

# method 1: add an op to initialize the variables
init_all_op = tf.global_variables_initializer()

# method 2
init_all_op = tf.variables_initializer(var_list=all_variable_list)

# Initialization of a variables using other existing variables
# Create another variable with the same value as 'weights'.
WeightsNew = tf.Variable(weights.initialized_value(), name="WeightsNew")

# Now, the variable must be initialized.
init_WeightsNew_op = tf.variables_initializer(var_list=[WeightsNew])

with tf.Session() as sess:
    # Run the initializer operation.
    sess.run(init_all_op)
    sess.run(init_custom_op)
    sess.run(init_WeightsNew_op)



