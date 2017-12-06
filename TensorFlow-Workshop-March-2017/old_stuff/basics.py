# To be interactively typed in order to explore TF basics.
# Following Oriley's: Hello TensorFlow

## Basic Exploration ##
import tensorflow as tf

graph = tf.get_default_graph()

graph.get_operations()

input_value = tf.constant(1.0)

operations = graph.get_operations()

operations

operations[0].node_def

sess = tf.Session()

# TensorFlow manages it's own state of things and maintains a method of evaluating and executing code.  
sess.run(input_value)

## The simplest TensorFlow neuron ##

weight = tf.Variable(0.8)

# Display the operations added to the graph as a result. 
for op in graph.get_operations(): print(op.name)

output_value = weight * input_value

op = graph.get_operations()[-1]
op.name

for op_input in op.inputs: print(op_input)

# Generates an operation which initializes all our variables ( in this case just weight ).
#if you add more variables you'll want to use tf.initialize_all_variables() again; a stale init wouldn't include the new variables.

init = tf.initialize_all_variables()
sess.run(init)

sess.run(output_value)

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.mul(w, x, name='output')

summary_writer = tf.train.SummaryWriter('log_simple_graph', sess.graph_def)

# Command line: tensorboard --logdir=log_simple_graph
#localhost:6006/#graphs

## Training a sinngle Neuron ##

y_ = tf.constant(0.0)

# Defining the loss function as the squared diff between current output and desired. 

loss = (y - y_)**2

optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)
grads_and_vars = optim.compute_gradients(loss)
sess.run(tf.initialize_all_variables())
sess.run(grads_and_vars[1][0])

sess.run(optim.apply_gradients(grads_and_vars))

sess.run(w)

train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
for i in range(100):
	print('before step {}, y is {}'.format(i, sess.run(y)))
	summary_str = sess.run(summary_y)
	summary_writer.add_summary(summary_str, i)
	sess.run(train_step)



sess.run(y)








