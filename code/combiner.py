import tensorflow as tf
import numpy as np
import os
import sys

ndim = int(sys.argv[1])
vec = np.zeros((ndim,))
for i in range(ndim):
	vec[i] = float(sys.argv[i+3])
modelpath = sys.argv[2]

var_list = tf.contrib.framework.list_variables('init/model_1')

var_values, var_dtypes = {}, {}
for (name, shape) in var_list:
	if not name.startswith("global_step"):
		var_values[name] = np.zeros(shape)

for i in range(ndim):
	reader = tf.contrib.framework.load_checkpoint('init/model_'+str(i+1))
	for name in var_values:
		tensor = reader.get_tensor(name)
		var_dtypes[name] = tensor.dtype
		var_values[name] += tensor * vec[i]

tf_vars = [tf.get_variable(name, shape=var_values[name].shape,dtype=var_dtypes[name]) for name in var_values]
placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
global_step = tf.Variable(0, name="global_step", trainable=False,dtype=tf.int64)

saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for p, assign_op, (name, value) in zip(placeholders, assign_ops,var_values.iteritems()):
		sess.run(assign_op, {p: value})
	saved_name = modelpath
	saver.save(sess, saved_name, global_step=global_step)
	sess.run(tf.global_variables_initializer())
