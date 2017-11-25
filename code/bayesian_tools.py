import time
import numpy
from scipy.integrate import quad
from tools import bleu_file, get_ref_files
import random
import copy
import os
import cPickle

# paths
multibleu = "~/preprocess_MT/multi-bleu.perl"
DEV = "/data/disk1/share/zjc/nist_thulac/dev_test/nist06"
src = DEV+'/nist06.cn'
ref = DEV+'/nist06.en'
DEV_split = "/data/disk1/share/zjc/nist_thulac/dev_test/nist06_split"
src_100 = DEV_split+'/nist06_100.cn'
ref_100 = DEV_split+'/nist06_100.en'
src_rem = DEV_split+'/nist06_rem.cn'
ref_rem = DEV_split+'/nist06_rem.en'
translator = '~/git/THUMT_171010/thumt/translate.py'
combiner = '~/ACL2018/code/combiner.py'
mapping = '/data/disk1/share/zjc/wmt17/exp/1709_1710/dict_zhen/zhen_1710.mapping'
system = "tensorflow" # "theano", "tensorflow"
if system == "tensorflow":
	import tensorflow as tf
	import numpy as np
	translator = '~/git/THUMT-TF/THUMT/thumt/launcher/translator.py'
	pythonpath = '~/git/THUMT-TF/THUMT'
	
	'''
	print 'initializing...'
	var_list = tf.contrib.framework.list_variables('init/model_1')
	print 'var list:', var_list
	var_zeros, var_dtypes = {}, {}
	for (name, shape) in var_list:
		if not name.startswith("global_step"):
			var_zeros[name] = np.zeros(shape)
	reader = tf.contrib.framework.load_checkpoint('init/model_1')
	for name in var_zeros:
		tensor = reader.get_tensor(name)
		var_dtypes[name] = tensor.dtype
	tf_vars = [tf.get_variable(name, shape=var_zeros[name].shape,dtype=var_dtypes[name]) for name in var_zeros]
	placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
	assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
	global_step = tf.Variable(0, name="global_step", trainable=False,dtype=tf.int64)
	print 'initialize finished'
	'''
# THUMT-TF
system_path = '~/git/THUMT-TF/THUMT'

# toy-test
toy = True
if toy:
	src_100 = '~/ACL2018/data/toydev/toy_2.cn'
	ref_100 = '~/ACL2018/data/toydev/toy_2.en'
	src_rem = '~/ACL2018/data/toydev/toy_2.cn'
	ref_rem = '~/ACL2018/data/toydev/toy_2.en'

def get_model(vec, modelpath):
	# THUMT-THEANO
	ndim = vec.shape[0]
	MT_models = []
	if system == 'theano':
		for i in range(ndim):
			tmp = 'init/model_'+str(i+1)+'.npz'
			MT_models.append(numpy.load(tmp))
		model = {}
		for key in MT_models[0]:
			if 'vocab' in key or 'config' in key:
				model[key] = MT_models[0][key]
			else:
				model[key] = MT_models[0][key] * vec[0]
		for i in range(1, ndim):
			for key in MT_models[1]:
				if 'vocab' in key or 'config' in key:
					continue
				else:
					model[key] += MT_models[i][key] * vec[i]
		numpy.savez(modelpath, **model)
		return
	# THUMT-tensorflow
	elif system == 'tensorflow':
		cmd = 'python '+combiner+' '+str(ndim)+' '+modelpath 
		for i in range(ndim):
			cmd += ' '+str(vec[i])
		print cmd
		os.system(cmd)
		'''
		var_values = {}
		for name in var_zeros:
			var_values[name] = var_zeros[name]
		for i in range(ndim):
			reader = tf.contrib.framework.load_checkpoint('init/model_'+str(i+1))
			for name in var_values:
				tensor = reader.get_tensor(name)
				var_dtypes[name] = tensor.dtype
				var_values[name] += tensor * vec[i]
		#tf.variable_scope('trainging/rnnsearch').reuse = tf.AUTO_REUSE
		saver = tf.train.Saver(tf.global_variables())
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for p, assign_op, (name, value) in zip(placeholders, assign_ops,var_values.iteritems()):
				sess.run(assign_op, {p: value})
			saved_name = modelpath
			saver.save(sess, saved_name, global_step=global_step)
		'''
		return

def translate(src, trg, model_path, device, log, mapping=None):
	# THUMT-THEANO
	if system == 'theano':
		cmd = 'THEANO_FLAGS=floatX=float32,lib.cnmem=0.2,device='+device+' python '+translator+' -i '+src+' -o '+trg+' -m '+model_path+' -unk -ln -map '+mapping+' > '+log
		os.system(cmd)
		return
	# THUMT-tensorflow
	elif system == 'tensorflow':
		# generate checkpoint
		modelname = model_path.split('/')[-1]
		cfile = open('model/checkpoint' ,'w')
		cfile.write('model_checkpoint_path: "'+modelname+'"')
		cfile.close()
		cmd = 'CUDA_VISIBLE_DEVICES='+device+' PYTHONPATH='+pythonpath+' python '+translator+' --model rnnsearch --vocabulary init/vocab.30000.zh.txt init/vocab.30000.en.txt --checkpoints '+model_path+'-0 --input '+src+' --output '+trg+' > '+log
		print 'translate:', cmd
		os.system(cmd)
		return


def multi_bleu(hypo,ref,evalpath):
	cmd = 'perl '+multibleu+' -lc '+ref+' < '+hypo+' > '+evalpath
	print cmd
	os.system(cmd)
	with open(evalpath, 'r') as f:
		content = f.read()
		if content.strip() == '':
			result = '0.00'
		else:
			result = content[7:12]
	if ',' in result:
		result = result.split(',')[0]
	print evalpath,'BLEU:', result
	result = float(result)
	return result

def checkline(src, trg):
	srcline = get_num_lines(src)
	trgline = get_num_lines(trg)
	if srcline != trgline:
		print 'line number different after translation!', srcline ,'!=',trgline
		assert 0 == 1

def evaluate_old(vec):
	print 'evaluate point:', vec
	#time.sleep(1)
	result = 2
	ndim = vec.shape[0]
	for i in range(ndim):
		result -= 10*(vec[i]-0.1*i)**2
	for i in range(ndim):
		result -= (vec[i]-0.05*i)**2
	return result

def getid(vec):
	result = str(round(vec[0], 3))
	dim = vec.shape[0]
	for i in range(1,dim):
		result += '_' + str(round(vec[i], 3))
	return result

def evaluate_MT_linear(vec, norm=False, device='cpu'):
	'''
		evaluate with the linear combination of MT models
	'''
	print 'evaluate point:', vec
	identifier = str(random.random())
	identifier = getid(vec)
	ndim = vec.shape[0]
	if norm:
		vec = vec/sum(vec)
	# paths
	trg = 'model/'+identifier+'.trans'
	modelpath = 'model/'+identifier+'.npz'
	if system == "tensorflow":
		modelpath = 'model/model_'+identifier
	modellog = 'model/'+identifier+'.log'
	evalpath='model/'+identifier+'.eval'
	trg_100 = 'model_split/'+identifier+'_100.trans'
	modellog_100 = 'model_split/'+identifier+'_100.log'
	evalpath_100 ='model_split/'+identifier+'_100.eval'
	trg_rem = 'model_split/'+identifier+'_rem.trans'
	modellog_rem = 'model_split/'+identifier+'_rem.log'
	evalpath_rem ='model_split/'+identifier+'_rem.eval'
	skip = False

	# check line number
	if os.path.exists(trg):
		srcline = get_num_lines(src)
		trgline = get_num_lines(trg)
		if srcline != trgline:
			print 'line number different!', srcline ,'!=',trgline
			print 'delete the pretranslation'
			os.system('rm '+ trg)

	if not os.path.exists(trg):
		print 'generating model...'
		# combine model
		timegs = time.time()
		get_model(vec, modelpath)
		timege = time.time()
		print 'generation finished, time used:', timege-timegs 
		
		# translate
		# translate first 100 sentences
		if not os.path.exists(trg_100):
			print 'translating 100 sentences'
			timets = time.time()
			translate(src_100,trg_100,modelpath,device,modellog_100,mapping)
			timete = time.time()
			print 'translaion finished, time used:', timete-timets
		else:
			print 'use existing 100 models'
		result_100 = multi_bleu(trg_100, ref_100, evalpath_100)
		# translate remaining sentences
		if result_100 > 25.:
			print 'not too bad, decoding remaining...'
			timets = time.time()
			translate(src_rem,trg_rem,modelpath,device,modellog_rem,mapping)
			timete = time.time()
			print 'translaion finished, time used:', timete-timets
			# combine 
			cmd = 'cat ' + trg_100 + ' ' + trg_rem +' > ' + trg
			os.system(cmd)
		else:
			print 'too bad, skipping'
			result = result_100
			skip = True
	else:
		print 'reuse the translation'

	if not skip:
		# check line number
		checkline(src, trg)

		# evaluate
		result = multi_bleu(trg, ref, evalpath)
	
	# clean
	if os.path.exists(modelpath):
		os.system('rm '+modelpath)
	if os.path.exists(modelpath+'.index'):
		os.system('rm '+modelpath+'.index')
	if os.path.exists(modelpath+'.meta'):
		os.system('rm '+modelpath+'.meta')
	if os.path.exists(modelpath+'.data-00000-of-00001'):
		os.system('rm '+modelpath+'.data-00000-of-00001')

	return result
		
def similarity(v1, v2):
	return numpy.exp(-0.5*sum((v1-v2)**2))

def gaussian(x, mean, var):
	return 1/(numpy.sqrt(2*math.pi)*var) * numpy.exp(-(x-mean)**2/(2*var**2))

def bayesian_predict(vec, x, y, k):
	k_new = numpy.zeros((1, len(x)))
	for i in range(len(x)):
		k_new[0,i] = similarity(vec, x[i])
	mean = numpy.dot(numpy.dot(k_new, numpy.linalg.inv(k)), y)
	var = similarity(vec, vec)-numpy.dot(numpy.dot(k_new, numpy.linalg.inv(k)), k_new.transpose())
	return mean, var

def prob_best(mean, var, best):
	result = quad(lambda x :gaussian(x,mean,var), best, numpy.inf)
	return result[0]

def expect_best(mean, var, best):
	result = quad(lambda x :gaussian(x,mean,var)*(x-best), best, numpy.inf)
	return result[0]

def get_num_lines(filename):
	with open(filename) as f:
		result = sum(1 for line in f)
	return result
