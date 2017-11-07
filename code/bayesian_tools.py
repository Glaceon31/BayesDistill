import time
import numpy
from scipy.integrate import quad
from tools import bleu_file, get_ref_files
import random
import copy
import os
import cPickle

def multi_bleu(hypo,ref,evalpath):
	multibleu = "~/preprocess_MT/multi-bleu.perl"
	cmd = 'perl '+multibleu+' -lc '+ref+' < '+hypo+' > '+evalpath
	print cmd
	os.system(cmd)
	with open(evalpath, 'r') as f:
		result = f.read()[7:12]
	if ',' in result:
		result = result.split(',')[0]
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
	src = '/data/disk1/share/zjc/nist_thulac/dev_test/nist06/nist06.cn'
	ref = '/data/disk1/share/zjc/nist_thulac/dev_test/nist06/nist06.en'
	trg = 'model/'+identifier+'.trans'
	modelpath = 'model/'+identifier+'.npz'
	modellog = 'model/'+identifier+'.log'
	evalpath='model/'+identifier+'.eval'
	src_100 = '/data/disk1/share/zjc/nist_thulac/dev_test/nist06_split/nist06_100.cn'
	ref_100 = '/data/disk1/share/zjc/nist_thulac/dev_test/nist06_split/nist06_100.en'
	trg_100 = 'model_split/'+identifier+'_100.trans'
	modellog_100 = 'model_split/'+identifier+'_100.log'
	evalpath_100 ='model_split/'+identifier+'_100.eval'
	src_rem = '/data/disk1/share/zjc/nist_thulac/dev_test/nist06_split/nist06_rem.cn'
	ref_rem = '/data/disk1/share/zjc/nist_thulac/dev_test/nist06_split/nist06_rem.en'
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
		print 'decoding...'
		# combine model
		MT_models = []
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
		
		# translate
		translator = '~/git/THUMT_171010/thumt/translate.py'
		mapping = '/data/disk1/share/zjc/wmt17/exp/dict_zhen/zhen_1710.mapping'
		# translate first 100 sentences
		if not os.path.exists(trg_100):
			print 'translating 100 sentences'
			cmd = 'THEANO_FLAGS=floatX=float32,lib.cnmem=0.2,device='+device+' python '+translator+' -i '+src_100+' -o '+trg_100+' -m '+modelpath+' -unk -ln -map '+mapping+' > '+modellog_100 
			os.system(cmd)
		else:
			print 'use existing 100 models'
		result_100 = multi_bleu(trg_100, ref_100, evalpath_100)
		# translate remaining sentences
		if result_100 > 25.:
			print 'not too bad, decoding remaining...'
			cmd = 'THEANO_FLAGS=floatX=float32,lib.cnmem=0.2,device='+device+' python '+translator+' -i '+src+' -o '+trg+' -m '+modelpath+' -unk -ln -map '+mapping+' > '+modellog 
			os.system(cmd)
			# combine 
			cmd = 'cat ' + trg_100 + ' ' + trg_rem +' > ' + trg
			os.system(cmd)
		else:
			print 'too bad, skipping'
			result = result_100
			skip = True
		#cmd = 'THEANO_FLAGS=floatX=float32,lib.cnmem=0.2,device='+device+' python '+translator+' -i '+src+' -o '+trg+' -m '+modelpath+' -unk -ln -map '+mapping+' > '+modellog 
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
