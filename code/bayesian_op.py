import numpy
import math
import random
import copy
from matplotlib import pyplot as plt
import time
import os
from tools import bleu_file, get_ref_files
from bayesian_tools import evaluate_old, evaluate_MT_linear
import argparse
import cPickle
import threading
import traceback
from multiprocessing.dummy import Pool as ThreadPool 

parser = argparse.ArgumentParser("")
parser.add_argument('-d','--ndim', type=int, required=True)
parser.add_argument('--device', default="cpu")
parser.add_argument('-n','--norm', action="store_true")
args = parser.parse_args()


global MT_models
MT_models = None


def evaluate(vec):
	#return evaluate_old(vec)
	try:
		return evaluate_MT_linear(vec, norm=args.norm, device=args.device)
	except:
		traceback.print_exc()
		raise RuntimeError

class DIRECT(object):

	def __init__(self, dimensions, load = False):
		self.dimensions = dimensions
		self.compare_count = 0
		if load:
			return
		self.rects = [{'left': numpy.zeros((self.dimensions,)), 'right':numpy.ones((self.dimensions,))}]
		self.samples = [numpy.ones((self.dimensions,))*0.5]
		self.results = [evaluate(s) for s in self.samples]
		self.bests = []
		self.best = self.results[0]
		self.best_index = 0
		self.num_iter = 0

	def save_checkpoint(self):
		tmp = {}
		tmp['bests'] = self.bests
		tmp['best'] = self.best
		tmp['best_index'] = self.best_index
		tmp['samples'] = self.samples
		tmp['results'] = self.results
		tmp['rects'] = self.rects
		tmp['num_iter'] = self.num_iter
		with open('checkpoint', 'w') as f:
			cPickle.dump(tmp, f)
		return

	def load(self, checkpoint):
		tmp = cPickle.load(open('checkpoint', 'r'))
		self.bests = tmp['bests']
		self.best = tmp['best']
		self.best_index = tmp['best_index']
		self.samples = tmp['samples']
		self.results = tmp['results']
		self.rects = tmp['rects']
		self.num_iter = tmp['num_iter']
		return

	def save_log(self):
		# write evaluation log
		elog = open('evaluation.log', 'w')
		for i in range(len(self.samples)):
			print >> elog, str(self.samples[i]) + '\t' + str(self.results[i])

		# write iteration log
		ilog = open('iteration.log', 'w')
		for i in range(len(self.bests)):
			print >> ilog, str(i) + '\t' + str(self.bests[i])


	def plot(self):
		x = []
		y = []
		for i in range(len(self.samples)):
			sample = self.samples[i]
			x.append(sample[0])
			y.append(sample[1])
		plt.scatter(x,y)
		plt.show()

	def plot_best(self):
		x = range(len(self.bests))
		plt.plot(x,self.bests)
		plt.show()

	def rect_value(self, rect):
		if rect.has_key('value'):
			#print 'good'
			return rect['value']
		centerp = self.center(rect)
		distance2 = 10000
		index = -1
		rect_start = time.time()
		for i in range(len(self.samples)):
			sample = self.samples[i]
			if sum((sample-centerp) ** 2) < distance2:
				distance2 = sum((sample-centerp) ** 2) 
				index = i
		rect['value'] = self.results[index]
		rect_end = time.time()
		#print 'rect time:', rect_end - rect_start, 'secs'
		return self.results[index]

	def center(self, rect):
		return (rect['left']+rect['right'])/2

	def diag(self, rect):
		if rect.has_key('dist') and rect['dist'] > 0:
			return rect['dist']
		else:
			rect['dist'] = numpy.sqrt(sum((rect['right']-rect['left'])**2))
		return rect['dist']

	def potential_optimal(self, rect):
		for i in range(len(self.rects)):
			self.compare_count += 1
			if self.rect_value(rect) >= self.rect_value(self.rects[i]) and self.diag(rect) >= self.diag(self.rects[i]):
				continue
			if self.rect_value(rect) <= self.rect_value(self.rects[i]) and self.diag(rect) <= self.diag(self.rects[i]):
				return False
		return True


	def potential_optimal_set(self):
		result = []
		for i in range(len(self.rects)):
			rect = self.rects[i]
			if self.potential_optimal(rect):
				result.append(i)
		return result

	def split_rect(self, index):
		dindex = [] #the longest dimensions
		maxd = 0
		rect = self.rects[index]
		# find the longest dimension(s)
		for d in range(self.dimensions):
			if rect['right'][d] - rect['left'][d] > maxd:
				maxd = rect['right'][d] - rect['left'][d]
				dindex = [d]
			elif rect['right'][d] - rect['left'][d] == maxd:
				dindex.append(d)
		centerp = self.center(rect)
		dist = maxd/3
		result_d = {}
		
		# sample
		new_samples = []
		new_results = []
		for d in dindex:
			newdirection = numpy.zeros((self.dimensions,))
			newdirection[d] = dist
			# positive
			newsample = centerp+newdirection
			new_samples.append(newsample)
			new_results.append(-1)
			# negative
			newsample = centerp-newdirection
			new_samples.append(newsample)
			new_results.append(-1)
		# synchorized evaluation
		pool = ThreadPool(1)
		new_results = pool.map(evaluate, new_samples)
		'''
		for i in range(len(new_samples)):
			newsample = new_samples[i]
			performance = evaluate(newsample)
			new_results[i] = performance
		'''

		# update best result
		print dindex 
		for i in range(len(new_samples)):
			performance = new_results[i]
			if performance > self.best:
				self.best = performance
				self.best_index = len(self.samples)-1+i
			d = dindex[i/2]
			if not result_d.has_key(d):
				result_d[d] = performance
			else:
				result_d[d] = max(result_d[d], performance)
		self.samples += new_samples
		self.results += new_results

		# split
		result_d = sorted(result_d.iteritems(), key=lambda x:x[1],reverse=True)
		print result_d
		nowrect = rect
		for tmp in result_d:
			d = tmp[0]
			newrect_l = {'left': nowrect['left']+0, 'right':nowrect['right']+0}
			newrect_r = {'left': nowrect['left']+0, 'right':nowrect['right']+0}
			newrect_l['right'][d] = nowrect['left'][d] + dist
			newrect_r['left'][d] = nowrect['right'][d] - dist
			self.rects[index]['left'][d] = nowrect['left'][d] + dist
			self.rects[index]['right'][d] = nowrect['right'][d] - dist
			self.rects[index]['dist'] = -1
			nowrect = self.rects[index]
			self.rects.append(newrect_l)
			self.rects.append(newrect_r)
		for rect in self.rects:
			tmp = self.rect_value(rect)
		#self.rects = sorted(self.rects, key=lambda x: x['value'], reverse = True)
	
	def step(self):
		print '---iter:',self.num_iter, '---'
		time_start = time.time()
		# find a potential optimal set
		poset = self.potential_optimal_set()
		time_findop = time.time()
		print 'findop:', time_findop - time_start, 'secs'
		print 'potentials:',len(poset),'/', len(self.rects)

		# split one potential optimal rect
		ran = random.randint(0,len(poset)-1)
		self.split_rect(poset[ran])	

		print 'best:',self.best
		self.bests.append(self.best)
		time_end = time.time()
		print 'step time:',  time_end - time_start, 'secs'
		sampler.num_iter += 1
		self.save_log()
		self.save_checkpoint()

if __name__ == "__main__":
	ndim = args.ndim

	MT_models = None

	# use DIRECT to resolve problem
	if os.path.exists('checkpoint'):
		print 'load from checkpoint'
		sampler = DIRECT(ndim, load = True)
		sampler.load('checkpoint')
	else:
		print 'start from scratch'
		sampler = DIRECT(ndim)

	maxiter = 1000
	while sampler.num_iter < maxiter:
		sampler.step()
	sampler.save_log()
	
	print 'total compare count:', sampler.compare_count

	#bp = numpy.asarray([0.,0.1,0.2,0.3,0.4,0.5,0.6])
	#bp = numpy.asarray([0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
	#print 'good point:', evaluate(bp)
	#sampler.plot_best()
	'''
	for num_iter in range(maxiter):
		print '---iter:',num_iter, '---'
		# sample new parameter set
		newvecs = sampler.get_samples(x)
		print 'newvecs:', newvecs

		# find an optimal parameter to append 
		index = -1
		prob_max = 0
		for i in range(len(newvecs)):
			mean, var = bayesian_predict(newvecs[i],x,y,k)
			pbest = prob_best(mean, var, best)
			ebest = expect_best(mean, var, best)
			print 'bayesian_predict:', newvecs[i], mean, var, pbest,ebest
			if pbest > prob_max:
				prob_max = pbest
				index = i
		newnode = numpy.zeros((1, len(x)))
		for i in range(len(x)):
			newnode[0,i] = similarity(newvecs[index], x[i])
		x.append(newvecs[index])
		y.append(evaluate(newvecs[index]))
		if y[-1] > best:
			best = y[-1]
		new_k = numpy.zeros((len(x), len(x)))
		new_k[0:len(x)-1,0:len(x)-1] = k
		new_k[0:len(x)-1,len(x)-1] = newnode.transpose().flatten()
		new_k[len(x)-1,0:len(x)-1] = newnode
		new_k[len(x)-1,len(x)-1] = similarity(newvecs[index], newvecs[index])
		k = new_k
		sampler.choose_node(newvecs[index])

		
		print 'x:',x
		print 'y:',y
		print 'best:',best
	'''

