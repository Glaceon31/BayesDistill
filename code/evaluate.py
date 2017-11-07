from bayesian_tools import evaluate_MT_linear
import numpy

def evaluate(vec):
	return evaluate_MT_linear(vec, device='gpu0')

if __name__ == "__main__":
	vec = numpy.asarray([0.2037037, 0.05555556, 0.71399177])
	print evaluate(vec)
