import numpy
from time import clock
import bayesopt
import argparse
import traceback
from bayesian_tools import evaluate_old, evaluate_MT_linear

parser = argparse.ArgumentParser("")
parser.add_argument('-d','--ndim', type=int, required=True)
parser.add_argument('--device', default="cpu")
parser.add_argument('-n','--norm', action="store_true")
args = parser.parse_args()

def evaluate(vec):
	#return -evaluate_old(vec)
	return -evaluate_MT_linear(vec, norm=args.norm, device=args.device)

params = {}
params['n_iterations'] = 1000
params['n_iter_relearn'] = 5
params['n_init_samples'] = 2

n = args.ndim                    # n dimensions
lb = numpy.zeros((n,))
ub = numpy.ones((n,))

if __name__ == "__main__":
	start = clock()
	mvalue, x_out, error = bayesopt.optimize(evaluate, n, lb, ub, params)

	print("Result", mvalue, "at", x_out)
	print("Running time:", clock() - start, "seconds")
