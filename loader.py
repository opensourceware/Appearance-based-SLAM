import scipy
import scipy.io
import config
import os

def load_sparse_mat():
    ground_truth = scipy.io.loadmat(config.GROUND_TRUTH)
    mat = ground_truth['ground_truth']
    arr = mat.toarray()
    ones = mat.nonzero()
    loop_closure_loc = []
    for index0, index1 in zip(ones[0], ones[1]):
        loop_closure_loc.append((index0, index1))	
    return loop_closure_loc

def chow_liu():
	data = scipy.io.loadmat(config.CLTREE)
	CLTree = data['ChowTree']
	CLTree.shape = (1, f['ChowTree'].shape[0])
	con = scipy.io.loadmat(config.CONDITIONALS)
	conditionals = con['RelevantConditionals']
	return CLTree, conditionals

def load_all():
	mar = scipy.io.loadmat(os.path.join(os.getcwd()+"/"+config.MARGINALS))
	marginal = mar['Marginals']
	cond = scipy.io.loadmat(os.path.join(os.getcwd()+"/"+config.CONDITIONALS))
	conditional = cond['RelevantConditionals']
	not_cond = scipy.io.loadmat(os.path.join(os.getcwd()+"/"+config.NOT_CONDITIONALS))
	not_conditional = not_cond['RelevantNotConditionals']
	return marginal, conditional, not_conditional

if __name__ == "__main__":
    loop_closure_loc = load_sparse_mat()

