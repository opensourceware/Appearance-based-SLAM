import scipy.io
import os
import config
import loader
import math
import numpy as np

def convertImg2Words(file):
	observations = []
	for i in xrange(5):
		#Parse all the 5 omni-directional files
		filename = os.path.join(os.getcwd()+"/"+config.WORD_DIR+"/"+file[:-5]+str(i)+'.words')
		f = open(filename, 'r')
		words = f.read().split('\n')
		clusters = [word.split()[-1] for word in words[1:-1]]
		observations += clusters
	return observations


class AppearanceModel:
	def __init__(self, init_scene_elems):
		self.scene_elems_prob = list(init_scene_elems)
		self.associated_obs = []
		self.likelihood = 0.0
		self.likelihood_ = 0.0
		self.conditional_e0 = 0.005
		self.conditional_e1 = 0.39

	def observation_likelihood(self, observations, marginal, conditional, not_conditional):
		obsLL = 0.0
		for i in xrange(10000):
			if str(i) in observations:
				cond = conditional[0, i]
				z_cond_e1 = self.conditional_e1
				z_cond_e0 = self.conditional_e0
				marg = marginal[i]
			else:
				cond = not_conditional[0, i]
				z_cond_e1 = (1-self.conditional_e1)
				z_cond_e0 = (1-self.conditional_e0)
				marg = (1-marginal[i])
			ratio_0 = marg*(1-z_cond_e0)*(1-cond)/((1-marg)*z_cond_e0*cond)
			ratio_1 = marg*(1-z_cond_e1)*(1-cond)/((1-marg)*z_cond_e1*cond)
			prob_0 = math.pow(1+ratio_0, -1)
			prob_1 = math.pow(1+ratio_1, -1)
			prob = (prob_1*self.scene_elems_prob[i]+\
				   prob_0*(1-self.scene_elems_prob[i]))
			obsLL+=math.log(prob)
		return obsLL

	def prior(self, loc_prev=None, loc_next=None):
		if (loc_prev is None) and (loc_next is None):
			return self.likelihood_
		elif loc_prev is None:
			return (self.likelihood_+loc_next.likelihood_)*1.0/2
		elif loc_next is None:
			return (loc_prev.likelihood_+self.likelihood_)*1.0/2
		else:
			#return math.log(math.pow(loc_prev.likelihood_*self.likelihood_*loc_next.likelihood_, 1.0/3))
			#Adding likelihood as they are in log space
			return (loc_prev.likelihood_+self.likelihood_+loc_next.likelihood_)*1.0/3

	def compute_and_update_likelihood(self, observations, marginal, conditional, 
		not_conditional, loc_prev=None, loc_next=None):
		self.likelihood = self.observation_likelihood(observations,
							 marginal, conditional, not_conditional)+self.prior(loc_prev, loc_next)
		return self.likelihood

	def updateAppearance(self, observations, marginal, conditional, not_conditional):
		#Observation likelihood is the normalizing term. Hence, we divide ll by prior.
		norm = self.observation_likelihood(observations, marginal, conditional, not_conditional)
		obs_e1 = 1.0
		for i in xrange(10000):
			if str(i) in observations:
				cond = conditional[0, i]
				z_cond_e1 = self.conditional_e1
				marg = marginal[i]
			else:
				cond = not_conditional[0, i]
				z_cond_e1 = (1-self.conditional_e1)
				marg = (1-marginal[i])
			ratio_1 = marg*(1-z_cond_e1)*(1-cond)/((1-marg)*z_cond_e1*cond)
			prob_1 = math.pow(1+ratio_1, -1)
			#if norm>0:
			#	print "Norm is greater than 0! "+str(norm)
			#else:
			#	print "Normal"
			#if prob>0:  
			#	print "Prob is greater than 0! "+str(prob)
			self.scene_elems_prob[i] = prob_1+self.scene_elems_prob[i]#-norm
			#obs_e1*=prob


#Compute scene marginals over training data
def compute_scene_marginals():
	all_observations = []
	with open(config.WORD_FILES, 'r') as f:
		files = f.read().split('\n')
	if files[-1]=='':
		files.pop()
	for file in files:
		all_observations += convertImg2Words(file)
	init_scene = [0.0]*10000
	for z in all_observations:
		init_scene[int(z)] += 1.0
	total_feat = len(all_observations)
	for i in xrange(config.NUM_WORDS):
		init_scene[i] /= total_feat
	return init_scene


class FABMAP:
	def __init__(self):
		self.locations = []
		self.LocLL = []
		self.marginal, self.conditional, self.not_conditional = loader.load_all()
		self.unmapped = []

	def test(self, observations):
		#Run the 70km dataset.
		for num, loc in enumerate(self.locations):
			if num<=0:
				loc_prev = None
			else:
				loc_prev = self.locations[num-1]
			if num>=(len(self.locations)-1):
				loc_next = None
			else:
				loc_next = self.locations[num+1]
			self.LocLL.append(loc.compute_and_update_likelihood(observations, self.marginal,
							 self.conditional, self.not_conditional, loc_prev, loc_next))
			##Update the history likelihood after appearance update
			#loc.likelihood_ = loc.likelihood
		umap = []
		for loc in self.unmapped:
			#No previous and future locations as these locations are unobserved or unmapped; 
			#Hence, no topological ordering.
			umap.append(loc.compute_and_update_likelihood(observations, 
				self.marginal, self.conditional, self.not_conditional))
		l = len(self.unmapped)
		avg = sum(umap)/l
		self.LocLL.append(avg)
		#Fix the same history likelihood value for all unmapped locations;
		#This is the average of likelihood values for unmapped locations.
		for loc in self.unmapped:
			loc.likelihood_ = avg
		return self.normalize()

	def normalize(self):
		LocLL = np.array(self.LocLL)
		return LocLL/np.sum(self.LocLL)

	#Re-initialize after each iteration of file/observation/image capture 
	def _init(self):
		self.LocLL = []

	#Initialize 100 unmapped images with associated likelihood values
	def init_unmapped(self, files, init_scene_elems):
		unmapped = []
		for file in files:
			loc = AppearanceModel(init_scene_elems)
			observations = convertImg2Words(file)
			loc.updateAppearance(observations, self.marginal, self.conditional, self.not_conditional)
			self.unmapped.append(loc)


def main():
	init_scene_elems = compute_scene_marginals()
	marginal, conditional, not_conditional = loader.load_all()
	with open(config.WORD_FILES, 'r') as f:
		files = f.read().split('\n')
	with open(config.UNMAPPED, 'r') as f:
		unmapped_files = f.read().split('\n')
	#Till the last element because its null string
	unmapped_files = [files.pop(int(num)) for num in unmapped_files[:-1]]
	fabmap = FABMAP()
	fabmap.init_unmapped(unmapped_files, init_scene_elems)
	for num, file in enumerate(files):
		observations = convertImg2Words(file)
		LocLL = fabmap.test(observations)
		fabmap._init()	#Clear off the LocLL array
		ind = LocLL.argmax()
		if ind==(LocLL.shape[0]-1):
			print "New Location initialized"
			new_loc = AppearanceModel(init_scene_elems)
			new_loc.updateAppearance(observations, marginal, conditional, not_conditional)
			fabmap.locations.append(new_loc)
		else:
			print "Updating old location"
			fabmap.locations[ind].updateAppearance(observations, marginal, conditional, not_conditional)
			fabmap.locations[ind].associated_obs.append(ind)
		#Set history of likelihood
		for loc in fabmap.locations:
			loc.likelihood_ = loc.likelihood


if __name__ == "__main__":
	main()

