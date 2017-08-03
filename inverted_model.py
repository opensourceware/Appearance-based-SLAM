import scipy.io
import os
import config
import loader
import math
import numpy as np
import pickle
#import cv2

def convertImg2Words(file):
	observations = []
	for i in range(5):
		#Parse all the 5 omni-directional files
		filename = os.path.join(os.getcwd()+"/"+config.WORD_DIR+"/"+file[:-5]+str(i)+'.words')
		f = open(filename, 'r')
		words = f.read().split('\n')
		clusters = [word.split()[-1] for word in words[1:-1]]
		observations += clusters
	return observations


class AppearanceModel:
	def __init__(self, init_scene_elems, null_scene_elems, marginal, conditional, not_conditional):
		self.scene_elems_prob = list(init_scene_elems)
		self.null_scene_elems_prob = list(null_scene_elems)
		self.conditional_e0 = 0.005
		self.conditional_e1 = 0.39
		self.default = self.init_default(marginal, conditional, not_conditional)
		self.init_likelihood()

	def init_default(self, marginal, conditional, not_conditional):
		num = []
		den = []
		for obs in range(10000):
			num.append(self.observationLL(obs, marginal, conditional, not_conditional, False, False))
			den.append(self.observationLL(obs, marginal, conditional, not_conditional, True, False))
		default = np.log(np.array(num)/sum(den))
		return default

	def init_likelihood(self):
		self.likelihood = sum(self.default)

	def observationLL(self, obs, marginal, conditional, not_conditional, null, z_bool):
		obs = int(obs)
		if z_bool:
			cond = conditional[0, obs]
			z_cond_e1 = self.conditional_e1
			z_cond_e0 = self.conditional_e0
			marg = marginal[obs]
		else:
			cond = not_conditional[0, obs]
			z_cond_e1 = (1-self.conditional_e1)
			z_cond_e0 = (1-self.conditional_e0)
			marg = (1-marginal[obs])
		ratio_0 = marg*(1-z_cond_e0)*(1-cond)/((1-marg)*z_cond_e0*cond)
		ratio_1 = marg*(1-z_cond_e1)*(1-cond)/((1-marg)*z_cond_e1*cond)
		prob_0 = math.pow(1+ratio_0, -1)
		prob_1 = math.pow(1+ratio_1, -1)
		if not null:
			prob = (prob_1*self.scene_elems_prob[obs]+\
					prob_0*(1-self.scene_elems_prob[obs]))
		else:
			prob = (prob_1*self.null_scene_elems_prob[obs]+\
					prob_0*(1-self.null_scene_elems_prob[obs]))
		return prob

	def updateAppearance(self, observations, marginal, conditional, not_conditional):
		#Observation likelihood is the normalizing term. Hence, we divide ll by prior.
		for obs in observations:
			obs = int(obs)
			cond = conditional[0, obs]
			z_cond_e1 = self.conditional_e1
			marg = marginal[obs]
			ratio_1 = marg*(1-z_cond_e1)*(1-cond)/((1-marg)*z_cond_e1*cond)
			prob_1 = math.pow(1+ratio_1, -1)
			self.scene_elems_prob[obs] = prob_1+self.scene_elems_prob[obs]#-norm


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
	for i in range(config.NUM_WORDS):
		init_scene[i] /= total_feat
	return init_scene


def compute_null_scene_elems(init_scene_elems, marginal, conditional, not_conditional):
	null_scene_elems = [0.0]*10000
	for i in range(10000):
		conditional_e1 = 0.39
		cond = not_conditional[0, i]
		z_cond_e1 = (1-conditional_e1)
		marg = (1-marginal[i])
		ratio_1 = marg*(1-z_cond_e1)*(1-cond)/((1-marg)*z_cond_e1*cond)
		prob_1 = math.pow(1+ratio_1, -1)
		null_scene_elems[i] = prob_1*init_scene_elems[i]
	return null_scene_elems


class FABMAP:
	def __init__(self):
		self.inverted_index = [[] for i in range(10000)]
		self.locations = []
		self.marginal, self.conditional, self.not_conditional = loader.load_all()
		self.unmapped = []

	def computeLL(self, loc, obs):
		obs = int(obs)
		num = loc.observationLL(obs, self.marginal, self.conditional, self.not_conditional, False, True)
		den = loc.observationLL(obs, self.marginal, self.conditional, self.not_conditional, True, False)
		return num/den

	#Build inverted index for every new observation as 
	#each observation is instantiated to a new Apperance Model.
	def build_inverted_index(self, loc, observations, init_scene_elems, null_scene_elems, marginal, conditional, not_conditional):
		for obs in observations:
			obs = int(obs)
			self.inverted_index[obs].append(loc)

	def test(self, observations):
		#Run the 70km dataset.
		for obs in observations:
			obs = int(obs)
			locations = self.inverted_index[obs]
			for loc in locations:
				llterm = self.computeLL(loc, obs)
				#The likelihood is updated for location in both self.inverted_index and self.locations
				#but this copy of loc is shared between the two.
				loc.likelihood += math.log(llterm)-loc.default[obs]

	#Initialize 100 unmapped images with associated likelihood values
	def init_unmapped(self, files, init_scene_elems, null_scene_elems, marginal, conditional, not_conditional):
		unmapped = []
		for file in files:
			loc = AppearanceModel(init_scene_elems, null_scene_elems, marginal, conditional, not_conditional)
			observations = convertImg2Words(file)
			loc.updateAppearance(observations, self.marginal, self.conditional, self.not_conditional)
			self.unmapped.append(loc)
			self.build_inverted_index(loc, observations, init_scene_elems, null_scene_elems, marginal, conditional, not_conditional)
		self.locations.append(self.unmapped)

	def max(self, observations):
		#Pick the most likely location
		locationLL = []
		for location in self.locations:
			sum = 0
			for loc_sample in location:
				sum+=loc_sample.likelihood
			locationLL.append(sum/len(location))
		ind = np.argmax(np.array(locationLL))
		return locationLL[ind], self.locations[ind], ind

	def init_likelihood(self, observations):
		for obs in observations:
			obs = int(obs)
			locations = self.inverted_index[obs]
			for loc in locations:
				loc.init_likelihood()


def main():
	#init_scene_elems = compute_scene_marginals()
	f = open(config.INIT_SCENE_MARGINALS, 'rb')
	init_scene_elems = pickle.load(f)
	f.close()
	marginal, conditional, not_conditional = loader.load_all()
	null_scene_elems = compute_null_scene_elems(init_scene_elems, marginal, conditional, not_conditional)
	with open(config.WORD_FILES, 'r') as f:
		word_files = f.read().split('\n')
	with open(config.UNMAPPED, 'r') as f:
		unmapped_files_num = f.read().split('\n')
	#Till the last element because its null string
	unmapped_files = [word_files[int(num)] for num in unmapped_files_num[:-1]]
	files = []
	for num, file in enumerate(word_files):
		if num not in unmapped_files_num:
			files.append(file)
	with open(config.TEST_FILES, 'r') as f:
		file_indexes = f.read().split('\n')
	files = [word_files[int(i)] for i in file_indexes[:-1]]
	fabmap = FABMAP()
	fabmap.init_unmapped(unmapped_files, init_scene_elems, null_scene_elems, marginal, conditional, not_conditional)
	matches = []
	print("Building path..")
	for num, file in enumerate(files[:6]):
		observations = convertImg2Words(file)
		fabmap.test(observations)
		ll, location, index = fabmap.max(observations)
		fabmap.init_likelihood(observations)
		print("New Location initialized")
		new_loc = AppearanceModel(init_scene_elems, null_scene_elems, marginal, conditional, not_conditional)
		new_loc.updateAppearance(observations, marginal, conditional, not_conditional)
		#if index==0:
		fabmap.locations.append([new_loc])
		#else:
		#	fabmap.locations[index].append(new_loc)
		fabmap.build_inverted_index(new_loc, observations, init_scene_elems, null_scene_elems, marginal, conditional, not_conditional)
		print("Location Index    Match Index")
		print("Predicted\t"+str((num, index)))
		#print((num, index))
		print("Ground Truth\t"+str((num, 0)))
		#print((num, 0))
		matches.append((num, index))
	print("Testing path..")
	for num, file in enumerate(files[6:]):
		observations = convertImg2Words(file)
		fabmap.test(observations)
		ll, location, index = fabmap.max(observations)
		fabmap.init_likelihood(observations)
		print("New Location initialized")
		new_loc = AppearanceModel(init_scene_elems, null_scene_elems, marginal, conditional, not_conditional)
		new_loc.updateAppearance(observations, marginal, conditional, not_conditional)
		if index==0:
			fabmap.locations.append([new_loc])
		else:
			fabmap.locations[index].append(new_loc)
		fabmap.build_inverted_index(new_loc, observations, init_scene_elems, null_scene_elems, marginal, conditional, not_conditional)
		print("Location Index    Match Index")
		print("Predicted\t"+str((num+6, index)))
		#print((num, index))
		if (num<5):
			print("Ground Truth\t"+str((num+6, num+1)))
		else:
			print("Ground Truth\t"+str((num+6, 0)))
#		imageA = cv2.imread('data/Images'+str(num+1)+'-1.png')
#		imageB = cv2.imread('data/Images'+str(num+1)+'-2.png')
		matches.append((num, index))

if __name__ == "__main__":
	main()

