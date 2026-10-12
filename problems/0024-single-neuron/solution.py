import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	
	# numpy
	x = np.array(features)
	y = np.array(labels)
	w = np.array(weights)

	p_y = x@w + bias
	p_y = 1 / (1 + np.exp(-p_y))
	probabilities = np.round(p_y, 4).tolist()

	mse = np.round(np.mean((p_y - y)**2), 4)

	return probabilities, mse

	# first principle
	probabilities = []
	for i in range(len(features)):
		feature = features[i]
		y = []
		for f, w in zip(feature, weights):
			y.append(f*w)
		
		y = sum(y) + bias
		y = 1 / (1 + math.exp(-y))
		probabilities.append(round(y, 4))
	
	sum_se = []
	for j in range(len(labels)):
		se = (probabilities
[j] - labels[j])**2
		sum_se.append(se)

	mse = round(sum(sum_se) / len(labels), 4)

	return probabilities, mse