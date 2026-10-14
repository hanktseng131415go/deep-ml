import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	scores = np.array(scores)
	max_scores = np.max(scores, axis=-1, keepdims=True)

	x = scores - max_scores

	y = x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))

	return np.round(y, 4)
