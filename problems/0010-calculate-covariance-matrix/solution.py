import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	v = np.asarray(vectors)
	n_features, n_obs = v.shape
	v_mean = np.mean(v, axis=1, keepdims=True)
	v_diff = v - v_mean
	if n_obs > 1:
		cov_m = (v_diff @ v_diff.T) / (n_obs - 1)
	else:
		cov_m = np.zeros(n_features, n_features)
	
	return cov_m