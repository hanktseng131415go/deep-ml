import math

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	max_x = max(x)
	exp_x = [math.exp(xi) for xi in x]
	sum_exp_x = sum(exp_x)
	s_x = [exp_xi / sum_exp_x for exp_xi in exp_x]

	out = [[0.]*len(x) for _ in range(len(x))]
	for i in range(len(x)):
		for j in range(len(x)):
			if i == j:
				out[i][j] = s_x[i] * (1 - s_x[j])
			else:
				out[i][j] = -s_x[i] * s_x[j]
	
	return out