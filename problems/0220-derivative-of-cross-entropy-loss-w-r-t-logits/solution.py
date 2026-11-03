import math

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""

	max_x = max(logits)
	soft_x = [math.exp(x - max_x) for x in logits]
	sum_x = sum(soft_x)
	prob = [x/sum_x for x in soft_x]

	grad = prob.copy()
	grad[target] -= 1.0
	return grad

	max_x = max(logits)
	x = [math.exp(i - max_x) for i in logits]
	sum_x = sum(x)
	prob = [i/sum_x for i in x]

	grad = prob.copy()
	grad[target] -= 1.0

	return grad
