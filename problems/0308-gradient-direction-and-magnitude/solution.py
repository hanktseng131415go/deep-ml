import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	out = {}
	gradient = np.array(gradient) 
	magnitude = sum(gradient**2)**0.5
	direction = gradient / magnitude if magnitude > 0. else np.zeros(gradient.shape)
	descent_direction = -direction

	out['magnitude'] = magnitude
	out['direction'] = direction
	out['descent_direction'] = descent_direction

	return out