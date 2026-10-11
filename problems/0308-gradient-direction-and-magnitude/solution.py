import torch

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient (float)
		- direction: Unit vector (torch.Tensor) in direction of steepest ascent
		- descent_direction: Unit vector (torch.Tensor) in direction of steepest descent
	"""
	# Your code here
	out = {}
	
	grad = torch.tensor(gradient)
	magnitude = grad.norm()
	direction = grad / magnitude if magnitude > 0 else torch.zeros(grad.shape)
	descent_direction = -direction

	out['magnitude'] = magnitude
	out['direction'] = direction
	out['descent_direction'] = descent_direction

	return out