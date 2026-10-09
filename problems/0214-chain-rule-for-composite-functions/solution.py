import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	# Your code here
	if not functions:
		return 1
	
	functions.reverse()
	inputs = [x]
	for op in functions:
		v = inputs[-1]
		if op == 'square':
			inputs.append(v**2)
		elif op == 'sin':
			inputs.append(np.sin(v))
		elif op == 'exp':
			inputs.append(np.exp(v))
		else:
			inputs.append(np.log(v))
	
	grad = 1
	for i, op in enumerate(functions):
		v = inputs[i]
		if op == 'square':
			grad *= (2*v)
		elif op == 'sin':
			grad *= (np.cos(v))
		elif op == 'exp':
			grad *= (np.exp(v))
		else:
			grad *= (1/v)
	
	return float(grad)
