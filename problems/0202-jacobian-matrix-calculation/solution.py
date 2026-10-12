import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Your code here
	j_list = []
	for i in range(len(x)):
		x_p = x.copy()
		x_p[i] += h

		x_m = x.copy()
		x_m[i] -= h

		f_p = np.array(f(x_p))
		f_m = np.array(f(x_m))

		j = (f_p - f_m) / (2*h)
		j_list.append(j)
	
	j_np = np.array(j_list).T

	return j_np.tolist()

