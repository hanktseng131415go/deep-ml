from typing import Callable
import numpy as np

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Hessian matrix of function f at the given point using finite differences.
	
	Args:
		f: A scalar function that takes a list of floats and returns a float
		point: The point at which to compute the Hessian (list of coordinates)
		h: Step size for finite differences (default: 1e-5)
		
	Returns:
		The Hessian matrix as a list of lists (n x n where n = len(point))
	"""
	# Your code here
	n = len(point)
	H = np.zeros((n, n))
	for i in range(n):
		for j in range(i, n):
			if i == j:
				x_p = point.copy()
				x_p[i] += h

				x_m = point.copy()
				x_m[i] -= h

				H[i, j] = (f(x_p) - 2*f(point) + f(x_m)) / (h**2)
			else:
				x_ijp = point.copy()
				x_ijp[i] += h; x_ijp[j] += h

				x_ipjm = point.copy()
				x_ipjm[i] += h; x_ipjm[j] -= h

				x_imjp = point.copy()
				x_imjp[i] -= h; x_imjp[j] += h

				x_imjm = point.copy()
				x_imjm[i] -= h; x_imjm[j] -= h

				h_tmp = (f(x_ijp) - f(x_ipjm) - f(x_imjp) + f(x_imjm)) / (4*h**2)

				H[i][j] = H[j][i] = h_tmp
	
	return H.tolist()
				