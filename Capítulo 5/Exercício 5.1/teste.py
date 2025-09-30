from __future__ import annotations

import numpy as np

from Códigos.core import solver
from Códigos.utils import io, paths

def main():
	sol = solver.Solver()
	sol.set_eq_dif("Laplace Equation (EDP)")
	sol.set_metodo("Jacobi")
	sol.set_iteracoes(250)
	sol.set_precisao(1e-7)
	sol.set_parametros({"x_i": -1, "x_f": 1, "y_i": -1, "y_f": 1})

	eps = 1e-12

	def retangulo(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
		mat = np.empty(X.shape, dtype=bool)
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				if np.abs(X[i, j]) <= 0.3 and np.abs(Y[i, j]) <= 0.3:
					mat[i, j] = True
				else:
					mat[i, j] = False
		return mat

	cc = [
		{"where": retangulo, "V": 1.0},
		{"where": lambda X, Y: np.isclose(X, -1, atol=eps), "V": 0},
		{"where": lambda X, Y: np.isclose(X, 1, atol=eps), "V": 0},
		{"where": lambda X, Y: np.isclose(Y, -1, atol=eps), "V": 0},
		{"where": lambda X, Y: np.isclose(Y, 1, atol=eps), "V": 0}
	]
	sol.add_parametros({"cc": cc})

	dados = sol.solve()         # Data retornado pela fábrica

	V = dados.solved            # matriz (n, n)
	X, Y = dados.espaco
	Ex, Ey = dados.Ex, dados.Ey

	io.salva_dat(V, paths.out_file("potencial"))
	io.salva_dat(X, paths.out_file("potencial_x"))
	io.salva_dat(Y, paths.out_file("potencial_y"))
	io.salva_dat(Ex, paths.out_file("potencial_Ex"))
	io.salva_dat(Ey, paths.out_file("potencial_Ey"))

if __name__ == "__main__":
	main()