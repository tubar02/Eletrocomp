from __future__ import annotations

import numpy as np

from Códigos.core import solver
from Códigos.utils import io

def main():
	sol = solver.Solver()
	sol.set_eq_dif("Laplace Equation (EDP)")
	sol.set_metodo("Jacobi")
	sol.set_iteracoes(250)
	sol.set_precisao(1e-5)
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

	io.salva_dat(V, "Dados\\potencial.dat")
	io.salva_dat(X, "Dados\\potencial_x.dat")
	io.salva_dat(Y, "Dados\\potencial_y.dat")
	io.salva_dat(Ex, "Dados\\potencial_Ex.dat")
	io.salva_dat(Ey, "Dados\\potencial_Ey.dat")

if __name__ == "__main__":
	main()