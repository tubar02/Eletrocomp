from __future__ import annotations

import numpy as np

from Códigos.core import solver

def salva_dados(matriz: np.ndarray, nome_arq: str):
	with open(nome_arq, "w") as arq:
		n_lin, n_col = matriz.shape
		arq.write(f"{n_lin} {n_col}\n")
		for i in range(matriz.shape[0]):
			linha = " ".join(matriz[i, :].astype(str))
			linha += "\n"
			arq.write(linha)

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

	salva_dados(V, "Dados\\a.dat")
	salva_dados(X, "Dados\\aX.dat")
	salva_dados(Y, "Dados\\aY.dat")

if __name__ == "__main__":
	main()