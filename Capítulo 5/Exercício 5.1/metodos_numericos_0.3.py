import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod

class Solver():
	def __init__(self, eq_dif = None, parametros = None, metodo = None, precisao = None, iteracoes = None):
		self.eq_dif: str = eq_dif
		self.parametros: dict = parametros
		self.metodo: str = metodo
		self.precisao: float = precisao
		self.iteracoes: int = iteracoes

	def set_eq_dif(self, eq_dif: str):
		self.eq_dif = eq_dif

	def set_parametros(self, parametros: dict):
		self.parametros = parametros

	def add_parametros(self, parametros: dict):
		self.parametros = self.parametros | parametros

	def set_metodo(self, metodo: str):
		self.metodo = metodo

	def set_precisao(self, precisao: int | float):
		self.precisao = precisao

	def set_iteracoes(self, iteracoes: int):
		self.iteracoes = iteracoes

	def solve(self) -> np.ndarray:
		self.dados = Data(self)
		self.dados = self.dados.calcula()
		return self.dados

class Data():
	def __init__(self, solver: Solver):
		self.solver = solver			

	def calcula(self):
		if self.solver.eq_dif == "Nuclear Decay (EDO)":
			dados = Nuclear_Decay(self.solver)

		elif self.solver.eq_dif == "Laplace Equation (EDP)":
			dados = Laplace_Equation(self.solver)

		dados.retorno()
		self.solved = dados.solved
		return dados
	
	@abstractmethod
	def retorno(self):
		pass

class Nuclear_Decay(Data):
	def __init__(self, solver: Solver):
		super().__init__(solver)

		self.solved = np.zeros((2, self.solver.iteracoes))
		self.solved[0][0] = self.solver.parametros["NU_0"]

	def retorno(self):
		if self.solver.metodo == "Euler":
			for i in range(1, self.solver.iteracoes):
				self.solved[0][i] = self.solved[0][i - 1] - (self.solved[0][i - 1] / self.solver.parametros["tau"]) * self.solver.precisao
				self.solved[1][i] = self.solved[1][i - 1] + self.solver.precisao

class Laplace_Equation(Data):
	def __init__(self, solver: Solver):
		super().__init__(solver)

		x = np.linspace(self.solver.parametros["x_i"], self.solver.parametros["x_f"], self.solver.iteracoes)
		y = np.linspace(self.solver.parametros["y_i"], self.solver.parametros["y_f"], self.solver.iteracoes)
		self.espaco = np.meshgrid(x, y)
		self.contorno = Laplace_Contorno(self)
		self.fixed = ~np.isnan(self.contorno.mascara)
		self.V_0 = np.where(self.fixed, self.contorno.mascara, 0)

	def retorno(self):
		if self.solver.metodo == "Jacobi":
			self.solved = self.V_0.copy()

			deltaV = np.infty
			while deltaV > self.solver.precisao:
				self.solved, deltaV = Laplace_Equation.atualiza_V(self)
				print(f"DeltaV = {deltaV:.3e}")

	@staticmethod
	def atualiza_V(self):
		V = self.solved
		Vcc = self.contorno.mascara
		fixed = self.fixed

		# Soma (S) e contagem (C) de vizinhos por fatiamento
		S = np.zeros_like(V)
		C = np.zeros_like(V)

		# cima
		S[1:,  :] += V[:-1, :]
		C[1:,  :] += 1
		# baixo
		S[:-1, :] += V[1:,  :]
		C[:-1, :] += 1
		# esquerda
		S[:, 1:]  += V[:, :-1]
		C[:, 1:]  += 1
		# direita
		S[:, :-1] += V[:, 1:]
		C[:, :-1] += 1

		Vnew = S / C                 # média dos vizinhos (2, 3 ou 4)
		np.copyto(Vnew, Vcc, where = fixed)  # reimpõe Dirichlet

		deltaV = np.mean(np.abs(Vnew - V))  # precisão por sítio
		return Vnew, deltaV

class Laplace_Contorno():
	"""
    condicoes: lista de dicts com {"where": Callable[[np.ndarray, np.ndarray], np.ndarray], "V": float}
    """
	def __init__(self, dados: Laplace_Equation):
		condicoes = dados.solver.parametros["cc"]
		X, Y = dados.espaco
		self.shape = X.shape
		self.mascara = np.full(self.shape, np.nan)
		for condicao in condicoes:
			mat_cond = condicao["where"](X, Y)
			self.mascara[mat_cond] = float(condicao["V"])

def main():
	soluciona = Solver()
	soluciona.set_eq_dif("Laplace Equation (EDP)")
	soluciona.set_parametros({"x_i": -1, "x_f": 1, "y_i": -1, "y_f": 1})

	eps = 1e-12
	dicio1 = {"where": lambda X, Y: np.isclose(X, -1, atol = eps), "V": -1}
	dicio2 = {"where": lambda X, Y: np.isclose(X, 1, atol = eps), "V": 1}
	list_dic = [dicio1, dicio2]

	soluciona.add_parametros({"cc": list_dic})

	soluciona.set_metodo("Jacobi")
	soluciona.set_precisao(1e-5)
	soluciona.set_iteracoes(250)

	dados = soluciona.solve()

	V = dados.solved

	plt.imshow(V, origin = "lower", aspect = "equal")
	plt.colorbar(label = "V")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Potencial V calculado")
	plt.show()
	
	X, Y = dados.espaco
	cs = plt.contourf(X, Y, V, levels=20, cmap="coolwarm")
	plt.colorbar(cs, label="Potencial V")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Linhas equipotenciais")
	plt.show()

	plt.pcolormesh(X, Y, V, shading="auto")
	plt.colorbar(label="Potencial V")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Potencial V calculado por Jacobi")
	plt.show()


if __name__ == "__main__":
	main()