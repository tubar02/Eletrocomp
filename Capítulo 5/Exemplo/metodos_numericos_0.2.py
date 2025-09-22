import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod

class Solver():
	def __init__(self, eq_dif = None, parametros = None, metodo = None, precisao = None, iteracoes = None):
		self.eq_dif: str = eq_dif
		self.parametros: dict = parametros
		self.metodo: str = metodo
		self.precisao: int | float = precisao
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
		dados = Data(self)
		dados.calcula()
		return dados.solved

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

		x = np.linspace(self.solver.parametros["x_i"], self.solver.parametros["x_f"], self.solver.precisao)
		y = np.linspace(self.solver.parametros["y_i"], self.solver.parametros["y_f"], self.solver.precisao)
		self.espaco = np.meshgrid(x, y)
		self.contorno = Laplace_Contorno(self)

		V_0 = np.zeros(self.espaco[0].shape, dtype = float)
		for (pos, _) in np.ndenumerate(V_0):
			if not np.isnan(self.contorno.mascara[pos]):
				V_0[pos] = self.contorno.mascara[pos]

	def retorno(self):
		if self.solver.metodo == "Jacobi":
			pass

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
	soluciona.set_precisao(50)
	soluciona.set_iteracoes(100)

	solved = soluciona.solve()

	'''
	NU, t = solved[0], solved[1]

	plt.plot(t, NU, "b-", label="Decaimento Nuclear (Euler)")
	plt.xlabel("Tempo")
	plt.ylabel("Número de núcleos")
	plt.title("Decaimento Nuclear via Euler")
	plt.legend()
	plt.grid(True)
	plt.show()
	'''

if __name__ == "__main__":
	main()