import numpy as np
import matplotlib.pyplot as plt

class Solver():
	def __init__(self, eq_dif = None, parametros = None, metodo = None, precisao = None, iteracoes = None):
		self.eq_dif = eq_dif
		self.parametros = parametros
		self.metodo = metodo
		self.precisao = precisao
		self.iteracoes = iteracoes

	def set_eq_dif(self, eq_dif):
		self.eq_dif = eq_dif

	def set_parametros(self, parametros):
		self.parametros = parametros

	def add_parametros(self, parametros):
		self.parametros = self.parametros | parametros

	def set_metodo(self, metodo):
		self.metodo = metodo

	def set_precisao(self, precisao):
		self.precisao = precisao

	def set_iteracoes(self, iteracoes):
		self.iteracoes = iteracoes

	def solve(self):
		dados = Data(self)
		dados.calcula()
		return dados.solved

class Data():
	def __init__(self, solver):
		self.solver = solver

		if solver.eq_dif == "Nuclear Decay (EDO)":
			self.solved = np.zeros((2, self.solver.iteracoes))
			self.solved[0][0] = self.solver.parametros["NU_0"]

		elif solver.eq_dif == "Laplace Equation (EDP)":
			x = np.linspace(self.solver.parametros["x_i"], self.solver.parametros["x_f"], self.solver.precisao)
			y = np.linspace(self.solver.parametros["y_i"], self.solver.parametros["y_f"], self.solver.precisao)
			self.espaco = np.meshgrid(x, y)

	def calcula(self):
		if self.solver.eq_dif == "Nuclear Decay (EDO)":
			self.nuclear_decay()

		elif self.solver.eq_dif == "Laplace Equation (EDP)":
			self.laplace_equation()

	def nuclear_decay(self):
		if self.solver.metodo == "Euler":
			for i in range(1, self.solver.iteracoes):
				self.solved[0][i] = self.solved[0][i - 1] - (self.solved[0][i - 1] / self.solver.parametros["tau"]) * self.solver.precisao
				self.solved[1][i] = self.solved[1][i - 1] + self.solver.precisao

	def laplace_equation(self):
		if self.solver.metodo == "Jacobi":
			def atualiza_V():
				delta_V = 0


def main():
	soluciona = Solver()
	soluciona.set_eq_dif("Laplace Equation (EDP)")

	borda = 1

	soluciona.set_parametros({"x_i": -1, "x_f": 1, "y_i": -1, "y_f": 1})
	soluciona.set_metodo("Jacobi")
	soluciona.set_precisao(10)
	soluciona.set_iteracoes(100)

	solved = soluciona.solve()
	NU, t = solved[0], solved[1]

	plt.plot(t, NU, "b-", label="Decaimento Nuclear (Euler)")
	plt.xlabel("Tempo")
	plt.ylabel("Número de núcleos")
	plt.title("Decaimento Nuclear via Euler")
	plt.legend()
	plt.grid(True)
	plt.show()



if __name__ == "__main__":
	main()