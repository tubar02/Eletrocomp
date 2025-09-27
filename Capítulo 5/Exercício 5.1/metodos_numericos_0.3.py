from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Callable

# ------------------------------------------
# Solver
# ------------------------------------------
class Solver:
	def __init__(self, eq_dif=None, parametros=None, metodo=None, precisao=None, iteracoes=None):
		self.eq_dif: str | None = eq_dif
		self.parametros: dict = parametros or {}
		self.metodo: str | None = metodo
		self.precisao: float | None = precisao
		self.iteracoes: int | None = iteracoes      # também usado como n (pontos/ eixo)

	def set_eq_dif(self, eq_dif: str):                 self.eq_dif = eq_dif
	def set_parametros(self, parametros: dict):        self.parametros = parametros
	def add_parametros(self, parametros: dict):        self.parametros |= parametros
	def set_metodo(self, metodo: str):                 self.metodo = metodo
	def set_precisao(self, precisao: float):           self.precisao = precisao
	def set_iteracoes(self, iteracoes: int):           self.iteracoes = iteracoes

	def solve(self) -> Data:
		dados = DataFactory.create(self)  # escolhe a subclasse certa
		dados.retorno()                   # executa o método numérico
		return dados

# ------------------------------------------
# Data base (abstrata) e fábrica
# ------------------------------------------
class Data(ABC):
	def __init__(self, solver: Solver):
		self.solver = solver		
		self.solved: np.ndarray | None = None

	@abstractmethod
	def retorno(self) -> None:
		pass

class DataFactory:
	@staticmethod
	def create(solver: Solver) -> Data:
		if solver.eq_dif == "Nuclear Decay (EDO)":
			return Nuclear_Decay(solver)
		if solver.eq_dif == "Laplace Equation (EDP)":
			return Laplace_Equation(solver)
		raise ValueError(f"Equação não suportada: {solver.eq_dif}")

# ------------------------------------------
# Nuclear Decay (exemplo 1D)
# ------------------------------------------
class Nuclear_Decay(Data):
	def __init__(self, solver: Solver):
		super().__init__(solver)
		n = solver.iteracoes
		self.solved = np.zeros((2, n), dtype = float)
		self.solved[0, 0] = solver.parametros["NU_0"]

	def retorno(self) -> None:
		if self.solver.metodo != "Euler":
			raise ValueError("Método não implementado para Nuclear Decay.")
		dt  = self.solver.precisao
		tau = self.solver.parametros["tau"]
		for i in range(1, self.solved.shape[1]):
			N_prev = self.solved[0, i-1]
			self.solved[0, i] = N_prev - (N_prev / tau) * dt
			self.solved[1, i] = self.solved[1, i-1] + dt

# ------------------------------------------
# Laplace 2D
# ------------------------------------------
class Laplace_Equation(Data):
	def __init__(self, solver: Solver):
		super().__init__(solver)

		# n pontos por eixo
		n = self.solver.iteracoes

		xi = self.solver.parametros["x_i"]
		xf = self.solver.parametros["x_f"]
		yi = self.solver.parametros["y_i"]
		yf = self.solver.parametros["y_f"]

		x = np.linspace(xi, xf, n, dtype=float)
		y = np.linspace(yi, yf, n, dtype=float)
		X, Y = np.meshgrid(x, y, indexing="xy")
		self.espaco: tuple[np.ndarray, np.ndarray] = (X, Y)

		# Condições de contorno (Dirichlet) via regras "where"
		self.contorno: np.ndarray  = self._build_Contorno()
		Vcc   = self.contorno.astype(float)
		self.fixed = ~np.isnan(Vcc)         # True = ponto travado (tem valor)
		self.V_0   = np.where(self.fixed, self.contorno, 0.0)  # estado inicial

	def retorno(self):
		if self.solver.metodo != "Jacobi":
			raise ValueError("Método não implementado para Laplace Equation.")

		self.solved = self.V_0.copy()
		deltaV = np.infty
		while deltaV > self.solver.precisao:
			self.solved, deltaV = self._atualiza_V()
			print(f"DeltaV = {deltaV:.3e}")

	def _atualiza_V(self):
		V = self.solved
		Vcc = self.contorno
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
	
	def _build_Contorno(self): 
		X, Y = self.espaco
		mascara = np.full(X.shape, np.nan, dtype = float)
		regras: list[dict[str, float | Callable[[np.ndarray, np.ndarray], np.ndarray]]] = self.solver.parametros["cc"]
		for r in regras:
			M = r["where"](X, Y)     # máscara booleana
			V = r["V"]
			mascara[M] = V
		return mascara
	
def salva_dados(matriz: np.ndarray, nome_arq: str):
	with open(nome_arq, "w") as arq:
		n_lin, n_col = matriz.shape
		arq.write(f"{n_lin} {n_col}\n")
		for i in range(matriz.shape[0]):
			linha = " ".join(matriz[i, :].astype(str))
			linha += "\n"
			arq.write(linha)

def main():
	sol = Solver()
	sol.set_eq_dif("Laplace Equation (EDP)")
	sol.set_metodo("Jacobi")
	sol.set_iteracoes(250)
	sol.set_precisao(1e-5)
	sol.set_parametros({"x_i": -1, "x_f": 1, "y_i": -1, "y_f": 1})

	eps = 1e-12
	cc = [
		{"where": lambda X, Y: np.isclose(X, -1.0, atol=eps), "V": -1.0},
		{"where": lambda X, Y: np.isclose(X,  1.0, atol=eps), "V": +1.0},
		# exemplo interno (opcional):
		# {"where": lambda X, Y: (X**2 + Y**2) <= 0.3**2, "V": 0.75},
	]
	sol.add_parametros({"cc": cc})

	dados = sol.solve()         # Data retornado pela fábrica

	V = dados.solved            # matriz (n, n)
	X, Y = dados.espaco

	salva_dados(V, "Dados\\potencial.dat")
	salva_dados(X, "Dados\\espacoX.dat")
	salva_dados(Y, "Dados\\espacoY.dat")

if __name__ == "__main__":
	main()