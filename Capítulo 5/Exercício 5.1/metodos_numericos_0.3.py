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
	def retorno(self) -> None: ...

class DataFactory:
	@staticmethod
	def create(solver: Solver) -> Data:
		if solver.eq_dif == "Nuclear Decay (EDO)":
			return Nuclear_Decay(solver)
		if solver.eq_dif == "Laplace Equation (EDP)":
			return Laplace_Equation(solver)
		raise ValueError(f"Equação não suportada: {solver.eq_dif}")

# ------------------------------------------
# Updater
# ------------------------------------------
class Updater:
	@staticmethod
	def for_ode(method: str):
		m = (method or "").lower()
		if m == "euler":
			return Updater._ode_euler
		raise ValueError(f"Método ODE não suportado: {method}")
	
	@staticmethod
	def for_pde(method: str):
		m = (method or "").lower()
		if m == "jacobi":
			return Updater._pde_jacobi
		raise ValueError(f"Método EDP não suportado: {method}")
	
	# ---------------------- ODE ----------------------
	@staticmethod
	def _ode_euler(problem: ODEProblem, i):
		y, t = problem.ode_arrays()
		dt  = problem.solver.precisao

		y_prev = y[..., i-1] if y.ndim > 1 else y[i-1]
		dydt = problem.ode_rhs(t[i-1], y_prev)
		if y.ndim > 1:
			y[..., i] = y_prev + dt * dydt
		else:
			y[i] = y_prev + dt * dydt
		t[i] = t[i-1] + dt
	
	# ---------------------- PDE ----------------------
class ODEProblem(Data):
	@abstractmethod
	def ode_rhs(self, t: float, y: np.ndarray) -> np.ndarray | float: ...
	@abstractmethod
	def ode_arrays(self) -> tuple[np.ndarray, np.ndarray]: ...

class PDEProblem(Data):
	@abstractmethod
	def pde_arrays(self) -> np.ndarray: ...
	@abstractmethod
	def bc_project(self, U: np.ndarray) -> None: ...
	@abstractmethod
	def stencil(self) -> dict[tuple[int, int], float]: ...

# ------------------------------------------
# Nuclear Decay (exemplo 1D)
# ------------------------------------------
class Nuclear_Decay(ODEProblem):
	def __init__(self, solver: Solver):
		super().__init__(solver)
		n = solver.iteracoes
		self.solved = np.zeros((2, n), dtype = float)
		self.solved[0, 0] = solver.parametros["NU_0"]
		self.solved[0, 1] = solver.parametros.get("t_0", 0)

	def ode_rhs(self, t: float, N: float) -> float:
		tau = self.solver.parametros["tau"]
		return -N / tau

	def ode_arrays(self) -> tuple[np.ndarray, np.ndarray]:
		N_arr = self.solved[0, :]
		t_arr = self.solved[1, :]
		return N_arr, t_arr

	def retorno(self) -> None:
		update = Updater.for_ode(self.solver.metodo)
		for i in range(1, self.solver.iteracoes):
			update(self, i)

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
'''
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
'''

def demo_edo():
	sol = Solver(eq_dif="Nuclear Decay (EDO)", metodo="Euler",
				precisao=0.05, iteracoes=200,
				parametros={"NU_0": 100.0, "tau": 1.0})
	dados = sol.solve()
	N, t = dados.solved[0], dados.solved[1]
	plt.plot(t, N, "-b")
	plt.xlabel("t"); plt.ylabel("N(t)"); plt.title("Decaimento Nuclear — Euler")
	plt.grid(True); plt.show()

def demo_edp():
	n = 200
	eps = 1e-12
	cc = [
		{"where": lambda X, Y: np.isclose(X, -1.0, atol=eps), "V": -1.0},
		{"where": lambda X, Y: np.isclose(X,  1.0, atol=eps), "V": +1.0},
	]
	sol = Solver(eq_dif="Laplace Equation (EDP)", metodo="Jacobi",
				precisao=1e-5, iteracoes=n,
				parametros={"x_i": -1, "x_f": 1, "y_i": -1, "y_f": 1, "cc": cc, "max_iters": 20000})
	dados = sol.solve()
	U = dados.solved
	X, Y = dados.espaco

	plt.imshow(U, origin="lower",
			extent=[X.min(), X.max(), Y.min(), Y.max()],
			aspect="equal", cmap="coolwarm")
	plt.colorbar(label="V")
	plt.title("Laplace 2D — Jacobi (Dirichlet)")
	plt.xlabel("x"); plt.ylabel("y")
	plt.show()

def main():
	demo_edo()
	demo_edp()

if __name__ == "__main__":
	main()