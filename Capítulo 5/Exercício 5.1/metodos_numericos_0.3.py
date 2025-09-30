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
	def _ode_euler(problem: ODEProblem, i: int):
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
	@staticmethod
	def _pde_jacobi(problem: PDEProblem):
		"""
		Jacobi genérico guiado por 'stencil' (sem padding, sem wrap).
		Requer do problema:
		- pde_arrays() -> U (2D)
		- bc_project(U) -> reimpõe CC (ex.: Dirichlet)
		- stencil() -> dict[(di,dj)] = peso  (ex.: Laplace 5-pontos)
		- rhs() opcional (b), default 0
		Resolve A U = b onde A vem do stencil.
		"""
		U = problem.pde_arrays()
		st = problem.stencil()  # ex.: {(+1,0):1, (-1,0):1, (0,+1):1, (0,-1):1, (0,0):-4}
		b = problem.rhs() if hasattr(problem, "rhs") else np.zeros_like(U)

		diag = st[(0, 0)]
		neighbors = [(shift, w) for shift, w in st.items() if shift != (0, 0)]

		S = np.zeros_like(U)

		# acumula w * U deslocado:
		for (di, dj), w in neighbors:
			if di > 0:
				S[di:,  :] += w * U[:-di, :]
			elif di < 0:
				S[:di,  :] += w * U[-di:, :]
			if dj > 0:
				S[:, dj:] += w * U[:, :-dj]
			elif dj < 0:
				S[:, :dj] += w * U[:, -dj:]

		Unew = (b - S) / diag   # isolando a diagonal
		problem.bc_project(Unew)

		delta = np.mean(np.abs(Unew - U))
		problem.solved = Unew
		return delta
	
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
class Laplace_Equation(PDEProblem):
	def __init__(self, solver: Solver):
		super().__init__(solver)
		n = solver.iteracoes
		xi, xf = solver.parametros["x_i"], solver.parametros["x_f"]
		yi, yf = solver.parametros["y_i"], solver.parametros["y_f"]

		x = np.linspace(xi, xf, n, dtype=float)
		y = np.linspace(yi, yf, n, dtype=float)
		X, Y = np.meshgrid(x, y, indexing="xy")
		self.espaco = (X, Y)

		# máscara
		self.Vcc = np.full(X.shape, np.nan, dtype=float)
		for rule in solver.parametros["cc"]:
			M = rule["where"](X, Y)   # máscara booleana
			self.Vcc[M] = rule["V"]

		self.fixed = ~np.isnan(self.Vcc)
		self.Vcc_filled = np.where(self.fixed, self.Vcc, 0.0)

		# estado inicial:
		self.solved = self.Vcc_filled.copy()

	def pde_arrays(self) -> np.ndarray:
		return self.solved

	def bc_project(self, U: np.ndarray) -> None:
		# reimpõe Dirichlet
		np.copyto(U, self.Vcc, where=self.fixed)

	def stencil(self) -> dict[tuple[int, int], float]:
		# Laplaciano 5-pontos homogêneo: +1 nos vizinhos cardeais, -4 no ponto
		return {(+1, 0): 1.0, (-1, 0): 1.0, (0, +1): 1.0, (0, -1): 1.0, (0, 0): -4.0}

	def retorno(self) -> None:
		update = Updater.for_pde(self.solver.metodo) 
		deltaV = np.inf
		while deltaV > self.solver.precisao:
			deltaV = update(self)
			print(f"DeltaV = {deltaV:.3e}")
	
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

	salva_dados(V, "Dados\\potencial_ret.dat")
	salva_dados(X, "Dados\\espacoX.dat")
	salva_dados(Y, "Dados\\espacoY.dat")

if __name__ == "__main__":
	main()