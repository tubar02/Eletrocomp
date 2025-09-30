import numpy as np

from Códigos.core import solver, methods

# ------------------------------------------
# Nuclear Decay (exemplo 1D)
# ------------------------------------------
class Nuclear_Decay(solver.ODEProblem):
	def __init__(self, solver: solver.Solver):
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
		update = methods.Updater.for_ode(self.solver.metodo)
		for i in range(1, self.solver.iteracoes):
			update(self, i)

# ------------------------------------------
# Laplace 2D
# ------------------------------------------
class Laplace_Equation(solver.PDEProblem):
	def __init__(self, solver: solver.Solver):
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
		update = methods	.Updater.for_pde(self.solver.metodo) 
		deltaV = np.inf
		while deltaV > self.solver.precisao:
			deltaV = update(self)
			print(f"DeltaV = {deltaV:.3e}")