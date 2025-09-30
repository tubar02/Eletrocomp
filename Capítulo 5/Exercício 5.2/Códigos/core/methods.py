from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Códigos.core import solver

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
	def _ode_euler(problem: solver.ODEProblem, i: int):
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
	def _pde_jacobi(problem: solver.PDEProblem):
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