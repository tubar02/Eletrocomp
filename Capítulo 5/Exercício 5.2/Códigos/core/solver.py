from __future__ import annotations

from typing import Any
from abc import ABC, abstractmethod

# ------------------------------------------
# Solver
# ------------------------------------------
class Solver:
	def __init__(self, eq_dif=None, parametros=None, metodo=None, precisao=None, iteracoes=None):
		self.eq_dif: str | None = eq_dif
		self.parametros: dict[str, Any] = parametros or {}
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
		self.solved = None

	@abstractmethod
	def retorno(self) -> None: ...

class DataFactory:
	@staticmethod
	def create(solver: Solver) -> Data:
		from Códigos.core import problems
		if solver.eq_dif == "Nuclear Decay (EDO)":
			return problems.Nuclear_Decay(solver)
		if solver.eq_dif == "Laplace Equation (EDP)":
			return problems.Laplace_Equation(solver)
		raise ValueError(f"Equação não suportada: {solver.eq_dif}")
	
class ODEProblem(Data):
	@abstractmethod
	def ode_rhs(self, t, y): ...
	@abstractmethod
	def ode_arrays(self): ...

class PDEProblem(Data):
	@abstractmethod
	def pde_arrays(self): ...
	@abstractmethod
	def bc_project(self, U) -> None: ...
	@abstractmethod
	def stencil(self): ...