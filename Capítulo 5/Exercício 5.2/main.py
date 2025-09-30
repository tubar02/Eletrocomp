import numpy as np
import matplotlib.pyplot as plt
from functools import wraps

from Códigos.utils import io, paths

def plotter(num_opcoes: int):
	def deco(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			# extrai/valida 'opcao'
			opcao = kwargs.get("opcao", 1)
			if not (1 <= opcao <= num_opcoes):
				raise ValueError(f"Opção deve ser um número inteiro de 1 a {num_opcoes}.")

			# extrai título/labels se vierem (ou usa defaults)
			titulo = kwargs.pop("titulo", "")
			eixo_x = kwargs.pop("eixo_x", "x")
			eixo_y = kwargs.pop("eixo_y", "y")

			# chama a função original
			out = func(*args, **kwargs)

			# aplica título/labels e mostra
			plt.title(titulo)
			plt.xlabel(eixo_x)
			plt.ylabel(eixo_y)
			plt.show()
			return out
		return wrapper
	return deco

def plota_superficie(X: np.ndarray, Y: np.ndarray, V: np.ndarray, label):
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection="3d")

	# cria a superfície
	surf = ax.plot_surface(X, Y, V, cmap="coolwarm", edgecolor="none")

	# adiciona barra de cores
	fig.colorbar(surf, shrink=0.5, aspect=10, label=label)
	ax.set_zlabel("V(x,y)")

def plot_campo_vetorial(Vx, Vy, X, Y, *, step=8):
	fig, ax = plt.subplots(figsize=(7,6))
	Q = ax.quiver(X[::step, ::step], Y[::step, ::step],	Vx[::step, ::step], Vy[::step, ::step],	
				pivot="mid", angles="xy", scale_units="xy", scale=None)
	ax.set_aspect("equal")

@plotter(4)
def plota_ex5_1(V: np.ndarray, X: np.ndarray, Y: np.ndarray, Ex: np.ndarray, Ey: np.ndarray, 
				*, titulo = "Potencial V", opcao: int = 1, label = "V"):
	if opcao == 1:
		# imagem 2D do potencial
		extent = [X.min(), X.max(), Y.min(), Y.max()]
		fig, ax = plt.subplots(figsize=(7, 6))
		im = ax.imshow(V, origin="lower", aspect="equal", cmap="coolwarm", extent=extent)
		plt.colorbar(im, ax=ax, label=label)
	elif opcao == 2:
		# equipotenciais
		fig, ax = plt.subplots(figsize=(7, 6))
		cs = ax.contourf(X, Y, V, levels=20, cmap="coolwarm")
		plt.colorbar(cs, ax=ax, label=label)
	elif opcao == 3:
		# superfície 3D
		plota_superficie(X, Y, V, label=label)
	elif opcao == 4:
		# campo elétrico
		plot_campo_vetorial(Ex, Ey, X, Y, step=12)

def main():
	paths.init_project_tree()

	V = io.le_dat(paths.out_file("potencial"))
	X = io.le_dat(paths.out_file("potencial_x"))
	Y = io.le_dat(paths.out_file("potencial_y"))
	Ex = io.le_dat(paths.out_file("potencial_Ex"))
	Ey = io.le_dat(paths.out_file("potencial_Ey"))

	titulos = ["Potencial V", "Linhas Equipotenciais", "Potencial V, visão 3D", "Campo Elétrico"]
	
	for i, titulo in enumerate(titulos):
		plota_ex5_1(V, X, Y, Ex, Ey, opcao = i + 1, titulo = titulo)
	'''
	certo = True
	eps = 1e-5
	for i in range(V.shape[0]):
		for j in range (V.shape[1]):
			try:
				soma_viz = V[i - 1, j] + V[i + 1, j] + V[i, j - 1] + V[i, j + 1]
				if not np.isclose(V, soma_viz, atol=eps):
					certo = False
			except:
				pass
	print(certo)
	'''

if __name__ == "__main__":
	main()