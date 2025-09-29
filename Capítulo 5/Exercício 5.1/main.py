import numpy as np
import matplotlib.pyplot as plt
from functools import wraps

def le_dat(nome_arq: str) -> np.ndarray:
	nome_arq += ".dat"
	with open(nome_arq) as arq:
		n_lin, n_col = [int(i) for i in arq.readline().split(" ")]
		matriz = np.zeros((n_lin, n_col), dtype = float)
		for i in range(n_lin):
			lista = [float(v) for v in arq.readline().split(" ")]
			matriz[i, :] = lista
	return matriz

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

@plotter(3)
def plota_ex5_1(V: np.ndarray, X: np.ndarray, Y: np.ndarray, opcao: int = 1):
    if opcao == 1:
        im = plt.imshow(V, origin="lower", aspect="equal", cmap="coolwarm")
        plt.colorbar(im, label="V")
    elif opcao == 2:
        cs = plt.contourf(X, Y, V, levels=20, cmap="coolwarm")
        plt.colorbar(cs, label="V")
    else:
        qc = plt.pcolormesh(X, Y, V, shading="auto", cmap="coolwarm")
        plt.colorbar(qc, label="V")

def main():
	pasta = "Dados\\"
	V = le_dat(pasta + "potencial_circ")
	X = le_dat(pasta + "espacoX")
	Y = le_dat(pasta + "espacoY")

	titulos = ["Potencial V", "Linhas Equipotenciais", "Potencial V"]
	
	for i, titulo in enumerate(titulos):
		plota_ex5_1(V, X, Y, i + 1, titulo = titulo)

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

if __name__ == "__main__":
	main()