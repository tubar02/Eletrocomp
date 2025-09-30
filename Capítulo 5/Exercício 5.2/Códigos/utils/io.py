from __future__ import annotations

from pathlib import Path
import numpy as np

def salva_dat(M: np.ndarray, caminho: Path) -> None:
	n_lin, n_col = M.shape
	with open(caminho, "w", encoding="utf-8") as f:
		f.write(f"{n_lin} {n_col}\n")
		for i in range(n_lin):
			f.write(" ".join(map(str, M[i, :])) + "\n")

def le_dat(caminho: Path) -> np.ndarray:
	with open(caminho, "r", encoding="utf-8") as f:
		n_lin, n_col = map(int, f.readline().split())
		M = np.zeros((n_lin, n_col), dtype=float)
		for i in range(n_lin):
			M[i, :] = list(map(float, f.readline().split()))
	return M