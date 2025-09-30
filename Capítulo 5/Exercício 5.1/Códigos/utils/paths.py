from __future__ import annotations

from pathlib import Path
from datetime import datetime

# -----------------------------------------------------------
# Descoberta da raiz do projeto
# -----------------------------------------------------------
def _discover_root(marker_names: tuple[str, ...] = ("pyproject.toml", ".git", "main.py")) -> Path:
	"""
	Sobe diretórios a partir deste arquivo até encontrar um 'marcador'
	(pyproject.toml, .git, etc.). Se não encontrar, usa o cwd.
	"""
	here = Path(__file__).resolve()
	for parent in [here, *here.parents]:
		for m in marker_names:
			if (parent / m).exists():
				return parent
	return Path.cwd()

ROOT: Path =  _discover_root()

# -----------------------------------------------------------
# Pastas padrão
# -----------------------------------------------------------
SRC_DIR: Path   = ROOT / "Códigos"
DATA_DIR: Path  = ROOT / "Dados"
IN_DIR: Path    = DATA_DIR / "inputs"
OUT_DIR: Path   = DATA_DIR / "outputs"
LOGS_DIR: Path  = ROOT / "logs"
FIGS_DIR: Path  = ROOT / "Gráficos"

# Diretório de execução (timestamp)
_RUN_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
RUNS_DIR: Path = ROOT / "runs" / _RUN_STAMP

# -----------------------------------------------------------
# Utils
# -----------------------------------------------------------
def ensure_dir(p: Path | str) -> Path:
	"""Garante que o diretório exista e retorna Path."""
	path = Path(p)
	path.mkdir(parents=True, exist_ok=True)
	return path

def init_project_tree() -> None:
	"""Cria a árvore mínima de diretórios do projeto."""
	for d in (SRC_DIR, DATA_DIR, IN_DIR, OUT_DIR, LOGS_DIR, FIGS_DIR, RUNS_DIR):
		ensure_dir(d)

def out_file(name: str, base_dir: Path = OUT_DIR, ext: str = ".dat") -> Path:
	"""
	Caminho para arquivos de saída de dados (.dat, .npy, .png, etc.).
	Se name já tiver extensão, respeita; senão, aplica 'ext'.
	"""
	stem = Path(name)
	if stem.suffix:
		return base_dir / stem.name
	return base_dir / (stem.name + ext)

# accessors
def src_path() -> Path:  return SRC_DIR
def data_path() -> Path: return DATA_DIR
def in_path() -> Path:   return IN_DIR
def out_path() -> Path:  return OUT_DIR
def log_path() -> Path:  return LOGS_DIR
def figs_path() -> Path: return FIGS_DIR
def runs_path() -> Path: return RUNS_DIR

def run_subdir(name: str) -> Path:
	"""Cria subpasta dentro de runs/<timestamp>/ para agrupar artefatos do caso."""
	return ensure_dir(RUNS_DIR / name)	

# Conveniência: imprimir paths principais quando rodar standalone
if __name__ == "__main__":
	init_project_tree()
	print("ROOT      :", ROOT)
	print("DATA_DIR  :", DATA_DIR)
	print("OUT_DIR   :", OUT_DIR)
	print("LOGS_DIR  :", LOGS_DIR)
	print("FIGS_DIR  :", FIGS_DIR)
	print("RUNS_DIR  :", RUNS_DIR)