"""
gpu_bootstrap.py — Preparação de GPU na primeira execução (app empacotado).

Objetivo: o instalador fica pequeno (build GPU, sem os ~2 GB de CUDA embutidos).
Na 1ª execução, se a máquina tiver placa NVIDIA, baixamos o runtime CUDA/cuDNN
direto do PyPI (hospedagem oficial, gratuita) e extraímos numa pasta local. Em
máquinas sem GPU, nada é baixado e o app roda em CPU.

Uso típico (antes de importar onnxruntime/rtmlib):

    from gpu_bootstrap import ensure_gpu_ready, cuda_dir
    ensure_gpu_ready(ask=True)      # detecta GPU, pergunta, baixa se preciso
    # depois, no _enable_cuda_dlls(), inclua cuda_dir() no PATH

Este módulo não depende de nada além da stdlib (urllib, zipfile, ctypes).
"""
import os
import sys
import json
import shutil
import zipfile
import subprocess
import urllib.request

# Versões exatas que casam com onnxruntime-gpu 1.22 (build CUDA 12) e suportam Blackwell.
CUDA_WHEELS = [
    ("nvidia-cudnn-cu12",        "9.24.0.43"),
    ("nvidia-cublas-cu12",       "12.9.2.10"),
    ("nvidia-cuda-runtime-cu12", "12.9.79"),
    ("nvidia-cufft-cu12",        "11.4.1.4"),
    ("nvidia-curand-cu12",       "10.3.10.19"),
    ("nvidia-cuda-nvrtc-cu12",   "12.9.86"),
    ("nvidia-nvjitlink-cu12",    "12.9.86"),
]

# DLL usada como "sentinela" para saber se o runtime já está presente.
_SENTINEL_DLL = "cudnn64_9.dll"


def app_data_dir():
    """Pasta persistente e gravável para o runtime CUDA baixado."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "AnaliseMarcha", "cuda_runtime")
    return d


def cuda_dir():
    """Pasta (flat) onde ficam as DLLs do CUDA baixado. Pode não existir ainda."""
    return os.path.join(app_data_dir(), "bin")


def has_nvidia_gpu():
    """True se houver GPU NVIDIA COM driver (via nvidia-smi, que vem no driver)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        # Caminho padrão do driver, caso não esteja no PATH.
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                            "System32", "nvidia-smi.exe")
        exe = cand if os.path.exists(cand) else None
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, timeout=15,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.returncode == 0 and b"GPU" in out.stdout
    except Exception:
        return False


def cuda_runtime_present():
    """True se as DLLs do CUDA já foram baixadas antes (sentinela existe)."""
    return os.path.exists(os.path.join(cuda_dir(), _SENTINEL_DLL))


def _win_wheel_url(pkg, version):
    """Descobre a URL do wheel win_amd64 no PyPI (hospedagem oficial)."""
    api = f"https://pypi.org/pypi/{pkg}/{version}/json"
    with urllib.request.urlopen(api, timeout=30) as r:
        meta = json.load(r)
    for f in meta.get("urls", []):
        fn = f.get("filename", "")
        if fn.endswith(".whl") and ("win_amd64" in fn or "win-amd64" in fn):
            return f["url"], f.get("size", 0)
    # Alguns wheels nvidia são "py3-none-win_amd64"; fallback: primeiro .whl win.
    for f in meta.get("urls", []):
        if f.get("filename", "").endswith(".whl") and "win" in f["filename"]:
            return f["url"], f.get("size", 0)
    raise RuntimeError(f"Wheel win_amd64 não encontrado para {pkg}=={version}")


def _download(url, dest, progress_cb=None, label=""):
    with urllib.request.urlopen(url, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress_cb and total:
                    progress_cb(label, done, total)


def _extract_dlls(whl_path, target_bin):
    """Extrai todas as .dll de dentro de pastas bin/ do wheel para target_bin (flat)."""
    with zipfile.ZipFile(whl_path) as z:
        for name in z.namelist():
            low = name.lower()
            if low.endswith(".dll") and "/bin/" in low.replace("\\", "/"):
                data = z.read(name)
                out = os.path.join(target_bin, os.path.basename(name))
                with open(out, "wb") as f:
                    f.write(data)


def download_cuda_runtime(progress_cb=None):
    """Baixa e extrai o runtime CUDA do PyPI para cuda_dir(). Retorna o caminho."""
    target = cuda_dir()
    os.makedirs(target, exist_ok=True)
    tmp = os.path.join(app_data_dir(), "_tmp")
    os.makedirs(tmp, exist_ok=True)
    try:
        for i, (pkg, ver) in enumerate(CUDA_WHEELS, 1):
            url, _size = _win_wheel_url(pkg, ver)
            whl = os.path.join(tmp, os.path.basename(url.split("?")[0]))
            _download(url, whl, progress_cb=progress_cb, label=f"[{i}/{len(CUDA_WHEELS)}] {pkg}")
            _extract_dlls(whl, target)
            try:
                os.remove(whl)
            except OSError:
                pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return target


def register_cuda_dir():
    """Coloca a pasta do CUDA baixado no PATH (para onnxruntime achar cuDNN/cuBLAS)."""
    d = cuda_dir()
    if os.path.isdir(d):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
        return True
    return False


def ensure_gpu_ready(ask=True):
    """
    Fluxo de 1ª execução. Retorna 'cuda' se o runtime está pronto/foi baixado,
    ou 'cpu' se não há GPU ou o usuário recusou.

    ask=True mostra uma janela tkinter perguntando antes de baixar (~1,5 GB).
    ask=False baixa automaticamente quando há GPU (sem perguntar).
    """
    if cuda_runtime_present():
        register_cuda_dir()
        return "cuda"
    if not has_nvidia_gpu():
        return "cpu"

    if ask and not _ask_user_download():
        return "cpu"

    # Baixa com uma barrinha de progresso (se tkinter disponível), senão no console.
    try:
        _download_with_progress_ui()
    except Exception as e:
        print(f"[AVISO] Falha ao baixar runtime CUDA: {e}\n[INFO] Seguindo em CPU.")
        return "cpu"

    if cuda_runtime_present():
        register_cuda_dir()
        return "cuda"
    return "cpu"


# ------------------------- UI opcional (tkinter) -------------------------
def _ask_user_download():
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk._default_root or tk.Tk()
        hide = root is not None and not tk._default_root
        try:
            root.withdraw()
        except Exception:
            pass
        ans = messagebox.askyesno(
            "Aceleração por GPU",
            "Detectamos uma GPU NVIDIA neste computador.\n\n"
            "Deseja baixar os componentes de GPU (~1,5 GB, uma única vez) para "
            "processar os vídeos bem mais rápido?\n\n"
            "Sim = usar a GPU (recomendado se você tem boa internet)\n"
            "Não = usar apenas a CPU (mais lento, mesma qualidade)")
        return bool(ans)
    except Exception:
        # Sem GUI: assume sim (ou troque para input() no console, se preferir).
        return True


def _download_with_progress_ui():
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        download_cuda_runtime(progress_cb=lambda l, d, t: None)
        return

    win = tk.Toplevel() if tk._default_root else tk.Tk()
    win.title("Baixando componentes de GPU…")
    win.geometry("420x120")
    lbl = ttk.Label(win, text="Preparando…")
    lbl.pack(pady=(14, 6), padx=14, anchor="w")
    pb = ttk.Progressbar(win, mode="determinate", maximum=100)
    pb.pack(fill="x", padx=14)
    win.update()

    def cb(label, done, total):
        pct = int(done * 100 / total) if total else 0
        lbl.config(text=f"{label}  —  {pct}%")
        pb["value"] = pct
        win.update()

    try:
        download_cuda_runtime(progress_cb=cb)
    finally:
        win.destroy()


if __name__ == "__main__":
    print("GPU NVIDIA detectada:", has_nvidia_gpu())
    print("CUDA já baixado:", cuda_runtime_present())
    print("Pasta destino:", cuda_dir())
