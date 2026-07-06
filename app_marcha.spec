# -*- mode: python ; coding: utf-8 -*-
# Build do PAINEL "Análise de Marcha" (app_marcha.py — dashboard ttkbootstrap).
#   Gerar:  pyinstaller app_marcha.spec
# Saída:   dist/AnaliseMarcha/  (pasta --onedir; empacote com o installer.iss)
#
# Observações:
#  - Entry = app_marcha.py; ele importa analise_marcha.py (motor) e gpu_bootstrap.py.
#  - Backends legados (torch/mmpose/mmcv/mediapipe) EXCLUÍDOS: app é rtmlib-only.
#  - Runtime CUDA NÃO é empacotado: o gpu_bootstrap baixa do PyPI na 1ª execução.
#  - Dados do usuário (Sujeitos/config/registro) NÃO vão no pacote: o app grava em
#    %LOCALAPPDATA%\AnaliseMarcha (ver data_dir() em app_marcha.py).

import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []
_ICON = "icone.ico" if os.path.exists("icone.ico") else None

# Pacotes que precisam vir completos (pacote + dados + DLLs).
for pkg in ("rtmlib", "onnxruntime", "ttkbootstrap", "PIL"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

hiddenimports += collect_submodules("scipy")
hiddenimports += ["gpu_bootstrap", "analise_marcha", "openpyxl",
                  "PIL.ImageTk", "PIL.Image"]

# Identidade visual (ícones das janelas + logos + pasta de ícones da sidebar).
if os.path.isdir("icons"):
    datas += [("icons", "icons")]
for _f in ("icone.ico", "icone_header.png", "icone.png"):
    if os.path.exists(_f):
        datas += [(_f, ".")]

excludes = [
    "torch", "torchvision", "mmpose", "mmcv", "mmengine", "mmdet",
    "mediapipe", "chumpy", "tensorflow", "onnxruntime_gpu",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "IPython", "notebook",
]

a = Analysis(
    ["app_marcha.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="AnaliseMarcha",
    console=False,          # app GUI; sem janela de console
    disable_windowed_traceback=False,
    icon=_ICON,
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="AnaliseMarcha",
)
