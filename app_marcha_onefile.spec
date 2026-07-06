# -*- mode: python ; coding: utf-8 -*-
# Build ONEFILE do PAINEL "Análise de Marcha" (app_marcha.py).
#   Gerar:  pyinstaller app_marcha_onefile.spec
# Saída:   dist/AnaliseMarcha.exe  (um único arquivo executável)
#
# Diferença para app_marcha.spec (onedir):
#  - Gera UM único .exe (tudo empacotado). Mais fácil de distribuir, porém maior
#    e com abertura um pouco mais lenta (auto-extrai em pasta temp a cada execução).
#  - Mesmas exclusões e coleta de pacotes do build onedir.

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

# ONEFILE: incluir binaries + datas dentro do próprio EXE (sem COLLECT).
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AnaliseMarcha",
    console=False,          # app GUI; sem janela de console
    disable_windowed_traceback=False,
    upx=False,
    icon=_ICON,
)
