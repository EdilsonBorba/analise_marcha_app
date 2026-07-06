# analise_marcha_grama.py
# Análise de marcha em superfície de grama com correção de perspectiva por cones
# Autor: Edilson Borba | borba.edi@gmail.com
# Parâmetros: mecânicos (trabalho externo, recovery, IRL), espaço-temporais bilaterais, angulares bilaterais

# ===================== Imports & Setup =====================
import logging, warnings, os, sys
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import cv2
try:
    import mediapipe as mp          # backend legado/opcional; não usado no app rtmlib
except Exception:
    mp = None
import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import PchipInterpolator

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk

# ===================== CUDA runtime bootstrap (onnxruntime-gpu) =====================
def _enable_cuda_dlls():
    """Registra os diretórios de DLL do runtime CUDA para o onnxruntime-gpu carregar
    cuDNN/cuBLAS. Precisa rodar ANTES de importar onnxruntime/rtmlib. Cobre dois casos:
      - dev: wheels nvidia-*-cu12 dentro do site-packages (.venv);
      - app empacotado: pasta baixada pelo gpu_bootstrap na 1ª execução.
    Sem GPU/wheels, é inofensivo."""
    import glob, sysconfig
    bindirs = []
    try:
        purelib = sysconfig.get_paths()["purelib"]
        bindirs += [d for d in glob.glob(os.path.join(purelib, "nvidia", "*", "bin"))
                    if glob.glob(os.path.join(d, "*.dll"))]
    except Exception:
        pass
    # App empacotado: runtime CUDA baixado pelo bootstrap (se existir).
    try:
        from gpu_bootstrap import cuda_dir
        d = cuda_dir()
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.dll")):
            bindirs.append(d)
    except Exception:
        pass
    if bindirs:
        os.environ["PATH"] = os.pathsep.join(bindirs) + os.pathsep + os.environ.get("PATH", "")
        for d in bindirs:
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
    return bindirs

_enable_cuda_dlls()

# ===================== Constantes =====================
CONE_SPACING_M = 2.5        # distância real entre cones (metros)
CONE_HEIGHT_M  = 0.235       # altura real do cone (metros)
VIS_THRESHOLD  = 0.50       # visibilidade mínima para landmark confiável
CUTOFF_HZ      = 3.0        # frequência de corte filtro (Hz)
FILTER_ORDER   = 2

# Parâmetros segmentares De Leva (1996) — masculino
# (fração da massa corporal, posição do COM a partir do ponto proximal como fração do comprimento)
SEG_PARAMS = {
    # segmento           massa_frac  com_frac_proximal
    'trunk':            (0.602,     0.58),   # cabeça + tronco + MMSS (não rastreados separadamente)
    'thigh_dir':        (0.100,     0.433),
    'thigh_esq':        (0.100,     0.433),
    'shank_dir':        (0.0465,    0.434),
    'shank_esq':        (0.0465,    0.434),
    'foot_dir':         (0.0145,    0.400),
    'foot_esq':         (0.0145,    0.400),
}
SEG_MASS_TOTAL = sum(v[0] for v in SEG_PARAMS.values())  # ~0.924

# Índices MediaPipe → nomes
LANDMARK_MAP = {
    11: "ombro_esq",   12: "ombro_dir",
    23: "quadril_esq", 24: "quadril_dir",
    25: "joelho_esq",  26: "joelho_dir",
    27: "tornozelo_esq", 28: "tornozelo_dir",
    29: "calcanhar_esq", 30: "calcanhar_dir",
    31: "ponta_esq",   32: "ponta_dir",
}
LM_INDICES = list(LANDMARK_MAP.keys())

# ===================== UI / Tema (clínico azul/branco) =====================
APP_TITLE = "Análise de Marcha"
# Arquivos de identidade visual (na pasta do projeto). O icone.ico e o icone_header.png
# são derivados do icone.png (gerados com Pillow); se trocar o icone.png, regenere-os.
APP_ICON_FILE = "icone.ico"          # ícone das janelas / exe / instalador
APP_LOGO_FILE = "icone_header.png"   # logo (pequena) exibida no cabeçalho das telas

# Paleta clínica (fácil de ajustar).
UI = {
    "bg":       "#f4f6f9",   # fundo geral (cinza-azulado bem claro)
    "surface":  "#ffffff",   # cartões / cabeçalho
    "primary":  "#1976d2",   # azul de acento
    "primary_d":"#125aa0",   # azul escuro (hover)
    "text":     "#22303c",   # texto principal
    "muted":    "#6b7a88",   # texto secundário
    "border":   "#d9e1ea",
    "success":  "#2e7d32",
}
FONT       = ("Segoe UI", 10)
FONT_BOLD  = ("Segoe UI Semibold", 10)
FONT_H1    = ("Segoe UI Semibold", 15)
FONT_SMALL = ("Segoe UI", 9)


def resource_path(rel):
    """Caminho de recurso que funciona em dev e no app empacotado (PyInstaller)."""
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel)


_ICON_PHOTO = {}
def set_window_icon(win):
    """Aplica o ícone à janela (barra de título / tarefas). Usa .ico se houver;
    senão tenta um PNG via iconphoto. Silencioso se nenhum existir."""
    try:
        ico = resource_path(APP_ICON_FILE)
        if os.path.exists(ico):
            win.iconbitmap(ico)
            return
    except Exception:
        pass
    try:
        png = resource_path(APP_LOGO_FILE)
        if os.path.exists(png):
            if "img" not in _ICON_PHOTO:
                _ICON_PHOTO["img"] = tk.PhotoImage(file=png)
            win.iconphoto(True, _ICON_PHOTO["img"])
    except Exception:
        pass


_THEME_DONE = [False]
def apply_clinical_theme(root):
    """Aplica um tema ttk claro/clínico. Idempotente."""
    try:
        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        root.configure(bg=UI["bg"])
        style.configure(".", background=UI["bg"], foreground=UI["text"], font=FONT)
        style.configure("TFrame", background=UI["bg"])
        style.configure("Card.TFrame", background=UI["surface"])
        style.configure("TLabel", background=UI["bg"], foreground=UI["text"], font=FONT)
        style.configure("Card.TLabel", background=UI["surface"], foreground=UI["text"], font=FONT)
        style.configure("H1.TLabel", background=UI["surface"], foreground=UI["primary"], font=FONT_H1)
        style.configure("Sub.TLabel", background=UI["surface"], foreground=UI["muted"], font=FONT_SMALL)
        style.configure("Muted.TLabel", background=UI["bg"], foreground=UI["muted"], font=FONT_SMALL)
        style.configure("TEntry", fieldbackground=UI["surface"], bordercolor=UI["border"])
        style.configure("TCombobox", fieldbackground=UI["surface"])
        style.configure("TCheckbutton", background=UI["bg"], foreground=UI["text"], font=FONT)
        style.configure("TProgressbar", troughcolor=UI["border"], background=UI["primary"],
                        bordercolor=UI["border"], lightcolor=UI["primary"], darkcolor=UI["primary"])
        style.configure("TButton", font=FONT, padding=(10, 6))
        style.configure("Accent.TButton", font=FONT_BOLD, padding=(14, 8),
                        foreground="#ffffff", background=UI["primary"], borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active", UI["primary_d"]), ("pressed", UI["primary_d"])],
                  foreground=[("disabled", "#e8eef5"), ("!disabled", "#ffffff")])
        _THEME_DONE[0] = True
    except Exception:
        pass


_LOGO_CACHE = {}
def _load_logo(max_h=40):
    """Carrega a logo (PNG) redimensionada para o cabeçalho, se existir. Cacheado."""
    if max_h in _LOGO_CACHE:
        return _LOGO_CACHE[max_h]
    img = None
    try:
        p = resource_path(APP_LOGO_FILE)
        if os.path.exists(p):
            ph = tk.PhotoImage(file=p)
            # subamostra em passos inteiros para caber na altura do cabeçalho
            factor = max(1, ph.height() // max_h)
            if factor > 1:
                ph = ph.subsample(factor, factor)
            img = ph
    except Exception:
        img = None
    _LOGO_CACHE[max_h] = img
    return img


def make_header(parent, title, subtitle=""):
    """Cria uma faixa de cabeçalho (logo + título) no topo de uma janela."""
    bar = tk.Frame(parent, bg=UI["surface"], highlightthickness=0)
    inner = tk.Frame(bar, bg=UI["surface"])
    inner.pack(fill="x", padx=16, pady=10)
    logo = _load_logo(40)
    if logo is not None:
        lbl_logo = tk.Label(inner, image=logo, bg=UI["surface"])
        lbl_logo.image = logo  # evita GC
        lbl_logo.pack(side="left", padx=(0, 12))
    txt = tk.Frame(inner, bg=UI["surface"])
    txt.pack(side="left", fill="x")
    tk.Label(txt, text=title, bg=UI["surface"], fg=UI["primary"], font=FONT_H1, anchor="w").pack(anchor="w")
    if subtitle:
        tk.Label(txt, text=subtitle, bg=UI["surface"], fg=UI["muted"], font=FONT_SMALL, anchor="w").pack(anchor="w")
    tk.Frame(bar, bg=UI["border"], height=1).pack(fill="x")
    bar.pack(fill="x", side="top")
    return bar


# ===================== Processamento de Sinal =====================
def butter_lp(data, cutoff, fs, order=FILTER_ORDER):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, np.asarray(data, dtype=float))


def interpolate_gaps(values, reliable_mask):
    """Interpola NaN / frames não-confiáveis com PCHIP."""
    x = np.arange(len(values))
    good = reliable_mask & np.isfinite(values)
    if good.sum() < 2:
        return values.copy()
    f = PchipInterpolator(x[good], values[good], extrapolate=True)
    out = values.copy().astype(float)
    out[~good] = f(x[~good])
    return out


def normalize_to_100(data_1d, n_pts=101):
    t0 = np.linspace(0, 100, len(data_1d))
    ti = np.linspace(0, 100, n_pts)
    return PchipInterpolator(t0, data_1d, extrapolate=True)(ti)

# ===================== Geometria & Ângulos =====================
def _vec(a, b):
    return b - a


def angle_3pts(a, b, c):
    """Ângulo não-assinado em b, formado por a–b–c (graus)."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def signed_angle_3pts(a, b, c):
    """Ângulo assinado em b (positivo = anti-horário no plano xy imagem)."""
    v1, v2 = a - b, c - b
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return math.degrees(math.atan2(cross, np.dot(v1, v2)))


def hip_flexion(ombro, quadril, joelho):
    """Flexão de quadril: positivo = flexão, negativo = extensão."""
    sa = signed_angle_3pts(ombro, quadril, joelho)
    return (180 - abs(sa)) * (1 if sa >= 0 else -1)


def knee_flexion(quadril, joelho, tornozelo):
    """Flexão de joelho: 0° = extensão completa."""
    return 180 - angle_3pts(quadril, joelho, tornozelo)


def ankle_angle(joelho, tornozelo, ponta):
    """Ângulo tornozelo (convenção: dorsiflexão positiva)."""
    tib  = joelho  - tornozelo
    foot = ponta   - tornozelo
    n1, n2 = np.linalg.norm(tib), np.linalg.norm(foot)
    if n1 == 0 or n2 == 0:
        return np.nan
    cross = tib[0]*foot[1] - tib[1]*foot[0]
    signed = math.degrees(math.atan2(cross, np.dot(tib, foot)))
    return -(signed - 107)


def trunk_inclination(ombro_mid, quadril_mid):
    """Inclinação do tronco em relação à vertical (imagem).
    Positivo = inclinação anterior."""
    vec = quadril_mid - ombro_mid   # ombro → quadril
    # ângulo com eixo Y da imagem (que aponta para baixo)
    return math.degrees(math.atan2(vec[0], vec[1]))

# ===================== Correção de Perspectiva =====================
def select_cones_opencv(frame, cone_spacing_m=CONE_SPACING_M, cone_height_m=CONE_HEIGHT_M):
    """
    Interface OpenCV para o usuário marcar todos os cones visíveis no frame.

    Fase 1: clique na BASE de cada cone, da ESQUERDA para a DIREITA
            (qualquer quantidade ≥ 2). ENTER confirma.
    Fase 2: clique no TOPO de cada cone marcado, na mesma ordem esq → dir.
            ENTER confirma.
    R desfaz o último clique da fase atual. Q pula a calibração.

    Retorna:
        cone_base_px : array (N,) com pixel_x da base de cada cone
        cone_real_x  : array (N,) com posição real (0, spacing, 2*spacing, ...)
        cone_scale_y : array (N,) com escala local m/px (altura real / altura em px)
    ou (None, None, None) se a calibração for pulada/insuficiente.
    """
    clone = frame.copy()
    bases, tops = [], []
    phase = [1]   # 1 = base, 2 = topo

    def mouse_cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if phase[0] == 1:
            bases.append((x, y))
        elif phase[0] == 2 and len(tops) < len(bases):
            tops.append((x, y))

    win_name = "Calibracao de Cones (clique nos cones)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, min(frame.shape[1], 1280), min(frame.shape[0], 720))
    cv2.setMouseCallback(win_name, mouse_cb)

    while True:
        disp = clone.copy()
        for i, (bx, by) in enumerate(bases):
            cv2.circle(disp, (bx, by), 8, (0, 255, 0), -1)
            cv2.putText(disp, f"B{i+1}", (bx+10, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for i, (tx, ty) in enumerate(tops):
            cv2.circle(disp, (tx, ty), 8, (0, 165, 255), -1)
            cv2.putText(disp, f"T{i+1}", (tx+10, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            bx, by = bases[i]
            cv2.line(disp, (bx, by), (tx, ty), (0, 165, 255), 2)

        if phase[0] == 1:
            instructions = [
                "FASE 1/2: clique na BASE de cada cone, ESQUERDA -> DIREITA",
                "R = desfazer ultimo | ENTER = confirmar bases (min. 2) | Q = pular calibracao",
            ]
        else:
            instructions = [
                f"FASE 2/2: clique no TOPO de cada cone (mesma ordem)  {len(tops)}/{len(bases)}",
                "R = desfazer ultimo | ENTER = confirmar | Q = pular calibracao",
            ]
        for i, txt in enumerate(instructions):
            cv2.putText(disp, txt, (10, 30 + i*28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(win_name, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            if phase[0] == 1 and bases:
                bases.pop()
            elif phase[0] == 2 and tops:
                tops.pop()
        elif key == 13:   # ENTER
            if phase[0] == 1:
                if len(bases) >= 2:
                    phase[0] = 2
            else:
                if len(tops) == len(bases):
                    break
        elif key == ord('q'):
            bases.clear()
            tops.clear()
            break

    cv2.destroyWindow(win_name)

    if len(bases) < 2 or len(tops) != len(bases):
        return None, None, None

    bases = np.array(bases, dtype=float)
    tops  = np.array(tops,  dtype=float)

    order = np.argsort(bases[:, 0])
    bases = bases[order]
    tops  = tops[order]

    cone_base_px = bases[:, 0]
    cone_real_x  = np.arange(len(cone_base_px)) * cone_spacing_m

    h_px = np.abs(bases[:, 1] - tops[:, 1])
    h_px[h_px == 0] = np.nan
    cone_scale_y = cone_height_m / h_px

    return cone_base_px, cone_real_x, cone_scale_y


def build_scale_functions(cone_base_px, cone_real_x, cone_scale_y, thigh_px_mean, thigh_real_m):
    """
    Constrói funções de escala pixel→metro.

    Eixo X: interpolação pela posição dos cones (corrige perspectiva horizontal).
    Eixo Y: interpolação pela altura aparente dos cones (corrige perspectiva
            de profundidade); cai para escala constante pela coxa se não
            houver calibração de altura.

    Retorna:
        scale_x_func : função(array pixel_x) → metros reais (absolutos)
        scale_y_func : função(array pixel_x) → metros/pixel (escala vertical local)
    """
    uy = thigh_real_m / thigh_px_mean

    if cone_base_px is not None and len(cone_base_px) >= 2:
        scale_x_func = PchipInterpolator(cone_base_px, cone_real_x, extrapolate=True)
    else:
        scale_x_func = lambda px: px * uy

    scale_y_func = lambda px: np.full(np.shape(px), uy)
    if cone_base_px is not None and cone_scale_y is not None:
        valid = np.isfinite(cone_scale_y) & (cone_scale_y > 0)
        n_bad = int((~valid).sum())
        if n_bad:
            print(f"[AVISO] {n_bad} cone(s) com altura inválida (base/topo "
                  f"marcados muito próximos) — ignorado(s) na escala Y.")
        if valid.sum() >= 2:
            scale_y_func = PchipInterpolator(cone_base_px[valid], cone_scale_y[valid],
                                              extrapolate=True)

    return scale_x_func, scale_y_func

# ===================== Pose Estimation Backends =====================
# O pipeline abaixo preserva o mesmo contrato do script original:
# qualquer backend deve retornar um DataFrame com as mesmas colunas de landmarks
# usadas no restante da análise.

MMPOSE_WHOLEBODY_MAP = {
    # COCO-WholeBody / RTMPose WholeBody: primeiros 17 pontos = COCO body;
    # 17-22 = pé esquerdo/direito: big toe, small toe, heel.
    "ombro_esq": 5,   "ombro_dir": 6,
    "quadril_esq": 11, "quadril_dir": 12,
    "joelho_esq": 13,  "joelho_dir": 14,
    "tornozelo_esq": 15, "tornozelo_dir": 16,
    "ponta_esq": 17,   "calcanhar_esq": 19,
    "ponta_dir": 20,   "calcanhar_dir": 22,
}

MMPOSE_COCO_MAP = {
    # COCO body 17 pontos. Não possui calcanhar/ponta; esses serão preenchidos
    # como fallback de baixa confiança a partir do tornozelo.
    "ombro_esq": 5,   "ombro_dir": 6,
    "quadril_esq": 11, "quadril_dir": 12,
    "joelho_esq": 13,  "joelho_dir": 14,
    "tornozelo_esq": 15, "tornozelo_dir": 16,
}

RTMLIB_HALPE26_MAP = {
    # rtmlib BodyWithFeet → RTMPose Halpe-26. Traz calcanhar e dedo (big toe)
    # REAIS dos dois pés — exatamente o que a análise de marcha precisa.
    # Convenção anatômica: índice "left" = lado esquerdo da pessoa = *_esq.
    "ombro_esq": 5,   "ombro_dir": 6,
    "quadril_esq": 11, "quadril_dir": 12,
    "joelho_esq": 13,  "joelho_dir": 14,
    "tornozelo_esq": 15, "tornozelo_dir": 16,
    "ponta_esq": 20,   "ponta_dir": 21,     # 20 = left_big_toe, 21 = right_big_toe
    "calcanhar_esq": 24, "calcanhar_dir": 25,
}

# Portão de tamanho do sujeito (rtmlib): o sujeito está em primeiro plano e é de longe
# o maior no quadro; gente ao fundo é pequena. Estimamos a altura típica do sujeito no
# vídeo (percentil das alturas por frame) e só aceitamos detecções de tamanho compatível,
# descartando pessoas pequenas (fundo) — inclusive no começo/fim, antes de o sujeito entrar.
RTMLIB_SUBJECT_HEIGHT_PCTL = 90   # percentil das alturas por frame p/ estimar o sujeito
RTMLIB_SUBJECT_MIN_FRAC    = 0.5  # fração dessa altura para aceitar uma detecção


def get_frame_at_time(video_path, time_s=None):
    """Lê um frame específico do vídeo. Usado para seleção de ROI."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if time_s is not None and fps and fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(time_s * fps), 0))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def select_subject_roi_opencv(frame):
    """
    Permite selecionar manualmente uma ROI do corredor/sujeito.

    Isso é especialmente importante no MMPose com det_model='whole_image':
    em vez de forçar o modelo a interpretar o frame inteiro como uma pessoa,
    ele interpreta apenas a região selecionada.

    Controles do OpenCV:
      - arraste o retângulo ao redor da trajetória completa do sujeito
      - ENTER/ESPAÇO confirma
      - C cancela e usa o frame inteiro
    """
    if frame is None:
        return None
    disp = frame.copy()
    cv2.putText(disp, "Selecione a ROI da pessoa/trajetoria | ENTER confirma | C cancela",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    win = "ROI MMPose - selecione a pessoa/trajetoria"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1280), min(frame.shape[0], 720))
    x, y, w, h = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)
    if w <= 10 or h <= 10:
        return None
    # pequena margem para não cortar pés/braços durante a caminhada
    margin_x = int(w * 0.08)
    margin_y = int(h * 0.12)
    x0 = max(int(x) - margin_x, 0)
    y0 = max(int(y) - margin_y, 0)
    x1 = min(int(x + w) + margin_x, frame.shape[1])
    y1 = min(int(y + h) + margin_y, frame.shape[0])
    return (x0, y0, x1 - x0, y1 - y0)


# ===================== MMPose ROI / Visual Focus =====================
def _flatten_mmpose_predictions(predictions):
    """Achata diferentes formatos de saída do MMPoseInferencer para lista de instâncias."""
    if predictions is None:
        return []
    if isinstance(predictions, dict):
        return [predictions]
    if not isinstance(predictions, list):
        return []
    out = []
    for item in predictions:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, list):
            out.extend([x for x in item if isinstance(x, dict)])
    return out


def _mmpose_instance_metrics(inst, score_thr=0.20):
    """Calcula bbox, área, confiança e completude de uma instância MMPose."""
    kps = np.asarray(inst.get("keypoints", []), dtype=float)
    sc = np.asarray(inst.get("keypoint_scores", []), dtype=float)
    if kps.ndim != 2 or kps.shape[0] == 0 or kps.shape[1] < 2:
        return None

    finite = np.isfinite(kps[:, 0]) & np.isfinite(kps[:, 1])
    if sc.size == kps.shape[0]:
        reliable = finite & (sc >= score_thr)
        conf_values = sc[reliable] if reliable.any() else sc[finite]
    else:
        reliable = finite
        conf_values = np.ones(int(finite.sum()))

    if not reliable.any():
        reliable = finite
    if not reliable.any():
        return None

    xs = kps[reliable, 0]
    ys = kps[reliable, 1]
    x0, y0 = float(np.nanmin(xs)), float(np.nanmin(ys))
    x1, y1 = float(np.nanmax(xs)), float(np.nanmax(ys))
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    area = width * height
    mean_conf = float(np.nanmean(conf_values)) if len(conf_values) else 0.0
    completeness = float(reliable.sum() / max(kps.shape[0], 1))
    # score simples: pessoa maior + keypoints confiáveis + corpo mais completo.
    score = float(area * mean_conf * max(completeness, 0.05))
    return {
        "bbox": (x0, y0, x1, y1),
        "area": area,
        "mean_confidence": mean_conf,
        "completeness": completeness,
        "score": score,
        "n_reliable": int(reliable.sum()),
    }


def _expand_bbox_to_roi(bbox, frame_shape, margin=0.30):
    """Expande bbox e converte para ROI OpenCV (x, y, w, h)."""
    x0, y0, x1, y1 = bbox
    h, w = frame_shape[:2]
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    mx = bw * margin
    my = bh * margin
    rx0 = max(int(round(x0 - mx)), 0)
    ry0 = max(int(round(y0 - my)), 0)
    rx1 = min(int(round(x1 + mx)), w)
    ry1 = min(int(round(y1 + my)), h)
    if rx1 <= rx0 or ry1 <= ry0:
        return None
    return (rx0, ry0, rx1 - rx0, ry1 - ry0)


def draw_focus_overlay(frame, roi=None, landmarks_row=None, bbox=None, title="", dim_alpha=0.62):
    """Mostra a pessoa/ROI em destaque e deixa o restante do frame fosco."""
    if frame is None:
        return None
    base = frame.copy()
    dim = (base.astype(np.float32) * (1.0 - float(dim_alpha))).astype(np.uint8)
    out = dim.copy()

    if roi is not None:
        x, y, w, h = [int(v) for v in roi]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = base[y0:y1, x0:x1]
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 3)
            cv2.putText(out, "PESSOA / ROI", (x0 + 8, max(y0 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    if bbox is not None:
        bx0, by0, bx1, by1 = [int(round(v)) for v in bbox]
        cv2.rectangle(out, (bx0, by0), (bx1, by1), (0, 255, 0), 2)

    if landmarks_row is not None:
        for nm in LANDMARK_MAP.values():
            x = landmarks_row.get(f"{nm}_x_px", np.nan)
            y = landmarks_row.get(f"{nm}_y_px", np.nan)
            v = landmarks_row.get(f"{nm}_vis", 0.0)
            if np.isfinite(x) and np.isfinite(y):
                color = (0, 255, 0) if float(v) >= VIS_THRESHOLD else (0, 165, 255)
                cv2.circle(out, (int(round(x)), int(round(y))), 5, color, -1)

    if title:
        cv2.rectangle(out, (0, 0), (min(frame.shape[1], 1180), 72), (0, 0, 0), -1)
        cv2.putText(out, title[:150], (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(out, "ENTER/ESPACO aceita | N/P troca | M ROI manual | Q cancela",
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)
    return out


def select_auto_person_roi_mmpose(video_path, start_sec=None, end_sec=None,
                                  model_alias="wholebody", device="cuda:0",
                                  sample_count=5, margin=0.32):
    """
    Seleciona automaticamente a pessoa principal/mais próxima com MMPose.
    Retorna (roi, meta). Se falhar/cancelar, roi=None.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0:
        fps = 30.0
    f_start = int(start_sec * fps) if start_sec is not None else 0
    f_end = int(end_sec * fps) if end_sec is not None else max(total - 1, 0)
    f_start = max(0, min(f_start, max(total - 1, 0)))
    f_end = max(f_start, min(f_end, max(total - 1, 0)))

    if total <= 0:
        cap.release()
        return None, {"roi_mode": "auto_person", "status": "video_sem_frames"}

    sample_frames = np.linspace(f_start, f_end, num=max(int(sample_count), 1), dtype=int)
    sample_frames = sorted(set(int(x) for x in sample_frames))

    print(f"[INFO] Seleção automática da pessoa: frames amostrados = {sample_frames}")
    try:
        inferencer = _init_mmpose_inferencer(model_alias=model_alias, device=device, use_whole_image=False)
    except Exception as e:
        if str(device).startswith("cuda"):
            print(f"[AVISO] Auto-person falhou em {device}: {e}")
            print("[INFO] Tentando auto-person em CPU...")
            try:
                inferencer = _init_mmpose_inferencer(model_alias=model_alias, device="cpu", use_whole_image=False)
            except Exception as e2:
                cap.release()
                return None, {"roi_mode": "auto_person", "status": "falha_inferencer", "erro": str(e2)}
        else:
            cap.release()
            return None, {"roi_mode": "auto_person", "status": "falha_inferencer", "erro": str(e)}

    candidates = []
    for fr in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            result = next(inferencer(frame, return_vis=False, show=False))
            instances = _flatten_mmpose_predictions(result.get("predictions"))
        except Exception as e:
            print(f"[AVISO] Auto-person falhou no frame {fr}: {e}")
            instances = []

        for cand_i, inst in enumerate(instances):
            metrics = _mmpose_instance_metrics(inst, score_thr=0.20)
            if metrics is None:
                continue
            # Remove detecções minúsculas e muito incompletas: geralmente fundo/objeto.
            if metrics["area"] < 1500 or metrics["n_reliable"] < 6:
                continue
            roi = _expand_bbox_to_roi(metrics["bbox"], frame.shape, margin=margin)
            if roi is None:
                continue
            row = _mmpose_instance_to_row(inst, fr)
            candidates.append({
                "frame_idx": int(fr),
                "candidate_idx": int(cand_i),
                "frame": frame.copy(),
                "instance": inst,
                "row": row,
                "roi": roi,
                **metrics,
            })

    cap.release()

    if not candidates:
        return None, {"roi_mode": "auto_person", "status": "sem_candidatos", "n_candidates": 0}

    candidates.sort(key=lambda c: c["score"], reverse=True)
    idx = 0
    win = "MMPose Auto-Person - revisar pessoa selecionada"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    first_frame_shape = candidates[0]["frame"].shape
    cv2.resizeWindow(win, min(first_frame_shape[1], 1280), min(first_frame_shape[0], 720))

    accepted = False
    manual_roi = None
    while True:
        c = candidates[idx]
        title = (f"Candidato {idx+1}/{len(candidates)} | frame {c['frame_idx']} | "
                 f"score={c['score']:.1f} | conf={c['mean_confidence']:.2f} | "
                 f"comp={c['completeness']:.2f} | apos aceitar: ROI acompanha a pessoa")
        disp = draw_focus_overlay(c["frame"], roi=c["roi"], landmarks_row=c["row"],
                                  bbox=c["bbox"], title=title, dim_alpha=0.66)
        cv2.imshow(win, disp)
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):  # ENTER ou ESPAÇO
            accepted = True
            break
        if key in (ord('n'), ord('N')):
            idx = (idx + 1) % len(candidates)
        elif key in (ord('p'), ord('P')):
            idx = (idx - 1) % len(candidates)
        elif key in (ord('m'), ord('M')):
            cv2.destroyWindow(win)
            manual_roi = select_subject_roi_opencv(c["frame"])
            break
        elif key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()

    if manual_roi is not None:
        return manual_roi, {
            "roi_mode": "auto_person",
            "status": "fallback_manual_roi",
            "n_candidates": len(candidates),
            "selected_frame": candidates[idx]["frame_idx"],
        }
    if not accepted:
        return None, {
            "roi_mode": "auto_person",
            "status": "cancelado",
            "n_candidates": len(candidates),
        }

    c = candidates[idx]
    return c["roi"], {
        "roi_mode": "auto_person",
        "status": "aceito",
        "n_candidates": len(candidates),
        "selected_frame": c["frame_idx"],
        "selected_candidate": c["candidate_idx"],
        "score": c["score"],
        "mean_confidence": c["mean_confidence"],
        "completeness": c["completeness"],
        "area_bbox_px2": c["area"],
        "tracking": True,
        "bbox_x0": c["bbox"][0],
        "bbox_y0": c["bbox"][1],
        "bbox_x1": c["bbox"][2],
        "bbox_y1": c["bbox"][3],
        "roi_x": c["roi"][0],
        "roi_y": c["roi"][1],
        "roi_w": c["roi"][2],
        "roi_h": c["roi"][3],
    }


def _offset_mmpose_row(row, offset_x=0, offset_y=0):
    """Soma offset de uma ROI às coordenadas MMPose para voltar ao frame original."""
    if not offset_x and not offset_y:
        return row
    row = row.copy()
    for nm in LANDMARK_MAP.values():
        xcol = f"{nm}_x_px"
        ycol = f"{nm}_y_px"
        if xcol in row and np.isfinite(row[xcol]):
            row[xcol] = float(row[xcol]) + float(offset_x)
        if ycol in row and np.isfinite(row[ycol]):
            row[ycol] = float(row[ycol]) + float(offset_y)
    return row


def _row_major_joint_score(row):
    """Pontuação simples de qualidade anatômica para rejeitar poses absurdas."""
    major = [
        "ombro_esq", "ombro_dir", "quadril_esq", "quadril_dir",
        "joelho_esq", "joelho_dir", "tornozelo_esq", "tornozelo_dir",
    ]
    scores = [float(row.get(f"{nm}_vis", 0.0)) for nm in major]
    finite = [np.isfinite(row.get(f"{nm}_x_px", np.nan)) and np.isfinite(row.get(f"{nm}_y_px", np.nan)) for nm in major]
    if not any(finite):
        return 0.0
    return float(np.nanmean(scores)) * (sum(finite) / len(major))


def _row_to_bbox(row, score_thr=0.15):
    """Cria uma bbox global a partir dos landmarks já convertidos para o frame original."""
    xs, ys = [], []
    for nm in LANDMARK_MAP.values():
        x = row.get(f"{nm}_x_px", np.nan)
        y = row.get(f"{nm}_y_px", np.nan)
        v = float(row.get(f"{nm}_vis", 0.0))
        if np.isfinite(x) and np.isfinite(y) and v >= score_thr:
            xs.append(float(x)); ys.append(float(y))
    if len(xs) < 4:
        # fallback: usa qualquer landmark finito se poucos pontos passaram no score
        for nm in LANDMARK_MAP.values():
            x = row.get(f"{nm}_x_px", np.nan)
            y = row.get(f"{nm}_y_px", np.nan)
            if np.isfinite(x) and np.isfinite(y):
                xs.append(float(x)); ys.append(float(y))
    if len(xs) < 4:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _expand_roi(roi, frame_shape, grow=0.25):
    """Expande uma ROI existente quando o rastreamento falha por alguns frames."""
    if roi is None:
        return None
    x, y, w, h = [int(v) for v in roi]
    H, W = frame_shape[:2]
    mx = int(round(w * grow))
    my = int(round(h * grow))
    x0 = max(x - mx, 0)
    y0 = max(y - my, 0)
    x1 = min(x + w + mx, W)
    y1 = min(y + h + my, H)
    if x1 <= x0 or y1 <= y0:
        return roi
    return (x0, y0, x1 - x0, y1 - y0)


def _clip_roi(roi, frame_shape):
    """Garante que a ROI fique dentro do frame."""
    if roi is None:
        return None
    x, y, w, h = [int(v) for v in roi]
    H, W = frame_shape[:2]
    x0 = max(x, 0); y0 = max(y, 0)
    x1 = min(x + w, W); y1 = min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def process_video_mediapipe(video_path, start_sec=None, end_sec=None, show_window=False):
    """
    Extrai landmarks MediaPipe do vídeo (segmento start_sec–end_sec).
    Retorna DataFrame com coords em pixels e visibilidade, lista de frames, fps, w, h.
    """
    if mp is None:
        raise ImportError("Backend 'mediapipe' indisponível neste ambiente. "
                          "Use o backend 'rtmlib' (padrão).")
    mp_pose = mp.solutions.pose
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    f_start = int(start_sec * fps) if start_sec is not None else 0
    f_end   = int(end_sec   * fps) if end_sec   is not None else total
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    data, frames = [], []
    fidx = f_start

    def draw_pose_overlay(frame, landmarks, visibility_threshold=VIS_THRESHOLD):
        overlay = frame.copy()

        points = {}
        for idx in LM_INDICES:
            lm = landmarks[idx]
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            points[idx] = (cx, cy, lm.visibility)

        for idx, (cx, cy, vis) in points.items():
            color = (0, 255, 0) if vis >= visibility_threshold else (0, 165, 255)
            cv2.circle(overlay, (cx, cy), 5, color, -1)
        return overlay

    with mp_pose.Pose(static_image_mode=False, model_complexity=2,
                      smooth_landmarks=True,
                      min_detection_confidence=0.50,
                      min_tracking_confidence=0.50) as pose:
        while cap.isOpened() and fidx <= f_end:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lm  = res.pose_landmarks.landmark
                row = {"frame": fidx}
                for idx in LM_INDICES:
                    nm = LANDMARK_MAP[idx]
                    row[f"{nm}_x_px"]  = lm[idx].x * w
                    row[f"{nm}_y_px"]  = lm[idx].y * h
                    row[f"{nm}_vis"]   = lm[idx].visibility
                data.append(row)
                frames.append(frame.copy())
            fidx += 1
            if show_window:
                disp = draw_pose_overlay(frame, lm) if res.pose_landmarks else frame
                cv2.imshow("Processando MediaPipe...", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    return pd.DataFrame(data).reset_index(drop=True), frames, fps, w, h


def _init_mmpose_inferencer(model_alias="wholebody", device="cuda:0", use_whole_image=True):
    """Inicializa MMPose de forma preguiçosa para o script ainda abrir sem MMPose instalado."""
    try:
        from mmpose.apis import MMPoseInferencer
    except Exception as e:
        raise ImportError(
            "MMPose não está instalado neste ambiente. Instale em um ambiente separado "
            "ou selecione backend='mediapipe'. Erro original: " + str(e)
        )

    kwargs = {"pose2d": model_alias, "device": device}
    if use_whole_image:
        # Evita depender do MMDetection; bom quando o vídeo tem uma única pessoa.
        kwargs["det_model"] = "whole_image"
    try:
        return MMPoseInferencer(**kwargs)
    except TypeError:
        # Compatibilidade com versões/assinaturas antigas.
        kwargs.pop("det_model", None)
        try:
            return MMPoseInferencer(**kwargs)
        except TypeError:
            return MMPoseInferencer(model_alias)


def _best_mmpose_instance(predictions, prev_center=None):
    """Escolhe a instância com maior confiança, área plausível e continuidade temporal."""
    instances = _flatten_mmpose_predictions(predictions)
    if not instances:
        return None

    def score_instance(inst):
        metrics = _mmpose_instance_metrics(inst, score_thr=0.15)
        if metrics is None:
            return 0.0
        score = metrics["score"]
        if prev_center is not None:
            kps = np.asarray(inst.get("keypoints", []), dtype=float)
            if kps.ndim == 2 and kps.shape[0] > 0:
                finite = np.isfinite(kps[:, 0]) & np.isfinite(kps[:, 1])
                if finite.any():
                    cx = float(np.nanmean(kps[finite, 0]))
                    cy = float(np.nanmean(kps[finite, 1]))
                    dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
                    score *= 1.0 / (1.0 + dist / 250.0)
        return score

    return max(instances, key=score_instance)


def _mmpose_instance_to_row(inst, frame_idx):
    """Converte uma instância MMPose para o mesmo padrão de colunas do MediaPipe."""
    row = {"frame": frame_idx}
    kps = np.asarray(inst.get("keypoints", []), dtype=float)
    scores = np.asarray(inst.get("keypoint_scores", []), dtype=float)

    if kps.ndim != 2 or kps.shape[1] < 2:
        return row

    # Se houver pelo menos 23 pontos, assume COCO-WholeBody e usa calcanhar/ponta.
    # Caso contrário, usa COCO body e cria fallback para pé.
    if kps.shape[0] >= 23:
        mapping = MMPOSE_WHOLEBODY_MAP
    else:
        mapping = MMPOSE_COCO_MAP

    for nm, idx in mapping.items():
        if idx < kps.shape[0]:
            row[f"{nm}_x_px"] = float(kps[idx, 0])
            row[f"{nm}_y_px"] = float(kps[idx, 1])
            row[f"{nm}_vis"]  = float(scores[idx]) if idx < scores.size else 1.0

    # Fallback se o modelo não fornecer calcanhar/ponta. Mantém pipeline vivo, mas
    # marca baixa confiança para o relatório de qualidade denunciar o problema.
    for side in ("esq", "dir"):
        ankle_x = row.get(f"tornozelo_{side}_x_px", np.nan)
        ankle_y = row.get(f"tornozelo_{side}_y_px", np.nan)
        ankle_v = row.get(f"tornozelo_{side}_vis", 0.0)
        knee_x  = row.get(f"joelho_{side}_x_px", ankle_x)
        knee_y  = row.get(f"joelho_{side}_y_px", ankle_y)

        # Vetor joelho→tornozelo como aproximação grosseira da orientação distal.
        vx = ankle_x - knee_x if np.isfinite(ankle_x) and np.isfinite(knee_x) else 0.0
        vy = ankle_y - knee_y if np.isfinite(ankle_y) and np.isfinite(knee_y) else 0.0

        if f"calcanhar_{side}_x_px" not in row:
            row[f"calcanhar_{side}_x_px"] = ankle_x - 0.10 * vx
            row[f"calcanhar_{side}_y_px"] = ankle_y - 0.10 * vy
            row[f"calcanhar_{side}_vis"]  = min(float(ankle_v), 0.25)
        if f"ponta_{side}_x_px" not in row:
            row[f"ponta_{side}_x_px"] = ankle_x + 0.25 * vx
            row[f"ponta_{side}_y_px"] = ankle_y + 0.25 * vy
            row[f"ponta_{side}_vis"]  = min(float(ankle_v), 0.25)

    # Garante todas as colunas esperadas.
    for nm in LANDMARK_MAP.values():
        row.setdefault(f"{nm}_x_px", np.nan)
        row.setdefault(f"{nm}_y_px", np.nan)
        row.setdefault(f"{nm}_vis",  0.0)
    return row


def process_video_mmpose(video_path, start_sec=None, end_sec=None, show_window=False,
                         model_alias="wholebody", device="cuda:0", use_whole_image=True,
                         roi=None, dynamic_roi=False, dynamic_margin=1.25):
    """
    Extrai landmarks com MMPose/RTMPose/RTMW e retorna o mesmo formato do MediaPipe.

    Modos de ROI:
      - roi=None + dynamic_roi=False: usa o frame inteiro.
      - roi fixa + dynamic_roi=False: usa sempre o mesmo recorte.
      - roi inicial + dynamic_roi=True: usa a ROI apenas como ponto inicial e depois
        acompanha a pessoa frame a frame, atualizando o recorte pelo bbox dos landmarks.

    O modo dynamic_roi é o recomendado para caminhada passando na frente da câmera
    (não-esteira), porque a pessoa se desloca ao longo do quadro.
    """
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        fps = 30.0
    f_start = int(start_sec * fps) if start_sec is not None else 0
    f_end   = int(end_sec   * fps) if end_sec   is not None else total
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    print(f"[INFO] Inicializando MMPose: pose2d={model_alias}, device={device}, "
          f"det_model={'whole_image' if use_whole_image else 'default'}, roi={roi}, dynamic_roi={dynamic_roi}")
    try:
        inferencer = _init_mmpose_inferencer(model_alias=model_alias, device=device, use_whole_image=use_whole_image)
    except Exception as e:
        if str(device).startswith("cuda"):
            print(f"[AVISO] Falha ao iniciar MMPose em {device}: {e}")
            print("[INFO] Tentando CPU...")
            inferencer = _init_mmpose_inferencer(model_alias=model_alias, device="cpu", use_whole_image=use_whole_image)
        else:
            raise

    data, frames = [], []
    fidx = f_start

    # ROI corrente. No modo dinâmico ela será atualizada a cada detecção válida.
    current_roi = _clip_roi(roi, (h, w, 3)) if roi is not None else None
    prev_global_center = None
    fail_count = 0
    last_good_roi = current_roi

    while cap.isOpened() and fidx <= f_end:
        ret, frame = cap.read()
        if not ret:
            break

        # Se a pessoa sumiu do crop por alguns frames, amplia a janela de busca.
        if dynamic_roi and fail_count > 0 and current_roi is not None:
            current_roi = _expand_roi(current_roi, frame.shape, grow=min(0.15 * fail_count, 0.75))

        if current_roi is not None:
            rx, ry, rw, rh = [int(v) for v in current_roi]
            proc_frame = frame[ry:ry+rh, rx:rx+rw]
            offset_x, offset_y = rx, ry
            # Continuidade temporal precisa estar nas coordenadas do crop atual.
            if prev_global_center is not None:
                prev_center_local = (prev_global_center[0] - offset_x, prev_global_center[1] - offset_y)
            else:
                prev_center_local = None
        else:
            proc_frame = frame
            offset_x, offset_y = 0, 0
            prev_center_local = prev_global_center

        if proc_frame is None or proc_frame.size == 0:
            fidx += 1
            fail_count += 1
            continue

        try:
            result = next(inferencer(proc_frame, return_vis=False, show=False))
            inst = _best_mmpose_instance(result.get("predictions"), prev_center=prev_center_local)
        except Exception as e:
            print(f"[AVISO] MMPose falhou no frame {fidx}: {e}")
            inst = None

        row = None
        row_score = 0.0
        if inst is not None:
            row = _mmpose_instance_to_row(inst, fidx)
            row = _offset_mmpose_row(row, offset_x, offset_y)
            row_score = _row_major_joint_score(row)
            if row_score < 0.08:
                row = None

        if row is not None:
            bbox_global = _row_to_bbox(row, score_thr=0.15)

            # Atualiza centro global para escolher a mesma pessoa no frame seguinte.
            xs = [row.get(f"{nm}_x_px", np.nan) for nm in LANDMARK_MAP.values()]
            ys = [row.get(f"{nm}_y_px", np.nan) for nm in LANDMARK_MAP.values()]
            finite = np.isfinite(xs) & np.isfinite(ys)
            if np.any(finite):
                prev_global_center = (float(np.nanmean(np.asarray(xs)[finite])),
                                      float(np.nanmean(np.asarray(ys)[finite])))

            # Tracking: a ROI acompanha a pessoa. A margem é grande de propósito para
            # não cortar pé/cabeça e para tolerar deslocamento entre frames.
            if dynamic_roi and bbox_global is not None:
                new_roi = _expand_bbox_to_roi(bbox_global, frame.shape, margin=dynamic_margin)
                if new_roi is not None:
                    current_roi = new_roi
                    last_good_roi = new_roi

            data.append(row)
            frames.append(frame.copy())
            fail_count = 0

            if show_window:
                title = f"MMPose tracking | frame {fidx} | score={row_score:.2f} | Q para parar visualizacao"
                disp = draw_focus_overlay(frame, roi=current_roi if current_roi is not None else roi,
                                          landmarks_row=row, bbox=bbox_global, title=title, dim_alpha=0.58)
                cv2.imshow("Processando MMPose - pessoa em foco/tracking", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            # Não salva linha ruim. No modo dinâmico, amplia a última ROI boa para tentar recuperar.
            fail_count += 1
            if dynamic_roi and last_good_roi is not None:
                current_roi = _expand_roi(last_good_roi, frame.shape, grow=min(0.20 * fail_count, 1.00))

            if show_window:
                title = f"MMPose tracking | frame {fidx} | SEM POSE valida | expandindo busca"
                disp = draw_focus_overlay(frame, roi=current_roi if current_roi is not None else roi,
                                          landmarks_row=None, title=title, dim_alpha=0.68)
                cv2.imshow("Processando MMPose - pessoa em foco/tracking", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        fidx += 1

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    return pd.DataFrame(data).reset_index(drop=True), frames, fps, w, h

def _rtmlib_select_subject(keypoints, scores, prev_center=None, score_thr=0.30, min_height=0.0):
    """Escolhe a pessoa principal entre as detecções do rtmlib.

    Critério: maior bbox × confiança média (o sujeito em primeiro plano é de
    longe o maior), com bônus de continuidade temporal para não pular para o
    caminhante de fundo entre frames. Retorna (kp, sc, center) ou (None, None, None).
    """
    best, best_score, best_center = None, -1.0, None
    for kp, sc in zip(keypoints, scores):
        kp = np.asarray(kp, dtype=float)
        sc = np.asarray(sc, dtype=float)
        if kp.ndim != 2 or kp.shape[0] == 0:
            continue
        rel = sc >= score_thr
        if rel.sum() < 6:                      # poucos pontos confiáveis: descarta
            continue
        xs, ys = kp[rel, 0], kp[rel, 1]
        bw = max(float(xs.max() - xs.min()), 1.0)
        bh = max(float(ys.max() - ys.min()), 1.0)
        if bh < min_height:                    # menor que o sujeito (ex.: fundo): descarta
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        s = bw * bh * float(sc[rel].mean())
        if prev_center is not None:
            dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
            s *= 1.0 / (1.0 + dist / 250.0)
        if s > best_score:
            best_score, best, best_center = s, (kp, sc), (cx, cy)
    if best is None:
        return None, None, None
    return best[0], best[1], best_center


def _rtmlib_person_height(kp, sc, score_thr=0.30):
    """Altura (px) da bbox de keypoints confiáveis de uma pessoa; None se poucos pontos."""
    kp = np.asarray(kp, dtype=float)
    sc = np.asarray(sc, dtype=float)
    if kp.ndim != 2 or kp.shape[0] == 0:
        return None
    rel = sc >= score_thr
    if rel.sum() < 6:
        return None
    ys = kp[rel, 1]
    return float(ys.max() - ys.min())


def _rtmlib_track_subject(dets, order, min_height=0.0, subj_h=0.0, score_thr=0.30):
    """Rastreia UM sujeito ao longo do vídeo, travando na trajetória para não pular
    para pessoas de fundo. Estratégia:
      1. Por frame, lista os candidatos que passam o portão de tamanho.
      2. Âncora = candidato de maior 'score' de todo o vídeo (sujeito no auge da presença).
      3. Anda para frente e para trás a partir da âncora, escolhendo em cada frame a
         detecção mais próxima da posição PREVISTA (posição + velocidade). Se o mais
         próximo estiver longe demais (salto = outra pessoa) ou não houver candidato,
         o frame é descartado (sujeito ausente).
    Retorna {fidx: (kp, sc)} só dos frames atribuídos ao sujeito.
    """
    # 1. Candidatos (pessoas) por frame, já filtrados pelo portão de tamanho.
    cand = {}
    for fidx in order:
        kps, scs = dets.get(fidx, (None, None))
        lst = []
        if kps is not None and len(kps):
            for kp, sc in zip(kps, scs):
                kp = np.asarray(kp, float); sc = np.asarray(sc, float)
                if kp.ndim != 2 or kp.shape[0] == 0:
                    continue
                rel = sc >= score_thr
                if rel.sum() < 6:
                    continue
                xs, ys = kp[rel, 0], kp[rel, 1]
                bh = float(ys.max() - ys.min())
                if bh < min_height:
                    continue
                lst.append({"c": (float(xs.mean()), float(ys.mean())), "h": bh, "kp": kp, "sc": sc})
        cand[fidx] = lst

    # 2. Trilhas por associação gulosa. A posição PREVISTA de cada trilha já extrapola
    #    pela sua velocidade, então a tolerância de casamento é FIXA (não cresce com o
    #    gap). Assim uma trilha parada (ex.: árvore/falso positivo, velocidade ≈0) não
    #    "captura" o sujeito que aparece longe — eles ficam em trilhas separadas.
    tol = max(subj_h * 0.55, 160.0)
    tracks = []   # {last_c, last_f, vel, frames:[(fidx, cand)]}
    for fidx in order:
        cs = cand[fidx]
        used = set()
        for tr in sorted(tracks, key=lambda t: t["last_f"], reverse=True):
            gap = fidx - tr["last_f"]
            if gap <= 0 or gap > 5:
                continue
            px = tr["last_c"][0] + tr["vel"][0] * gap
            py = tr["last_c"][1] + tr["vel"][1] * gap
            bi, bd = -1, 1e18
            for i, c in enumerate(cs):
                if i in used:
                    continue
                d = math.hypot(c["c"][0] - px, c["c"][1] - py)
                if d < bd:
                    bd, bi = d, i
            if bi >= 0 and bd <= tol:
                c = cs[bi]; used.add(bi)
                tr["vel"] = ((c["c"][0] - tr["last_c"][0]) / gap, (c["c"][1] - tr["last_c"][1]) / gap)
                tr["last_c"] = c["c"]; tr["last_f"] = fidx
                tr["frames"].append((fidx, c))
        for i, c in enumerate(cs):
            if i not in used:
                tracks.append({"last_c": c["c"], "last_f": fidx, "vel": (0.0, 0.0),
                               "frames": [(fidx, c)]})

    if not tracks:
        return {}

    # 3. Sujeito = trilha que MAIS se desloca horizontalmente (caminha atravessando o
    #    quadro), entre as de presença razoável. O fundo fica ~parado → span pequeno.
    def span(tr):
        xs = [fc[1]["c"][0] for fc in tr["frames"]]
        return (max(xs) - min(xs)) if xs else 0.0
    def presence_size(tr):
        return len(tr["frames"]) * float(np.mean([fc[1]["h"] for fc in tr["frames"]]))
    min_len = max(8, int(0.12 * len(order)))
    valid = [tr for tr in tracks if len(tr["frames"]) >= min_len] or tracks
    subj = max(valid, key=lambda tr: (span(tr), presence_size(tr)))

    # 4. Corta as PONTAS PARADAS da trilha do sujeito. Um falso positivo estático
    #    (ex.: uma árvore detectada como pessoa, do tamanho do sujeito) perto da
    #    trajetória acaba fundido na trilha nos frames antes de entrar / depois de
    #    sair. Como o sujeito está CAMINHANDO, mantemos só o trecho contíguo em que
    #    ele de fato se desloca; as pontas quase-imóveis (a árvore/parado) saem.
    fr = sorted(subj["frames"], key=lambda x: x[0])
    xs = np.array([fc[1]["c"][0] for fc in fr], dtype=float)
    if len(xs) >= 8:
        k = 5
        xs_s = np.convolve(xs, np.ones(k) / k, mode="same")   # suaviza jitter
        v = np.abs(np.gradient(xs_s))
        pos = v[v > 0]
        vmin = max(2.0, 0.30 * float(np.median(pos))) if pos.size else 2.0
        moving = v > vmin
        if moving.any():
            i0 = int(np.argmax(moving))
            i1 = len(moving) - 1 - int(np.argmax(moving[::-1]))
            fr = fr[i0:i1 + 1]

    print(f"[INFO] rtmlib: {len(tracks)} trilha(s); sujeito com deslocamento "
          f"horizontal ~{span(subj):.0f}px; mantidos {len(fr)} frames em movimento.")
    return {fc[0]: (fc[1]["kp"], fc[1]["sc"]) for fc in fr}


def _pick_rtmlib_device(requested="auto"):
    """Decide de forma robusta se dá para usar a GPU.

    'auto'/'cuda' → usa GPU só se o CUDAExecutionProvider estiver listado E as DLLs
    de runtime (cuDNN/cuBLAS) realmente carregarem; senão cai para CPU de forma
    limpa (sem os avisos vermelhos do ONNX ao tentar e falhar). 'cpu' força CPU.
    """
    req = str(requested).lower().strip()
    if req == "cpu":
        return "cpu"
    _enable_cuda_dlls()
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return "cpu"
    except Exception:
        return "cpu"
    # Confirma que o runtime CUDA está de fato presente (não só o pacote -gpu).
    import ctypes
    for dll in ("cudnn64_9.dll", "cublasLt64_12.dll"):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            return "cpu"
    return "cuda"


def process_video_rtmlib(video_path, start_sec=None, end_sec=None, show_window=False,
                         mode="performance", device="auto", progress_cb=None, frame_cb=None):
    """
    Extrai landmarks com rtmlib (RTMDet/YOLOX + RTMPose Halpe-26) via ONNX Runtime.
    Retorna o MESMO formato do MediaPipe: (DataFrame, frames, fps, w, h).

    Vantagens sobre os backends anteriores para este cenário (marcha de perfil,
    pessoa pequena no quadro, caminhante ao fundo):
      - detector de pessoa dedicado → recorte justo e upscalado → keypoints melhores;
      - calcanhar e dedo REAIS dos dois pés (Halpe-26) para eventos de marcha;
      - roda em GPU (Blackwell) via onnxruntime-gpu; sem depender de mmcv/torch-cuda.
    """
    _enable_cuda_dlls()
    try:
        import onnxruntime as ort
        from rtmlib import BodyWithFeet, draw_skeleton
    except Exception as e:
        raise ImportError(
            "rtmlib/onnxruntime-gpu não estão instalados neste ambiente. "
            "Instale com: pip install rtmlib onnxruntime-gpu. Erro original: " + str(e))

    dev = _pick_rtmlib_device(device)
    if str(device).lower().strip() in ("auto", "cuda") and dev == "cpu":
        print("[INFO] GPU indisponível — rodando rtmlib em CPU (mais lento, mesma qualidade).")

    print(f"[INFO] Inicializando rtmlib BodyWithFeet (Halpe-26): mode={mode}, device={dev}")
    model = BodyWithFeet(mode=mode, backend="onnxruntime", device=dev)

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0:
        fps = 30.0
    f_start = int(start_sec * fps) if start_sec is not None else 0
    f_end   = int(end_sec   * fps) if end_sec   is not None else total
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    # ---- Passo 1: inferência em TODOS os frames; guarda detecções e alturas ----
    # A inferência (parte cara) roda uma única vez. As alturas por frame servem para
    # estimar o tamanho do sujeito e, com isso, o portão que descarta gente do fundo.
    dets = {}          # fidx -> (keypoints, scores)
    heights = []       # altura (px) do maior candidato por frame
    n_total = max(f_end - f_start + 1, 1)
    fidx = f_start
    while cap.isOpened() and fidx <= f_end:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            keypoints, scores = model(frame)
        except Exception as e:
            print(f"[AVISO] rtmlib falhou no frame {fidx}: {e}")
            keypoints, scores = None, None
        dets[fidx] = (keypoints, scores)
        kp0 = sc0 = None
        if keypoints is not None and len(keypoints):
            kp0, sc0, _ = _rtmlib_select_subject(keypoints, scores, prev_center=None)
            if kp0 is not None:
                hgt = _rtmlib_person_height(kp0, sc0)
                if hgt:
                    heights.append(hgt)
        if progress_cb:
            progress_cb("Detectando pose (1/2)", fidx - f_start + 1, n_total)
        # Gancho aditivo: entrega o frame CRU (sem esqueleto) para exibição ao vivo em
        # outra UI (ex.: app_marcha). Durante a detecção ainda não se sabe quem é o
        # sujeito, então não marcamos ninguém — o app mostra o esqueleto só no replay
        # final, já com o sujeito rastreado. Só é usado quando frame_cb é passado.
        if frame_cb is not None:
            try:
                frame_cb(frame, fidx - f_start + 1, n_total)
            except Exception:
                pass
        fidx += 1
    cap.release()

    # ---- Estima o tamanho do sujeito e define o portão de tamanho ----
    min_h, subj_h = 0.0, 0.0
    if heights:
        subj_h = float(np.percentile(heights, RTMLIB_SUBJECT_HEIGHT_PCTL))
        min_h = RTMLIB_SUBJECT_MIN_FRAC * subj_h
        print(f"[INFO] rtmlib: altura típica do sujeito ~{subj_h:.0f}px | "
              f"gate de tamanho = {min_h:.0f}px (descarta pessoas menores, ex.: fundo)")

    # ---- Rastreamento do sujeito: trava na trajetória (não pula para o fundo) ----
    order = sorted(dets.keys())
    chosen = _rtmlib_track_subject(dets, order, min_height=min_h, subj_h=subj_h)
    print(f"[INFO] rtmlib: rastreamento travou o sujeito em {len(chosen)}/{len(order)} frames.")

    # ---- Passo 2: reabre o vídeo, usa a trajetória rastreada e monta as linhas ----
    data, frames = [], []

    def _store(fidx, frame, kp, sc):
        row = {"frame": fidx}
        for nm, idx in RTMLIB_HALPE26_MAP.items():
            if idx < kp.shape[0]:
                row[f"{nm}_x_px"] = float(kp[idx, 0])
                row[f"{nm}_y_px"] = float(kp[idx, 1])
                row[f"{nm}_vis"]  = float(sc[idx]) if idx < sc.shape[0] else 0.0
        for nm in LANDMARK_MAP.values():
            row.setdefault(f"{nm}_x_px", np.nan)
            row.setdefault(f"{nm}_y_px", np.nan)
            row.setdefault(f"{nm}_vis",  0.0)
        data.append(row)
        frames.append(frame.copy())

    def _run_pass(track):
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, f_start)
        prev_center = None
        j = f_start
        while cap2.isOpened() and j <= f_end:
            ret, frame = cap2.read()
            if not ret:
                break
            kp = sc = None
            if track is not None:
                if j in track:
                    kp, sc = track[j]
            else:
                keypoints, scores = dets.get(j, (None, None))
                if keypoints is not None and len(keypoints):
                    kp, sc, center = _rtmlib_select_subject(keypoints, scores,
                                                            prev_center=prev_center, min_height=0.0)
                    if center is not None:
                        prev_center = center
            if kp is not None:
                _store(j, frame, kp, sc)
            if progress_cb:
                progress_cb("Selecionando sujeito (2/2)", j - f_start + 1, n_total)
            if show_window:
                if kp is not None:
                    disp = draw_skeleton(frame.copy(), kp[None, ...], sc[None, ...],
                                         openpose_skeleton=False, kpt_thr=0.3)
                else:
                    disp = frame
                cv2.putText(disp, f"rtmlib Halpe26 | frame {j} | Q para parar",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Processando rtmlib (RTMPose Halpe-26)", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap2.release()
                    return
            j += 1
        cap2.release()

    _run_pass(chosen)

    # Rede de segurança: se o rastreamento não pegou nada, refaz sem trilha (maior×conf).
    if not data:
        print("[AVISO] rtmlib: rastreamento sem frames — refazendo com seleção simples.")
        _run_pass(None)

    if show_window:
        cv2.destroyAllWindows()

    return pd.DataFrame(data).reset_index(drop=True), frames, fps, w, h


def process_video(video_path, start_sec=None, end_sec=None, show_window=False,
                  backend="mediapipe", mmpose_model="wholebody", mmpose_device="cuda:0",
                  mmpose_whole_image=True, mmpose_roi=None, mmpose_dynamic_roi=False,
                  rtmlib_mode="performance", rtmlib_device="auto", progress_cb=None, frame_cb=None):
    """Wrapper único para preservar o restante do pipeline."""
    backend = str(backend).lower().strip()
    if backend == "mediapipe":
        return process_video_mediapipe(video_path, start_sec, end_sec, show_window)
    if backend == "mmpose":
        return process_video_mmpose(video_path, start_sec, end_sec, show_window,
                                   model_alias=mmpose_model,
                                   device=mmpose_device,
                                   use_whole_image=mmpose_whole_image,
                                   roi=mmpose_roi,
                                   dynamic_roi=mmpose_dynamic_roi)
    if backend == "rtmlib":
        return process_video_rtmlib(video_path, start_sec, end_sec, show_window,
                                    mode=rtmlib_mode, device=rtmlib_device,
                                    progress_cb=progress_cb, frame_cb=frame_cb)
    raise ValueError(f"Backend desconhecido: {backend}")

# ===================== Pipeline de Dados =====================
def fix_feet(df):
    """Interpola todos os landmarks com visibilidade abaixo do limiar."""
    df = df.copy()
    for nm in LANDMARK_MAP.values():
        vis_col = f"{nm}_vis"
        if vis_col not in df.columns:
            continue
        vis = df[vis_col].values
        rel = vis >= VIS_THRESHOLD
        for ax in ('x_px', 'y_px'):
            col = f"{nm}_{ax}"
            if col not in df.columns:
                continue
            vals = df[col].values.copy().astype(float)
            vals[~rel] = np.nan
            df[col] = interpolate_gaps(vals, rel)
    return df


def apply_scale(df, scale_x_func, scale_y_func):
    """Converte pixel → metro para todos os landmarks.

    A escala vertical é local (função de pixel_x), corrigindo a perspectiva
    de profundidade para cada landmark de acordo com sua posição horizontal.
    """
    df = df.copy()
    for nm in LANDMARK_MAP.values():
        x_px = df[f"{nm}_x_px"].values
        df[f"{nm}_x_m"] = scale_x_func(x_px)
        df[f"{nm}_y_m"] = df[f"{nm}_y_px"].values * scale_y_func(x_px)
    return df


def filter_coords(df, fs, cutoff=CUTOFF_HZ):
    """Aplica filtro passa-baixa Butterworth nas coordenadas em metros."""
    df = df.copy()
    for col in [c for c in df.columns if c.endswith("_m")]:
        df[col] = butter_lp(df[col].values, cutoff, fs)
    return df


def compute_angles(df):
    """Calcula ângulos bilaterais: quadril, joelho, tornozelo, pé, tronco."""
    df = df.copy()

    for side in ('dir', 'esq'):
        def pt(name):
            return np.column_stack([df[f"{name}_x_m"].values,
                                    df[f"{name}_y_m"].values])

        ombro     = pt(f"ombro_{side}")
        quadril   = pt(f"quadril_{side}")
        joelho    = pt(f"joelho_{side}")
        tornozelo = pt(f"tornozelo_{side}")
        calcanhar = pt(f"calcanhar_{side}")
        ponta     = pt(f"ponta_{side}")

        hip_a, knee_a, ank_a, foot_a = [], [], [], []
        for i in range(len(df)):
            hip_a.append( hip_flexion(ombro[i], quadril[i], joelho[i]) )
            knee_a.append( knee_flexion(quadril[i], joelho[i], tornozelo[i]) )
            ank_a.append( ankle_angle(joelho[i], tornozelo[i], ponta[i]) )
            fv = ponta[i] - calcanhar[i]
            foot_a.append( math.degrees(math.atan2(-fv[1], fv[0])) )   # ângulo do pé c/ horizontal

        df[f"hip_angle_{side}"]    = hip_a
        df[f"knee_angle_{side}"]   = knee_a
        df[f"ankle_angle_{side}"]  = ank_a
        df[f"foot_angle_{side}"]   = foot_a

    # Tronco (usando médias)
    om_mid = np.column_stack([
        (df["ombro_dir_x_m"]  + df["ombro_esq_x_m"])  / 2,
        (df["ombro_dir_y_m"]  + df["ombro_esq_y_m"])  / 2,
    ])
    qd_mid = np.column_stack([
        (df["quadril_dir_x_m"] + df["quadril_esq_x_m"]) / 2,
        (df["quadril_dir_y_m"] + df["quadril_esq_y_m"]) / 2,
    ])
    df["trunk_angle"] = [trunk_inclination(om_mid[i], qd_mid[i]) for i in range(len(df))]
    return df


def filter_angles(df, fs, cutoff=CUTOFF_HZ):
    """Filtra colunas de ângulo."""
    df = df.copy()
    for col in [c for c in df.columns if "angle" in c]:
        df[f"{col}_filt"] = butter_lp(df[col].ffill().bfill().values, cutoff, fs)
    return df

# ===================== Estimativa de COM =====================
def estimate_com(df):
    """
    Estima a posição do COM global por modelo multi-segmentar (De Leva 1996).
    Retorna arrays x_com, y_com em metros.
    """
    segs = {
        'trunk':      ('ombro_dir',    'quadril_dir',   'ombro_esq',   'quadril_esq',  True),
        'thigh_dir':  ('quadril_dir',  'joelho_dir',    None, None, False),
        'thigh_esq':  ('quadril_esq',  'joelho_esq',    None, None, False),
        'shank_dir':  ('joelho_dir',   'tornozelo_dir', None, None, False),
        'shank_esq':  ('joelho_esq',   'tornozelo_esq', None, None, False),
        'foot_dir':   ('calcanhar_dir','ponta_dir',     None, None, False),
        'foot_esq':   ('calcanhar_esq','ponta_esq',     None, None, False),
    }

    n = len(df)
    x_com = np.zeros(n)
    y_com = np.zeros(n)

    for seg, (p1, d1, p2, d2, bilateral_avg) in segs.items():
        mfrac, cfrac = SEG_PARAMS[seg]
        if bilateral_avg:
            # tronco: média bilateral dos endpoints
            px = (df[f"{p1}_x_m"].values + df[f"{p2}_x_m"].values) / 2
            py = (df[f"{p1}_y_m"].values + df[f"{p2}_y_m"].values) / 2
            dx = (df[f"{d1}_x_m"].values + df[f"{d2}_x_m"].values) / 2
            dy = (df[f"{d1}_y_m"].values + df[f"{d2}_y_m"].values) / 2
        else:
            px = df[f"{p1}_x_m"].values
            py = df[f"{p1}_y_m"].values
            dx = df[f"{d1}_x_m"].values
            dy = df[f"{d1}_y_m"].values

        x_com += mfrac * (px + cfrac * (dx - px))
        y_com += mfrac * (py + cfrac * (dy - py))

    x_com /= SEG_MASS_TOTAL
    y_com /= SEG_MASS_TOTAL
    return x_com, y_com


def compute_mechanical_work(x_com, y_com, body_mass, fs, leg_length=None):
    """
    Trabalho mecânico externo + mecanismo pendular (Cavagna et al.).

    Convenção de imagem: y_com aumenta para baixo.
    Altura real: h = -y_com (invertido) → colunas energéticas positivas.

    Wv = trabalho para elevar o COM (incrementos positivos de Ep)
    Wf = trabalho para acelerar o COM para frente (incrementos positivos de Ekf)
    Ek é separado em Ekf (forward) e Ekv (vertical).
    IRL = Recovery × (v_media / v_otima), v_otima = sqrt(g × L_membro)
    Referência IRL: Tartaruga et al.
    """
    g  = 9.81
    M  = body_mass
    dt = 1.0 / fs

    h   = -(y_com - y_com.mean())

    vx  = np.gradient(x_com, dt)
    vy  = -np.gradient(y_com, dt)      # para cima = positivo

    Ep   = M * g * h
    Ekf  = 0.5 * M * vx**2            # energia cinética forward (horizontal)
    Ekv  = 0.5 * M * vy**2            # energia cinética vertical
    Ek   = Ekf + Ekv                   # energia cinética total
    Emec = Ep + Ek

    def pos_inc(E):
        dE = np.diff(E)
        return float(dE[dE > 0].sum())

    Wv    = pos_inc(Ep)                # trabalho vertical (levantar COM)
    Wf    = pos_inc(Ekf)               # trabalho forward (acelerar COM)
    W_ext = pos_inc(Emec)

    M_s   = M if M > 0 else np.nan
    total_disp = abs(x_com[-1] - x_com[0]) if len(x_com) else np.nan
    dist  = (M_s * total_disp) if (total_disp and total_disp > 0) else np.nan

    Wv_pkg  = Wv    / M_s  if M_s  else np.nan
    Wf_pkg  = Wf    / M_s  if M_s  else np.nan
    Wext_pk = W_ext / M_s  if M_s  else np.nan
    Wv_pkm  = Wv    / dist if dist else np.nan
    Wf_pkm  = Wf    / dist if dist else np.nan
    Wext_pkm= W_ext / dist if dist else np.nan

    denom = Wv + Wf
    R = (denom - W_ext) / denom * 100.0 if denom > 0 else np.nan

    mean_speed = float(np.abs(np.nanmean(vx)))

    # IRL = Recovery × (v_media / v_otima); v_otima = sqrt(g × L_membro)
    if leg_length and leg_length > 0 and not np.isnan(R):
        v_otima = math.sqrt(g * leg_length)
        IRL = (R / 100.0) * (mean_speed / v_otima)
    else:
        IRL = np.nan

    # LRI_Tartaruga = 100 × SSWS / OWS  (Peyré-Tartaruga & Monteiro, 2016)
    # OWS = sqrt(Fr_otimo × g × LLL), Fr_otimo = 0.25  →  OWS = 0.5 × sqrt(g × LLL)
    # Resultado em % — 100% indica que a velocidade selecionada = velocidade ótima
    if leg_length and leg_length > 0:
        OWS = 0.5 * math.sqrt(g * leg_length)
        LRI_Tartaruga = 100.0 * mean_speed / OWS
    else:
        LRI_Tartaruga = np.nan

    return {
        "W_ext_J":          W_ext,
        "W_ext_J_per_kg":   Wext_pk,
        "W_ext_J_per_kg_m": Wext_pkm,
        "Wv_J":             Wv,
        "Wv_J_per_kg":      Wv_pkg,
        "Wv_J_per_kg_m":    Wv_pkm,
        "Wf_J":             Wf,
        "Wf_J_per_kg":      Wf_pkg,
        "Wf_J_per_kg_m":    Wf_pkm,
        "Recovery_pct":     R,
        "IRL":              IRL,
        "LRI_Tartaruga":    LRI_Tartaruga,
        "Mean_Speed_ms":    mean_speed,
        "Total_Disp_m":     total_disp,
        # séries temporais para gráficos e cálculos por passo
        "_Ep": Ep, "_Ekf": Ekf, "_Ekv": Ekv, "_Ek": Ek, "_Emec": Emec,
        "_x_com": x_com, "_y_com": y_com, "_vx": vx, "_vy": vy,
    }

# ===================== Detecção de Eventos =====================
def _walk_direction(x_com):
    """Retorna +1 (esq→dir) ou -1 (dir→esq) baseado na direção do COM."""
    dx = np.diff(x_com)
    return 1 if np.nanmean(dx) > 0 else -1


def detect_events(df, fs, x_com, walk_dir=None):
    """
    Detecta TD e TO bilaterais.
    Usa calcanhar-quadril para TD; ponta-quadril para TO.
    Fallback para tornozelo-quadril se visibilidade do calcanhar for ruim.

    walk_dir: +1 = IDA (esq→dir no vídeo), -1 = VOLTA (dir→esq).
    Se None, auto-detecta pelo COM.
    """
    if walk_dir is None:
        walk_dir = _walk_direction(x_com)
    results = {}

    for side, h in (('R', 'dir'), ('L', 'esq')):
        heel_vis = df[f"calcanhar_{h}_vis"].values
        good_heel_frac = (heel_vis >= VIS_THRESHOLD).mean()

        if good_heel_frac >= 0.5:
            td_raw = (df[f"calcanhar_{h}_x_m"] - df[f"quadril_{h}_x_m"]).values
        else:
            # fallback: tornozelo mais visível na grama
            td_raw = (df[f"tornozelo_{h}_x_m"] - df[f"quadril_{h}_x_m"]).values

        to_raw = (df[f"ponta_{h}_x_m"] - df[f"quadril_{h}_x_m"]).values

        # Ajusta sinal ao sentido de caminhada (VOLTA inverte os picos)
        td_sig = butter_lp(td_raw * walk_dir, 3, fs)
        to_sig = butter_lp(to_raw * walk_dir, 3, fs)

        min_dist = max(int(fs * 0.7), 5)
        prom     = np.std(td_sig) * 0.15

        TD, _ = find_peaks(td_sig, distance=min_dist, prominence=prom)
        TO    = _refine_to(TD, to_sig, fs)

        results[f"TD_{side}"]     = TD
        results[f"TO_{side}"]     = TO
        results[f"td_sig_{side}"] = td_sig
        results[f"to_sig_{side}"] = to_sig

    return results


def _refine_to(TD, to_signal, fs, guard_s=0.05):
    """Encontra TO como mínimo do sinal ponta-quadril entre TDs consecutivos."""
    TO  = []
    g   = int(guard_s * fs)
    for i in range(len(TD) - 1):
        a = TD[i] + g
        b = TD[i+1] - g
        if b > a:
            seg = to_signal[a:b]
            TO.append(a + int(np.argmin(seg)))
    return np.array(TO, dtype=int)



def _fill_short_nan_gaps(sig, max_gap=6):
    """Preenche buracos curtos para cálculo de velocidade/sinais auxiliares."""
    sig = np.asarray(sig, dtype=float).copy()
    n = len(sig)
    bad = ~np.isfinite(sig)
    if not bad.any():
        return sig
    idx = np.arange(n)
    good = ~bad
    if good.sum() < 2:
        return sig
    interp = np.interp(idx, idx[good], sig[good])
    # Só usa interpolação para gaps curtos.
    i = 0
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < n and bad[j]:
            j += 1
        if (j - i) <= max_gap:
            sig[i:j] = interp[i:j]
        i = j
    return sig


def _prune_events_by_interval(events, fs, min_s=0.45, max_s=1.80):
    """Remove eventos muito próximos e preserva os de maior plausibilidade temporal."""
    events = np.asarray(sorted(set(map(int, events))), dtype=int)
    if len(events) <= 1:
        return events
    kept = [events[0]]
    for ev in events[1:]:
        dt = (ev - kept[-1]) / fs
        if dt >= min_s:
            kept.append(ev)
        else:
            # Eventos colados: mantém o mais tardio só se o intervalo anterior ficar menos ruim.
            if len(kept) >= 2 and (ev - kept[-2]) / fs >= min_s:
                kept[-1] = ev
    kept = np.asarray(kept, dtype=int)
    if len(kept) > 2:
        intervals = np.diff(kept) / fs
        valid = np.r_[True, intervals <= max_s]
        kept = kept[valid]
    return kept


def detect_events_robust(df, fs, x_com, walk_dir=None):
    """
    Detector alternativo de TD/TO por votação simples.
    Não substitui revisão manual: só entrega eventos iniciais mais estáveis.
    Combina calcanhar, tornozelo e ponta com peso por visibilidade.
    """
    if walk_dir is None:
        walk_dir = _walk_direction(x_com)

    results = {}
    for side, h in (('R', 'dir'), ('L', 'esq')):
        heel = (df[f"calcanhar_{h}_x_m"] - df[f"quadril_{h}_x_m"]).values * walk_dir
        ankle = (df[f"tornozelo_{h}_x_m"] - df[f"quadril_{h}_x_m"]).values * walk_dir
        toe = (df[f"ponta_{h}_x_m"] - df[f"quadril_{h}_x_m"]).values * walk_dir

        heel_v = df[f"calcanhar_{h}_vis"].values
        ankle_v = df[f"tornozelo_{h}_vis"].values
        toe_v = df[f"ponta_{h}_vis"].values

        heel = _fill_short_nan_gaps(heel)
        ankle = _fill_short_nan_gaps(ankle)
        toe = _fill_short_nan_gaps(toe)

        # Pesos globais de confiabilidade: se calcanhar estiver ruim, tornozelo assume mais.
        wh = float(np.nanmean(heel_v >= VIS_THRESHOLD))
        wa = float(np.nanmean(ankle_v >= VIS_THRESHOLD))
        wt = float(np.nanmean(toe_v >= VIS_THRESHOLD))
        denom = max(wh + wa, 1e-6)
        td_mix = (wh * heel + wa * ankle) / denom
        to_mix = toe if wt >= 0.35 else 0.6 * toe + 0.4 * ankle

        td_sig = butter_lp(td_mix, 3, fs)
        to_sig = butter_lp(to_mix, 3, fs)

        # Sinal auxiliar: pé mais parado perto do contato.
        foot_x = df[f"tornozelo_{h}_x_m"].values * walk_dir
        foot_x = _fill_short_nan_gaps(foot_x)
        vfoot = butter_lp(np.gradient(foot_x, 1/fs), 4, fs)
        stillness = -np.abs(vfoot)
        stillness = (stillness - np.nanmean(stillness)) / (np.nanstd(stillness) + 1e-9)
        td_z = (td_sig - np.nanmean(td_sig)) / (np.nanstd(td_sig) + 1e-9)
        td_score = td_z + 0.20 * stillness

        min_dist = max(int(fs * 0.55), 5)
        prom = max(float(np.nanstd(td_score) * 0.10), 1e-6)
        TD, _ = find_peaks(td_score, distance=min_dist, prominence=prom)
        TD = _prune_events_by_interval(TD, fs, min_s=0.45, max_s=1.80)

        # TO: mínimo da ponta/quadril entre TDs, evitando bordas do intervalo.
        TO = []
        for i in range(len(TD) - 1):
            a = TD[i] + max(int(0.08 * fs), 1)
            b = TD[i+1] - max(int(0.08 * fs), 1)
            if b <= a:
                continue
            seg = to_sig[a:b]
            if len(seg) >= 3:
                TO.append(a + int(np.argmin(seg)))
        TO = np.asarray(TO, dtype=int)

        results[f"TD_{side}"] = TD
        results[f"TO_{side}"] = TO
        results[f"td_sig_{side}"] = td_sig
        results[f"to_sig_{side}"] = to_sig

    return results

# ===================== Parâmetros Espaço-Temporais =====================
def build_spatiotemporal(df, evts, fs, x_com, body_mass, leg_length):
    """
    Computa todos os parâmetros espaço-temporais bilaterais.
    Retorna DataFrames de passos, passadas e médias.
    """
    TD_R, TO_R = evts["TD_R"], evts["TO_R"]
    TD_L, TO_L = evts["TD_L"], evts["TO_L"]

    walk_dir = _walk_direction(x_com)

    # ---- velocidade ----
    dur = (len(df) - 1) / fs
    disp = abs(x_com[-1] - x_com[0])
    speed = disp / dur if dur > 0 else np.nan

    # ---- helper: coluna de posição horizontal do quadril ----
    def qx(side):
        return df[f"quadril_{side}_x_m"].values * walk_dir

    rows_steps    = []
    rows_strides  = []
    step_cols = [
        "Lado", "TD_Frame", "TC_Contra_Frame", "TO_Frame", "Tempo_Passo_s",
        "Comprimento_Passo_m", "Cadencia_passo_Hz", "Apoio_s", "Balanco_s",
        "Ciclo_s", "Duty_Factor", "Duplo_Apoio_s", "Velocidade_ms",
        "Quadril_TD_deg", "Joelho_TD_deg", "Tornozelo_TD_deg",
        "Quadril_TO_deg", "Joelho_TO_deg", "Tornozelo_TO_deg",
    ]

    # ---- helper: ângulo filtrado de um lado em um frame específico ----
    def ang_at(side, frame, ang):
        col = f"{ang}_angle_{side}_filt"
        if col not in df.columns or pd.isna(frame):
            return np.nan
        idx = int(frame)
        if not (0 <= idx < len(df)):
            return np.nan
        return df[col].iloc[idx]
    stride_cols = [
        "Lado", "Ordem", "TD_Frame", "Prox_TD_Frame", "Tempo_Passada_s",
        "Comprimento_Passada_m", "Cadencia_passada_Hz",
    ]

    for (side_label, side, TD, TO, TO_contra, TD_contra, hip_col) in [
        ("Direita",  "dir", TD_R, TO_R, TO_L, TD_L, qx("dir")),
        ("Esquerda", "esq", TD_L, TO_L, TO_R, TD_R, qx("esq")),
    ]:
        # Passadas
        for i in range(len(TD) - 1):
            t0, t1 = TD[i], TD[i+1]
            time_pa = (t1 - t0) / fs
            comp_pa = abs(hip_col[t1] - hip_col[t0])
            rows_strides.append({
                "Lado": side_label,
                "Ordem": i + 1,
                "TD_Frame": int(t0),
                "Prox_TD_Frame": int(t1),
                "Tempo_Passada_s": time_pa,
                "Comprimento_Passada_m": comp_pa,
                "Cadencia_passada_Hz": 1.0 / time_pa if time_pa > 0 else np.nan,
            })

        # Passos (até o TD do lado contralateral)
        for i in range(len(TD) - 1):
            tr  = TD[i]
            trn = TD[i + 1]
            mids = [tl for tl in TD_contra if tr < tl < trn]
            if not mids:
                continue
            tc = mids[0]
            time_p = (tc - tr) / fs
            comp_p = abs(hip_col[tc] - hip_col[tr])

            to_mesma = TO[i] if i < len(TO) else np.nan
            stance   = (to_mesma - tr) / fs if not np.isnan(to_mesma) else np.nan
            swing    = ((trn - to_mesma) / fs) if not np.isnan(to_mesma) else np.nan
            cycle    = (trn - tr) / fs
            duty     = stance / cycle if (stance and cycle) else np.nan

            # Duplo apoio: sobreposição com TO contralateral
            to_c_list = [to for to, td in zip(TO_contra, TD_contra) if td <= tr]
            to_c = to_c_list[-1] if to_c_list else np.nan
            ds   = (to_c - tr) / fs if (not np.isnan(to_c) and to_c > tr) else np.nan

            rows_steps.append({
                "Lado": side_label,
                "TD_Frame": int(tr),
                "TC_Contra_Frame": int(tc),
                "TO_Frame": int(to_mesma) if not np.isnan(to_mesma) else np.nan,
                "Tempo_Passo_s": time_p,
                "Comprimento_Passo_m": comp_p,
                "Cadencia_passo_Hz": 1.0 / time_p if time_p > 0 else np.nan,
                "Apoio_s": stance,
                "Balanco_s": swing,
                "Ciclo_s": cycle,
                "Duty_Factor": duty,
                "Duplo_Apoio_s": ds,
                "Velocidade_ms": speed,
                "Quadril_TD_deg":    ang_at(side, tr, "hip"),
                "Joelho_TD_deg":     ang_at(side, tr, "knee"),
                "Tornozelo_TD_deg":  ang_at(side, tr, "ankle"),
                "Quadril_TO_deg":    ang_at(side, to_mesma, "hip"),
                "Joelho_TO_deg":     ang_at(side, to_mesma, "knee"),
                "Tornozelo_TO_deg":  ang_at(side, to_mesma, "ankle"),
            })

    df_steps   = pd.DataFrame(rows_steps, columns=step_cols)
    if not df_steps.empty:
        df_steps = df_steps.sort_values("TD_Frame").reset_index(drop=True)

    df_strides = pd.DataFrame(rows_strides, columns=stride_cols)
    if not df_strides.empty:
        df_strides = df_strides.sort_values(["Lado", "Ordem"]).reset_index(drop=True)

    # Médias por lado
    if not df_steps.empty:
        grp_step = df_steps.groupby("Lado").agg(
            N_Passos              =("Tempo_Passo_s",        "count"),
            Tempo_Passo_s         =("Tempo_Passo_s",        "mean"),
            Comprimento_Passo_m   =("Comprimento_Passo_m",  "mean"),
            Cadencia_Hz           =("Cadencia_passo_Hz",    "mean"),
            Apoio_s               =("Apoio_s",              "mean"),
            Balanco_s             =("Balanco_s",            "mean"),
            Duty_Factor           =("Duty_Factor",          "mean"),
            Duplo_Apoio_s         =("Duplo_Apoio_s",        "mean"),
            Velocidade_ms         =("Velocidade_ms",        "first"),
            Quadril_TD_deg        =("Quadril_TD_deg",       "mean"),
            Joelho_TD_deg         =("Joelho_TD_deg",        "mean"),
            Tornozelo_TD_deg      =("Tornozelo_TD_deg",     "mean"),
            Quadril_TO_deg        =("Quadril_TO_deg",       "mean"),
            Joelho_TO_deg         =("Joelho_TO_deg",        "mean"),
            Tornozelo_TO_deg      =("Tornozelo_TO_deg",     "mean"),
        ).reset_index()
    else:
        grp_step = pd.DataFrame(columns=[
            "Lado", "N_Passos", "Tempo_Passo_s", "Comprimento_Passo_m",
            "Cadencia_Hz", "Apoio_s", "Balanco_s", "Duty_Factor",
            "Duplo_Apoio_s", "Velocidade_ms",
            "Quadril_TD_deg", "Joelho_TD_deg", "Tornozelo_TD_deg",
            "Quadril_TO_deg", "Joelho_TO_deg", "Tornozelo_TO_deg",
        ])

    if not df_strides.empty:
        grp_stride = df_strides.groupby("Lado").agg(
            N_Passadas            =("Tempo_Passada_s",       "count"),
            Tempo_Passada_s       =("Tempo_Passada_s",       "mean"),
            Comprimento_Passada_m =("Comprimento_Passada_m", "mean"),
            Cadencia_passada_Hz   =("Cadencia_passada_Hz",   "mean"),
        ).reset_index()
    else:
        grp_stride = pd.DataFrame(columns=[
            "Lado", "N_Passadas", "Tempo_Passada_s",
            "Comprimento_Passada_m", "Cadencia_passada_Hz",
        ])

    return df_steps, df_strides, grp_step, grp_stride

# ===================== Parâmetros Angulares por Passada =====================
def angular_per_stride(df, TD_R, fs):
    """Extrai ROM angular por passada (lado direito como referência de ciclo)."""
    rows = []
    for i in range(len(TD_R) - 1):
        s, e = TD_R[i], TD_R[i+1]
        cyc  = df.iloc[s:e]
        if len(cyc) < 3:
            continue
        row_d = {"Passada": i+1, "Tempo_s": (e - s) / fs}
        for side in ('dir', 'esq'):
            for ang in ('hip', 'knee', 'ankle', 'foot'):
                col = f"{ang}_angle_{side}_filt"
                if col in df.columns:
                    vals = cyc[col].dropna()
                    if len(vals):
                        row_d[f"{ang}_{side}_max"] = vals.max()
                        row_d[f"{ang}_{side}_min"] = vals.min()
                        row_d[f"{ang}_{side}_ROM"] = vals.max() - vals.min()
        trunk_col = "trunk_angle_filt"
        if trunk_col in df.columns:
            vals = cyc[trunk_col].dropna()
            if len(vals):
                row_d["trunk_max"]  = vals.max()
                row_d["trunk_min"]  = vals.min()
                row_d["trunk_ROM"]  = vals.max() - vals.min()
        rows.append(row_d)
    return pd.DataFrame(rows)

# ===================== Mecânica por Passo =====================
def compute_step_mechanics(Ep, Ekf, Ekv, x_com, body_mass, fs, step_events):
    """
    Computa trabalho mecânico, congruência e phase shift por passo.

    step_events: lista de (frame_inicio, frame_fim).
    Wv  = trabalho vertical (incrementos positivos de Ep).
    Wf  = trabalho forward (incrementos positivos de Ekf).
    % Congruência: Bishop et al. 2008, eq. 4.
    Phase shift α/β: Cavagna & Legramandi 2020, eq. 2-3.
    """
    g  = 9.81
    M  = body_mass
    dt = 1.0 / fs

    def pos_inc(E):
        dE = np.diff(E)
        return float(dE[dE > 0].sum()) if len(dE) > 0 else np.nan

    rows = []
    for i, (s, e) in enumerate(step_events):
        if e <= s or (e - s) < 4:
            continue
        ep   = Ep[s:e]
        ekf  = Ekf[s:e]
        ekv  = Ekv[s:e]
        ek   = ekf + ekv
        emec = ep + ek

        Wv_s   = pos_inc(ep)
        Wf_s   = pos_inc(ekf)
        Wext_s = pos_inc(emec)

        denom = Wv_s + Wf_s
        R_s   = (denom - Wext_s) / denom * 100.0 if denom > 0 else np.nan

        M_s       = M if M > 0 else np.nan
        e_clip    = min(e, len(x_com) - 1)
        step_disp = abs(float(x_com[e_clip]) - float(x_com[s]))
        dist      = M_s * step_disp if (M_s and step_disp > 0) else np.nan

        Wv_pkg   = Wv_s   / M_s  if M_s  else np.nan
        Wf_pkg   = Wf_s   / M_s  if M_s  else np.nan
        Wext_pkg = Wext_s / M_s  if M_s  else np.nan
        Wv_pkm   = Wv_s   / dist if dist else np.nan
        Wf_pkm   = Wf_s   / dist if dist else np.nan
        Wext_pkm = Wext_s / dist if dist else np.nan

        # % Congruência (Bishop 2008, eq. 4): dEp/dt × dEk/dt
        dEp  = np.gradient(ep, dt)
        dEk  = np.gradient(ek, dt)
        cong = dEp * dEk
        pct_cong = float((cong > 0).mean() * 100.0)

        # Phase shift α e β (Cavagna & Legramandi 2020, eq. 2-3)
        # α = 360° × t_pk+ / τ; t_pk+ = t(max Ek) − t(min Ep)
        # β = 360° × t_pk- / τ; t_pk- = t(min Ek) − t(max Ep)
        step_period = (e - s) / fs
        idx_ek_max = int(np.argmax(ek))
        idx_ep_min = int(np.argmin(ep))
        idx_ek_min = int(np.argmin(ek))
        idx_ep_max = int(np.argmax(ep))
        t_pk_plus  = (idx_ek_max - idx_ep_min) / fs
        t_pk_minus = (idx_ek_min - idx_ep_max) / fs
        alpha = 360.0 * t_pk_plus  / step_period if step_period > 0 else np.nan
        beta  = 360.0 * t_pk_minus / step_period if step_period > 0 else np.nan

        rows.append({
            "Passo":                i + 1,
            "Frame_Inicio":         int(s),
            "Frame_Fim":            int(e),
            "Tempo_Passo_s":        step_period,
            "Wv_J":                 Wv_s,
            "Wf_J":                 Wf_s,
            "W_ext_J":              Wext_s,
            "Wv_J_per_kg":          Wv_pkg,
            "Wf_J_per_kg":          Wf_pkg,
            "W_ext_J_per_kg":       Wext_pkg,
            "Wv_J_per_kg_m":        Wv_pkm,
            "Wf_J_per_kg_m":        Wf_pkm,
            "W_ext_J_per_kg_m":     Wext_pkm,
            "Recovery_pct":         R_s,
            "Congruencia_pct":      pct_cong,
            "PhaseShift_alpha_deg": alpha,
            "PhaseShift_beta_deg":  beta,
        })

    return pd.DataFrame(rows)


def interpolate_mechanics_per_step(Ep, Ekf, Ekv, Emec, step_events, n_pts=101):
    """Normaliza curvas de energia para 0–100% por passo."""
    result = {}
    pct = np.linspace(0, 100, n_pts)
    for i, (s, e) in enumerate(step_events):
        if e <= s or (e - s) < 4:
            continue
        result[f"Passo_{i+1}"] = pd.DataFrame({
            "Ep_J":   normalize_to_100(Ep[s:e],   n_pts),
            "Ekf_J":  normalize_to_100(Ekf[s:e],  n_pts),
            "Ekv_J":  normalize_to_100(Ekv[s:e],  n_pts),
            "Emec_J": normalize_to_100(Emec[s:e], n_pts),
        }, index=pct)
    return result


def angular_per_step(df, step_events, fs):
    """Extrai ROM angular por passo (TD → TD contralateral)."""
    rows = []
    for i, (s, e) in enumerate(step_events):
        if e <= s or (e - s) < 3:
            continue
        cyc = df.iloc[s:e]
        row = {"Passo": i + 1, "Frame_Inicio": int(s), "Tempo_s": (e - s) / fs}
        for side in ('dir', 'esq'):
            for ang in ('hip', 'knee', 'ankle', 'foot'):
                col = f"{ang}_angle_{side}_filt"
                if col in df.columns:
                    vals = cyc[col].dropna()
                    if len(vals):
                        row[f"{ang}_{side}_max"] = vals.max()
                        row[f"{ang}_{side}_min"] = vals.min()
                        row[f"{ang}_{side}_ROM"] = vals.max() - vals.min()
        trunk_col = "trunk_angle_filt"
        if trunk_col in df.columns:
            vals = cyc[trunk_col].dropna()
            if len(vals):
                row["trunk_max"] = vals.max()
                row["trunk_min"] = vals.min()
                row["trunk_ROM"] = vals.max() - vals.min()
        rows.append(row)
    return pd.DataFrame(rows)


def interpolate_angles_per_step(df, step_events, n_pts=101):
    """Normaliza curvas angulares para 0–100% por passo."""
    result = {}
    for i, (s, e) in enumerate(step_events):
        if e <= s or (e - s) < 3:
            continue
        cyc = {}
        for side in ("dir", "esq"):
            for ang in ("hip", "knee", "ankle", "foot"):
                col = f"{ang}_angle_{side}_filt"
                if col in df.columns:
                    cyc[col] = normalize_to_100(df.iloc[s:e][col].values, n_pts)
        if "trunk_angle_filt" in df.columns:
            cyc["trunk_angle_filt"] = normalize_to_100(
                df.iloc[s:e]["trunk_angle_filt"].values, n_pts)
        result[f"Passo_{i+1}"] = pd.DataFrame(cyc, index=np.linspace(0, 100, n_pts))
    return result


# ===================== GUI: Revisão de Eventos =====================
class EventReviewWindow:
    """
    Janela matplotlib+tkinter para revisar e corrigir eventos TD/TO.
    Clique esquerdo = adicionar evento do tipo selecionado.
    Clique direito  = remover o evento mais próximo do tipo selecionado.
    """
    def __init__(self, parent, df, evts, fs):
        self.df      = df
        self.evts    = {k: list(v) for k, v in evts.items()
                        if not k.startswith("_") and not k.startswith("td_sig") and not k.startswith("to_sig")}
        self.sigs    = {k: v for k, v in evts.items()
                        if k.startswith("td_sig") or k.startswith("to_sig")}
        self.fs      = fs
        self.accepted = False

        self.win = tk.Toplevel(parent)
        self.win.title("Revisão de Eventos — Clique para editar")
        set_window_icon(self.win)
        self.win.state("zoomed")

        # painel de controle
        ctrl = tk.Frame(self.win, bg="#2b2b2b")
        ctrl.pack(side=tk.TOP, fill=tk.X)

        tk.Label(ctrl, text="Modo de edição:", bg="#2b2b2b", fg="white",
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10, pady=5)

        self.mode_var = tk.StringVar(value="TD_R")
        colors = {"TD_R": "#FF4444", "TO_R": "#FF9900",
                  "TD_L": "#4488FF", "TO_L": "#00CCFF"}
        labels = {"TD_R": "TD Direito", "TO_R": "TO Direito",
                  "TD_L": "TD Esquerdo", "TO_L": "TO Esquerdo"}
        for k, lbl in labels.items():
            tk.Radiobutton(ctrl, text=lbl, variable=self.mode_var, value=k,
                           bg="#2b2b2b", fg=colors[k], selectcolor="#444",
                           activebackground="#2b2b2b",
                           font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=8)

        tk.Button(ctrl, text="✓  Aceitar e Continuar", bg="#4CAF50", fg="white",
                  font=("Arial", 11, "bold"), padx=10, pady=4,
                  command=self._accept).pack(side=tk.RIGHT, padx=15, pady=5)

        tk.Label(ctrl,
                 text="Esq-clique = adicionar  |  Dir-clique = remover",
                 bg="#2b2b2b", fg="#AAAAAA", font=("Arial", 9)).pack(side=tk.RIGHT, padx=10)

        # figura
        self.fig, self.axes = plt.subplots(4, 1, figsize=(15, 9),
                                           sharex=True, facecolor="#1e1e1e")
        self.fig.subplots_adjust(hspace=0.15)
        for ax in self.axes:
            ax.set_facecolor("#1e1e1e")
            for sp in ax.spines.values():
                sp.set_color("#555")
            ax.tick_params(colors="#aaa")
            ax.yaxis.label.set_color("#ccc")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.time_ax = np.arange(len(df)) / fs
        self._event_colors = {"TD_R": "#FF4444", "TO_R": "#FF9900",
                               "TD_L": "#4488FF", "TO_L": "#00CCFF"}
        self._draw()
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.win.wait_window()

    def _draw(self):
        t = self.time_ax
        ax = self.axes

        # ── Eixo 0: sinal TD/TO Direito ──
        ax[0].cla(); ax[0].set_facecolor("#1e1e1e")
        if "td_sig_R" in self.sigs:
            ax[0].plot(t, self.sigs["td_sig_R"], color="#aaa", lw=0.8, label="Calcanhar-Quadril Dir")
        if "to_sig_R" in self.sigs:
            ax[0].plot(t, self.sigs["to_sig_R"], color="#ddd", lw=0.8, ls="--", label="Ponta-Quadril Dir")
        for fr in self.evts.get("TD_R", []):
            ax[0].axvline(fr/self.fs, color="#FF4444", lw=1.4, ls="--")
        for fr in self.evts.get("TO_R", []):
            ax[0].axvline(fr/self.fs, color="#FF9900", lw=1.4, ls=":")
        ax[0].set_ylabel("Direito (m)", color="#ccc")
        ax[0].legend(fontsize=7, facecolor="#333", labelcolor="#ccc")

        # ── Eixo 1: sinal TD/TO Esquerdo ──
        ax[1].cla(); ax[1].set_facecolor("#1e1e1e")
        if "td_sig_L" in self.sigs:
            ax[1].plot(t, self.sigs["td_sig_L"], color="#aaa", lw=0.8, label="Calcanhar-Quadril Esq")
        if "to_sig_L" in self.sigs:
            ax[1].plot(t, self.sigs["to_sig_L"], color="#ddd", lw=0.8, ls="--", label="Ponta-Quadril Esq")
        for fr in self.evts.get("TD_L", []):
            ax[1].axvline(fr/self.fs, color="#4488FF", lw=1.4, ls="--")
        for fr in self.evts.get("TO_L", []):
            ax[1].axvline(fr/self.fs, color="#00CCFF", lw=1.4, ls=":")
        ax[1].set_ylabel("Esquerdo (m)", color="#ccc")
        ax[1].legend(fontsize=7, facecolor="#333", labelcolor="#ccc")

        # ── Eixo 2: ângulos de quadril ──
        ax[2].cla(); ax[2].set_facecolor("#1e1e1e")
        for side, col_c, lbl in [("dir","#FF6666","Quadril Dir"), ("esq","#6699FF","Quadril Esq")]:
            col = f"hip_angle_{side}_filt"
            if col in self.df.columns:
                ax[2].plot(t, self.df[col], color=col_c, lw=1, label=lbl)
        ax[2].set_ylabel("Quadril (°)", color="#ccc")
        ax[2].legend(fontsize=7, facecolor="#333", labelcolor="#ccc")

        # ── Eixo 3: ângulos de joelho ──
        ax[3].cla(); ax[3].set_facecolor("#1e1e1e")
        for side, col_c, lbl in [("dir","#FF6666","Joelho Dir"), ("esq","#6699FF","Joelho Esq")]:
            col = f"knee_angle_{side}_filt"
            if col in self.df.columns:
                ax[3].plot(t, self.df[col], color=col_c, lw=1, label=lbl)
        ax[3].set_ylabel("Joelho (°)", color="#ccc")
        ax[3].set_xlabel("Tempo (s)", color="#ccc")
        ax[3].legend(fontsize=7, facecolor="#333", labelcolor="#ccc")

        for a in self.axes:
            a.tick_params(colors="#aaa")
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_click(self, event):
        if event.inaxes is None or event.xdata is None:
            return
        frame = int(np.clip(event.xdata * self.fs, 0, len(self.df) - 1))
        mode  = self.mode_var.get()
        lst   = self.evts.setdefault(mode, [])
        if event.button == 1:
            lst.append(frame); lst.sort()
        elif event.button == 3 and lst:
            idx = min(range(len(lst)), key=lambda i: abs(lst[i] - frame))
            lst.pop(idx)
        self._draw()

    def _accept(self):
        self.accepted = True
        self.win.destroy()

    def get_events(self):
        return {k: np.array(v, dtype=int) for k, v in self.evts.items()}

# ===================== GUI: Seleção de Parâmetros =====================
def ask_parameters(root, video_path, df_excel):
    """
    Janela de configuração inicial com auto-preenchimento pelo Excel.
    Retorna dict com todos os parâmetros ou None se cancelado.
    """
    params = {}
    win = tk.Toplevel(root)
    win.title("Configuração do Ensaio")
    win.resizable(False, False)
    win.configure(bg=UI["bg"])
    apply_clinical_theme(win)
    set_window_icon(win)
    win.grab_set()
    make_header(win, "Configuração do ensaio",
                "Preencha os dados do sujeito e do vídeo. Pose e direção são automáticas.")

    row_excel = None
    basename  = os.path.splitext(os.path.basename(video_path))[0].upper()
    if df_excel is not None:
        parts = basename.split("_")
        if len(parts) >= 4:
            subject = parts[0]
            speed   = parts[1]
            try:
                load = float(parts[2].replace("KG", ""))
            except ValueError:
                load = None
            direction = parts[3]

            def row_match(r):
                vid = str(r.get("VIDEO NAME", "")).upper()
                spd = str(r.get("SPEED",      "")).upper()
                lkg = r.get("LOAD (KG)", None)
                dir_ = str(r.get("Unnamed: 1", "")).upper().strip()
                name_ok = vid.startswith(subject)
                spd_ok  = spd == speed
                load_ok = (load is None) or (lkg is not None and abs(float(lkg) - load) < 0.1)
                dir_ok  = dir_ == direction or direction not in ("IDA", "VOLTA")
                return name_ok and spd_ok and load_ok and dir_ok

            matches = df_excel[df_excel.apply(row_match, axis=1)]
            if len(matches):
                row_excel = matches.iloc[0]

    def _field(parent, label, default="", row=0, width=18):
        ttk.Label(parent, text=label, anchor="w", style="TLabel").grid(
            row=row, column=0, sticky="w", padx=8, pady=4)
        var = tk.StringVar(value=str(default))
        ttk.Entry(parent, textvariable=var, width=width).grid(
            row=row, column=1, sticky="w", padx=8, pady=4)
        return var

    frm = ttk.Frame(win, style="TFrame", padding=(16, 12))
    frm.pack(fill="both", expand=True)

    # Auto-preenchimento a partir do nome do arquivo e do Excel
    auto_name = basename
    auto_bm   = ""
    auto_leg  = ""
    auto_dir  = "IDA"   # default
    auto_load = ""
    # Extrai direção e carga do nome do arquivo (ex: FABIO_VAS_14KG_VOLTA)
    parts_fn = basename.split("_")
    if len(parts_fn) >= 4:
        auto_dir  = parts_fn[3]   # IDA ou VOLTA
        auto_load = parts_fn[2]   # ex: 14KG

    if row_excel is not None:
        subj = str(row_excel.get("VIDEO NAME", "")).upper()
        spd  = str(row_excel.get("SPEED",      "")).upper()
        lkg  = row_excel.get("LOAD (KG)", "")
        dir_ = str(row_excel.get("Unnamed: 1","")).upper().strip()
        if dir_ in ("IDA", "VOLTA"):
            auto_dir = dir_
        auto_name = f"{subj}_{spd}_{lkg}kg_{auto_dir}"
        bm_val  = row_excel.get("BM (KG)", "")
        leg_val = row_excel.get("Leg Length Right (GT to ground) (m)", "")
        auto_bm  = "" if pd.isna(bm_val)  else str(bm_val)
        auto_leg = "" if pd.isna(leg_val) else str(leg_val)

    # Coxa estimada = 53 % do comprimento do membro (GT ao solo)
    # Derivação automática: se leg_length disponível mas coxa não informada,
    # usa thigh = 0.53 × leg_length (parâmetro antropométrico padrão)
    try:
        auto_thigh = f"{float(auto_leg) * 0.53:.3f}"
    except Exception:
        auto_thigh = ""

    cap_tmp = cv2.VideoCapture(video_path)
    fps_default = cap_tmp.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_tmp.release()
    total_dur = total_frames / fps_default if fps_default > 0 else 0

    v_name  = _field(frm, "Identificação (paciente/trial):", auto_name,   row=0, width=30)
    v_bm    = _field(frm, "Massa corporal BM (kg):",         auto_bm,     row=1)
    v_leg   = _field(frm, "Comprimento membro inf. GT (m):", auto_leg,    row=2)
    v_thigh = _field(frm, "Comprimento da coxa (m) [GT→joelho]:", auto_thigh, row=3)
    v_fps   = _field(frm, "Frequência de aquisição (Hz):",   f"{fps_default:.2f}", row=4)
    v_tini  = _field(frm, f"Início análise (s) [0–{total_dur:.1f}]:", "0",  row=5)
    v_tfim  = _field(frm, f"Fim análise (s)   [0–{total_dur:.1f}]:", f"{total_dur:.1f}", row=6)

    # Direção da caminhada: NÃO é mais escolhida na tela. O sentido do movimento é
    # auto-detectado pela trajetória do COM em run_analysis (a física vem do dado).
    # Mantemos só o rótulo IDA/VOLTA vindo do nome do arquivo/Excel, para identificação.
    dir_var = tk.StringVar(value=auto_dir)

    # Pose: fixado no melhor backend/modelo. rtmlib = RTMPose-x Halpe-26 (o mais
    # robusto/pesado), com detector dedicado e pés reais; GPU automática (cai para
    # CPU sozinho). Estas escolhas foram removidas da tela de propósito, para o
    # operador não precisar decidir nada sobre pose. (Mantidas como variáveis fixas
    # para não quebrar o restante do pipeline.)
    backend_var         = tk.StringVar(value="rtmlib")
    rtmlib_mode_var     = tk.StringVar(value="performance")
    rtmlib_device_var   = tk.StringVar(value="auto")
    mmpose_model_var    = tk.StringVar(value="wholebody")
    mmpose_device_var   = tk.StringVar(value="cuda:0")
    mmpose_roi_mode_var = tk.StringVar(value="full_frame")
    mmpose_whole_image_var = tk.BooleanVar(value=True)

    ttk.Label(frm, text="Detector de eventos:", anchor="w", style="TLabel").grid(
        row=8, column=0, sticky="w", padx=8, pady=4)
    detector_var = tk.StringVar(value="atual")
    ttk.Combobox(frm, textvariable=detector_var,
                 values=["atual", "robusto"], state="readonly", width=16).grid(
        row=8, column=1, sticky="w", padx=8, pady=4)

    show_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(frm, text="Mostrar vídeo durante processamento",
                    variable=show_var, style="TCheckbutton").grid(
        row=9, column=0, columnspan=2, sticky="w", pady=(6, 2))

    ttk.Separator(frm, orient="horizontal").grid(
        row=10, column=0, columnspan=2, sticky="ew", pady=8)

    ok_flag = [False]
    def _ok():
        ok_flag[0] = True
        win.destroy()

    ttk.Button(frm, text="Continuar  →", style="Accent.TButton", command=_ok).grid(
        row=11, column=0, columnspan=2, pady=(4, 4), sticky="e")

    win.wait_window()
    if not ok_flag[0]:
        return None

    try:    bm    = float(v_bm.get())
    except: bm    = None
    try:    leg   = float(v_leg.get())
    except: leg   = None
    try:    thigh = float(v_thigh.get())
    except: thigh = leg * 0.53 if leg else None
    try:    fs    = float(v_fps.get())
    except: fs    = fps_default
    try:    t_ini = float(v_tini.get())
    except: t_ini = 0.0
    try:    t_fim = float(v_tfim.get())
    except: t_fim = None

    walk_dir = 1 if dir_var.get() == "IDA" else -1

    return {
        "name":      v_name.get().strip() or basename,
        "bm":        bm,
        "leg":       leg,
        "thigh":     thigh,
        "walk_dir":  walk_dir,
        "direction": dir_var.get(),
        "fs":        fs,
        "show":      show_var.get(),
        "t_ini":     t_ini,
        "t_fim":     t_fim,
        "backend":   backend_var.get().strip().lower(),
        "rtmlib_mode": rtmlib_mode_var.get().strip().lower() or "performance",
        "rtmlib_device": rtmlib_device_var.get().strip() or "auto",
        "mmpose_model": mmpose_model_var.get().strip() or "wholebody",
        "mmpose_device": mmpose_device_var.get().strip() or "cuda:0",
        "mmpose_whole_image": bool(mmpose_whole_image_var.get()),
        "mmpose_roi_mode": mmpose_roi_mode_var.get().strip().lower(),
        "detector": detector_var.get().strip().lower(),
    }

# ===================== Relatório de Qualidade =====================
def _max_false_run(mask):
    """Maior sequência de False em uma máscara booleana."""
    max_run = run = 0
    for val in mask:
        if not val:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def save_quality_report(output_path, name, df_raw, evts, backend="mediapipe", detector="atual", roi_info=None):
    """Salva relatório de qualidade dos landmarks/eventos sem alterar as saídas originais."""
    os.makedirs(os.path.join(output_path, "Qualidade"), exist_ok=True)
    rows = []
    n = len(df_raw)
    for nm in LANDMARK_MAP.values():
        vis_col = f"{nm}_vis"
        x_col = f"{nm}_x_px"
        y_col = f"{nm}_y_px"
        if vis_col not in df_raw.columns:
            rows.append({
                "Paciente": name, "Backend": backend, "Detector": detector,
                "Landmark": nm, "Frames": n, "Pct_Valido": 0.0,
                "Frames_Invalidos": n, "Maior_Gap_Frames": n,
                "Alerta": "coluna ausente",
            })
            continue
        valid = (
            (df_raw[vis_col].values >= VIS_THRESHOLD) &
            np.isfinite(df_raw[x_col].values) &
            np.isfinite(df_raw[y_col].values)
        )
        pct = float(valid.mean() * 100.0) if n else 0.0
        alert = ""
        if pct < 50 and any(k in nm for k in ["calcanhar", "ponta", "tornozelo"]):
            alert = "baixa confiabilidade no pé"
        elif pct < 50:
            alert = "baixa confiabilidade"
        rows.append({
            "Paciente": name,
            "Backend": backend,
            "Detector": detector,
            "Landmark": nm,
            "Frames": n,
            "Pct_Valido": pct,
            "Frames_Invalidos": int((~valid).sum()),
            "Maior_Gap_Frames": int(_max_false_run(valid)),
            "Alerta": alert,
        })

    event_summary = pd.DataFrame([{
        "Paciente": name,
        "Backend": backend,
        "Detector": detector,
        "TD_R": len(evts.get("TD_R", [])),
        "TO_R": len(evts.get("TO_R", [])),
        "TD_L": len(evts.get("TD_L", [])),
        "TO_L": len(evts.get("TO_L", [])),
    }])

    path = os.path.join(output_path, "Qualidade", f"{name}_Qualidade_Deteccao.xlsx")
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="Landmarks", index=False)
        event_summary.to_excel(w, sheet_name="Eventos", index=False)
        if roi_info is not None:
            pd.DataFrame([roi_info]).to_excel(w, sheet_name="ROI", index=False)
    return path


# ===================== Geração de Output =====================
def save_outputs(output_path, name, df_filt, df_steps, df_strides,
                 grp_step, grp_stride, df_ang_stride, df_ang_step,
                 mech, df_mech_step, mec_interp_step,
                 ang_interp_step, evts, fs):
    """Salva todos os resultados em subpastas organizadas."""
    subdirs = ["Angulares_Gerais", "Angulares_Interpolados", "Angulares_Interpolados_Passo",
               "Espaco_Temporais", "Mecanica", "Graficos", "Eventos", "Marcadores"]
    for sd in subdirs:
        os.makedirs(os.path.join(output_path, sd), exist_ok=True)

    marker_cols = ["frame"]
    for nm in LANDMARK_MAP.values():
        marker_cols.extend([
            f"{nm}_x_px", f"{nm}_y_px", f"{nm}_vis",
            f"{nm}_x_m", f"{nm}_y_m",
        ])
    df_markers = df_filt[[c for c in marker_cols if c in df_filt.columns]].copy()
    df_markers.insert(0, "Paciente", name)
    with pd.ExcelWriter(os.path.join(output_path, "Marcadores",
                                     f"{name}_Marcadores.xlsx"),
                        engine="xlsxwriter") as w:
        df_markers.to_excel(w, sheet_name="Marcadores", index=False)

    # ── Angulares gerais ──
    ang_cols = ["frame"] + [c for c in df_filt.columns
                             if "_m" in c or "angle" in c]
    df_ang = df_filt[[c for c in ang_cols if c in df_filt.columns]].copy()
    df_ang.insert(0, "Paciente", name)
    with pd.ExcelWriter(os.path.join(output_path, "Angulares_Gerais",
                                     f"{name}_Angulares_Gerais.xlsx"),
                        engine="xlsxwriter") as w:
        df_ang.to_excel(w, sheet_name="Dados_Filtrados", index=False)

    # ── Angulares interpolados 0-100% ──
    TD_R = evts.get("TD_R", np.array([], dtype=int))
    ang_interp = {}
    for i in range(len(TD_R) - 1):
        s, e = TD_R[i], TD_R[i+1]
        if e <= s:
            continue
        cyc = {}
        for side in ("dir", "esq"):
            for ang in ("hip", "knee", "ankle", "foot"):
                col = f"{ang}_angle_{side}_filt"
                if col in df_filt.columns:
                    cyc[col] = normalize_to_100(df_filt.iloc[s:e][col].values)
        if "trunk_angle_filt" in df_filt.columns:
            cyc["trunk_angle_filt"] = normalize_to_100(df_filt.iloc[s:e]["trunk_angle_filt"].values)
        ang_interp[f"Passada_{i+1}"] = pd.DataFrame(cyc, index=np.linspace(0, 100, 101))

    if ang_interp:
        df_interp = pd.concat(ang_interp, axis=1)
        df_interp.index.name = "Ciclo_%"
        df_interp.insert(0, "Paciente", name)
        with pd.ExcelWriter(os.path.join(output_path, "Angulares_Interpolados",
                                         f"{name}_Angulares_Interpolados.xlsx"),
                            engine="xlsxwriter") as w:
            df_interp.to_excel(w, sheet_name="Interpolados_0_100")

    # ── Angulares interpolados por passo 0-100% ──
    if ang_interp_step:
        df_interp_step = pd.concat(ang_interp_step, axis=1)
        df_interp_step.index.name = "Ciclo_%"
        df_interp_step.insert(0, "Paciente", name)
        with pd.ExcelWriter(os.path.join(output_path, "Angulares_Interpolados_Passo",
                                         f"{name}_Angulares_Interpolados_Passo.xlsx"),
                            engine="xlsxwriter") as w:
            df_interp_step.to_excel(w, sheet_name="Interpolados_0_100")

    # ── Espaço-temporais ──
    et_path = os.path.join(output_path, "Espaco_Temporais",
                            f"{name}_Espaco_Temporais.xlsx")
    with pd.ExcelWriter(et_path, engine="xlsxwriter") as w:
        df_steps.to_excel(w, sheet_name="Passos", index=False)
        df_strides.to_excel(w, sheet_name="Passadas", index=False)
        grp_step.to_excel(w, sheet_name="Medias_Passos", index=False)
        grp_stride.to_excel(w, sheet_name="Medias_Passadas", index=False)
        df_ang_stride.to_excel(w, sheet_name="Angular_Por_Passada", index=False)
        if df_ang_step is not None and not df_ang_step.empty:
            df_ang_step.to_excel(w, sheet_name="Angular_Por_Passo", index=False)

    # ── Mecânica ──
    mec_summary = pd.DataFrame([{
        "Paciente":           name,
        "W_ext_J":            mech["W_ext_J"],
        "W_ext_J_per_kg":     mech["W_ext_J_per_kg"],
        "W_ext_J_per_kg_m":   mech["W_ext_J_per_kg_m"],
        "Wv_J":               mech["Wv_J"],
        "Wv_J_per_kg":        mech["Wv_J_per_kg"],
        "Wv_J_per_kg_m":      mech["Wv_J_per_kg_m"],
        "Wf_J":               mech["Wf_J"],
        "Wf_J_per_kg":        mech["Wf_J_per_kg"],
        "Wf_J_per_kg_m":      mech["Wf_J_per_kg_m"],
        "Recovery_pct":       mech["Recovery_pct"],
        "IRL":                mech["IRL"],
        "LRI_Tartaruga":      mech["LRI_Tartaruga"],
        "Mean_Speed_ms":      mech["Mean_Speed_ms"],
        "Total_Disp_m":       mech["Total_Disp_m"],
        "Nota_unidades":      "J/(kg·m) = energia especifica por massa e distancia horizontal",
        "Nota_IRL":           "IRL = Recovery x (v/v_otima); v_otima = sqrt(g x L)",
        "Nota_LRI":           "LRI_Tartaruga = 100 x SSWS/OWS; OWS = 0.5*sqrt(g*LLL); Peyre-Tartaruga & Monteiro 2016",
        "Nota_Wv":            "Wv = trabalho para elevar o COM (incrementos positivos de Ep)",
        "Nota_Wf":            "Wf = trabalho para acelerar o COM (incrementos positivos de Ekf = 0.5*M*vx^2)",
        "Nota_Congruencia":   "% Congruencia por passo na aba Mecanica_Por_Passo; Bishop et al. 2008",
        "Nota_PhaseShift":    "Phase shift alpha/beta por passo; Cavagna & Legramandi 2020",
    }])
    mec_curves = pd.DataFrame({
        "Ep_J":      mech["_Ep"],
        "Ekf_J":     mech["_Ekf"],
        "Ekv_J":     mech["_Ekv"],
        "Ek_J":      mech["_Ek"],
        "Emec_J":    mech["_Emec"],
        "x_com_m":   mech["_x_com"],
        "y_com_m":   mech["_y_com"],
        "vx_com_ms": mech["_vx"],
        "vy_com_ms": mech["_vy"],
    })
    with pd.ExcelWriter(os.path.join(output_path, "Mecanica",
                                     f"{name}_Mecanica.xlsx"),
                        engine="xlsxwriter") as w:
        mec_summary.to_excel(w, sheet_name="Resumo", index=False)
        mec_curves.to_excel(w, sheet_name="Series_Temporais", index=False)
        if df_mech_step is not None and not df_mech_step.empty:
            df_mech_step.to_excel(w, sheet_name="Mecanica_Por_Passo", index=False)
        if mec_interp_step:
            df_mec_interp = pd.concat(mec_interp_step, axis=1)
            df_mec_interp.index.name = "Ciclo_%"
            df_mec_interp.to_excel(w, sheet_name="Series_Interp_Passo")

    # ── Eventos ──
    ev_rows = []
    for side, key_td, key_to in [("Direita","TD_R","TO_R"),("Esquerda","TD_L","TO_L")]:
        TD = evts.get(key_td, [])
        TO = evts.get(key_to, [])
        for j, td in enumerate(TD):
            ev_rows.append({
                "Paciente": name, "Lado": side, "Ordem": j+1,
                "TD_Frame": int(td), "TD_Tempo_s": td/fs,
                "TO_Frame": int(TO[j]) if j < len(TO) else np.nan,
                "TO_Tempo_s": TO[j]/fs if j < len(TO) else np.nan,
            })
    pd.DataFrame(ev_rows).to_excel(
        os.path.join(output_path, "Eventos", f"{name}_Eventos.xlsx"), index=False)

    # ── Gráficos ──
    _plot_signals(output_path, name, df_filt, mech, evts, fs)
    _plot_angles(output_path, name, df_filt, evts, fs)

    # ── Resumo Geral ──
    _save_resumo_geral(output_path, name, mech, grp_step, grp_stride,
                       df_ang_stride, df_mech_step,
                       df_steps=df_steps, df_ang_step=df_ang_step)


def _build_master_step_table(name, df_steps, df_mech_step, df_ang_step):
    """
    Constrói tabela mestre com uma linha por passo:
    espaço-temporais + mecânica + ROM angular, tudo junto.
    """
    if df_steps is None or df_steps.empty:
        return pd.DataFrame()

    # Base: passos com TC_Contra válido, ordenados por TD_Frame
    base = df_steps.dropna(subset=["TC_Contra_Frame"]).copy()
    base["TD_Frame"] = base["TD_Frame"].astype(int)
    base = base.sort_values("TD_Frame").reset_index(drop=True)
    base.insert(0, "Paciente", name)

    # Merge mecânica (chave: TD_Frame = Frame_Inicio)
    if df_mech_step is not None and not df_mech_step.empty:
        drop_cols = [c for c in ("Passo", "Frame_Fim", "Tempo_Passo_s")
                     if c in df_mech_step.columns]
        dm = df_mech_step.drop(columns=drop_cols, errors="ignore")
        dm = dm.rename(columns={"Frame_Inicio": "TD_Frame"})
        dm["TD_Frame"] = dm["TD_Frame"].astype(int)
        base = base.merge(dm, on="TD_Frame", how="left")

    # Merge angular (chave: TD_Frame = Frame_Inicio)
    if df_ang_step is not None and not df_ang_step.empty:
        drop_cols = [c for c in ("Passo", "Tempo_s") if c in df_ang_step.columns]
        da = df_ang_step.drop(columns=drop_cols, errors="ignore")
        if "Frame_Inicio" in da.columns:
            da = da.rename(columns={"Frame_Inicio": "TD_Frame"})
            da["TD_Frame"] = da["TD_Frame"].astype(int)
            base = base.merge(da, on="TD_Frame", how="left")

    return base


def _save_resumo_geral(output_path, name, mech, grp_step, grp_stride,
                       df_ang_stride, df_mech_step=None,
                       df_steps=None, df_ang_step=None):
    """
    Salva/acumula Resumo_Geral.xlsx na raiz da pasta de saída com 3 abas:
      Resumo    — uma linha por trial (acumula entre sessões)
      Medias_SD — média ± DP de todas as variáveis da tabela mestre por passo
      Por_Passo — tabela mestre: uma linha por passo com ET + mecânica + angular
    """
    # ── Linha de resumo global (1 por trial) ──
    row = {"Paciente": name}
    for key in ("W_ext_J_per_kg_m", "Wv_J_per_kg_m", "Wf_J_per_kg_m",
                "Recovery_pct", "IRL", "LRI_Tartaruga", "Mean_Speed_ms"):
        row[key] = mech.get(key, np.nan)

    if df_mech_step is not None and not df_mech_step.empty:
        for col in ("Congruencia_pct", "PhaseShift_alpha_deg",
                    "PhaseShift_beta_deg", "Recovery_pct"):
            if col in df_mech_step.columns:
                row[f"Passo_mean_{col}"] = df_mech_step[col].mean()

    if grp_step is not None and not grp_step.empty:
        for col in ("Tempo_Passo_s", "Comprimento_Passo_m", "Cadencia_Hz",
                    "Apoio_s", "Balanco_s", "Duty_Factor",
                    "Duplo_Apoio_s", "Velocidade_ms",
                    "Quadril_TD_deg", "Joelho_TD_deg", "Tornozelo_TD_deg",
                    "Quadril_TO_deg", "Joelho_TO_deg", "Tornozelo_TO_deg"):
            if col in grp_step.columns:
                row[f"ET_{col}"] = grp_step[col].mean()

    if grp_stride is not None and not grp_stride.empty:
        for col in ("Tempo_Passada_s", "Comprimento_Passada_m", "Cadencia_passada_Hz"):
            if col in grp_stride.columns:
                row[f"ET_{col}"] = grp_stride[col].mean()

    ang_cols = [
        "hip_dir_max",   "hip_dir_min",   "hip_dir_ROM",
        "hip_esq_max",   "hip_esq_min",   "hip_esq_ROM",
        "knee_dir_max",  "knee_dir_min",  "knee_dir_ROM",
        "knee_esq_max",  "knee_esq_min",  "knee_esq_ROM",
        "ankle_dir_max", "ankle_dir_min", "ankle_dir_ROM",
        "ankle_esq_max", "ankle_esq_min", "ankle_esq_ROM",
        "trunk_max",     "trunk_min",     "trunk_ROM",
    ]
    if df_ang_stride is not None and not df_ang_stride.empty:
        for col in ang_cols:
            if col in df_ang_stride.columns:
                row[f"ANG_{col}"] = df_ang_stride[col].mean()

    df_new = pd.DataFrame([row])

    # ── Tabela mestre por passo ──
    df_master = _build_master_step_table(name, df_steps, df_mech_step, df_ang_step)

    # ── Média ± DP da tabela mestre ──
    if not df_master.empty:
        num_cols = df_master.select_dtypes(include=[np.number]).columns.tolist()
        means = df_master[num_cols].mean().rename(lambda c: f"{c}_mean")
        sds   = df_master[num_cols].std().rename(lambda c: f"{c}_SD")
        ns    = df_master[num_cols].count().rename(lambda c: f"{c}_N")
        stat_row = pd.concat([means, sds, ns]).to_frame().T
        stat_row.insert(0, "Paciente", name)
        # Reordena: Var_mean, Var_SD, Var_N intercalados
        ordered_cols = ["Paciente"]
        for c in num_cols:
            ordered_cols += [f"{c}_mean", f"{c}_SD", f"{c}_N"]
        stat_row = stat_row.reindex(columns=ordered_cols)
    else:
        stat_row = pd.DataFrame([{"Paciente": name}])

    # ── Lê arquivo existente e atualiza ──
    resumo_path = os.path.join(output_path, "Resumo_Geral.xlsx")
    sheets = {"Resumo": df_new, "Medias_SD": stat_row, "Por_Passo": df_master}

    if os.path.exists(resumo_path):
        try:
            existing = pd.read_excel(resumo_path, sheet_name=None)
            for sheet_name, df_new_sheet in sheets.items():
                if sheet_name in existing:
                    old = existing[sheet_name]
                    old = old[old["Paciente"] != name]
                    sheets[sheet_name] = pd.concat([old, df_new_sheet],
                                                   ignore_index=True)
        except Exception:
            pass

    with pd.ExcelWriter(resumo_path, engine="xlsxwriter") as w:
        for sheet_name, df_sheet in sheets.items():
            if df_sheet is not None and not df_sheet.empty:
                df_sheet.to_excel(w, sheet_name=sheet_name, index=False)
                ws = w.sheets[sheet_name]
                ws.set_column(0, len(df_sheet.columns) - 1, 20)


def _plot_signals(output_path, name, df, mech, evts, fs):
    """Gráfico: sinais de detecção + energias mecânicas."""
    t = np.arange(len(df)) / fs
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    for ax, side, key_td, key_to, sig_td, sig_to, c_td, c_to in [
        (axes[0], "Direito",  "TD_R","TO_R","td_sig_R","to_sig_R","red",  "orange"),
        (axes[1], "Esquerdo", "TD_L","TO_L","td_sig_L","to_sig_L","blue", "cyan"),
    ]:
        if sig_td in evts:
            ax.plot(t, evts[sig_td], lw=0.8, color="gray", label="CalcHeel-Quadril")
        if sig_to in evts:
            ax.plot(t, evts[sig_to], lw=0.8, color="silver", ls="--", label="Ponta-Quadril")
        for fr in evts.get(key_td, []):
            ax.axvline(fr/fs, color=c_td, lw=1.2, ls="--")
        for fr in evts.get(key_to, []):
            ax.axvline(fr/fs, color=c_to, lw=1.2, ls=":")
        ax.set_ylabel(f"Sinal {side} (m)")
        ax.legend(fontsize=7)

    # Energias
    axes[2].plot(t, mech["_Ep"],   lw=1.2, label="Ep (J)", color="blue")
    axes[2].plot(t, mech["_Ek"],   lw=1.2, label="Ek (J)", color="red")
    axes[2].plot(t, mech["_Emec"], lw=1.5, label="Emec (J)", color="black")
    axes[2].set_ylabel("Energia (J)")
    axes[2].legend(fontsize=7)

    # Velocidade no SENTIDO da marcha (positiva mesmo na VOLTA, em que o COM anda
    # para x negativo). Preserva a oscilação, só orienta o sinal pela direção média.
    v_dir = mech["_vx"] * (np.sign(np.nanmean(mech["_vx"])) or 1.0)
    axes[3].plot(t, v_dir, lw=1, color="green")
    axes[3].set_ylabel("Vel. COM — sentido marcha (m/s)")
    axes[3].set_xlabel("Tempo (s)")
    axes[3].axhline(0, color="gray", lw=0.5)

    fig.suptitle(f"{name} — Sinais e Energia Mecânica\n"
                 f"Recovery={mech['Recovery_pct']:.1f}%  |  "
                 f"Vmedia={mech['Mean_Speed_ms']:.2f} m/s  |  "
                 f"Wext={mech['W_ext_J_per_kg_m']:.2f} J/(kg·m)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "Graficos", f"{name}_Sinais_Mecanica.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def _plot_angles(output_path, name, df, evts, fs):
    """Gráfico: ângulos bilaterais com faixas de contato."""
    t  = np.arange(len(df)) / fs
    TD_R = evts.get("TD_R", []); TO_R = evts.get("TO_R", [])
    TD_L = evts.get("TD_L", []); TO_L = evts.get("TO_L", [])

    ang_pairs = [
        ("hip_angle_dir_filt",   "hip_angle_esq_filt",   "Quadril (°)"),
        ("knee_angle_dir_filt",  "knee_angle_esq_filt",  "Joelho (°)"),
        ("ankle_angle_dir_filt", "ankle_angle_esq_filt", "Tornozelo (°)"),
        ("trunk_angle_filt",     None,                   "Tronco (°)"),
    ]

    fig, axes = plt.subplots(len(ang_pairs), 1, figsize=(14, 12), sharex=True)
    for ax, (col_d, col_e, ylabel) in zip(axes, ang_pairs):
        if col_d and col_d in df.columns:
            ax.plot(t, df[col_d], color="red",  lw=1, label="Direito")
        if col_e and col_e in df.columns:
            ax.plot(t, df[col_e], color="blue", lw=1, label="Esquerdo")
        ymin, ymax = ax.get_ylim()
        for td, to in zip(TD_R, TO_R):
            ax.axvspan(td/fs, to/fs, color="red",  alpha=0.08)
        for td, to in zip(TD_L, TO_L):
            ax.axvspan(td/fs, to/fs, color="blue", alpha=0.08)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("Tempo (s)")
    fig.suptitle(f"{name} — Ângulos Bilaterais (faixas = fase de apoio)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "Graficos", f"{name}_Angulos_Bilaterais.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

# ===================== Pipeline Principal =====================
def run_analysis(root, video_path, output_path, df_excel):
    """Executa o pipeline completo para um vídeo."""
    params = ask_parameters(root, video_path, df_excel)
    if params is None:
        return False

    name     = params["name"]
    bm       = params["bm"]
    leg      = params.get("leg")
    thigh_m  = params["thigh"]
    fs_req   = params["fs"]
    show     = params["show"]
    walk_dir = params["walk_dir"]
    t_ini    = params.get("t_ini", 0.0)
    t_fim    = params.get("t_fim", None)
    backend  = params.get("backend", "mediapipe")
    detector = params.get("detector", "atual")
    mmpose_model = params.get("mmpose_model", "wholebody")
    mmpose_device = params.get("mmpose_device", "cuda:0")
    mmpose_whole_image = params.get("mmpose_whole_image", True)
    mmpose_roi_mode = params.get("mmpose_roi_mode", "auto_track")
    rtmlib_mode = params.get("rtmlib_mode", "balanced")
    rtmlib_device = params.get("rtmlib_device", "cuda")

    # Salva cada backend em subpasta separada para permitir comparação sem sobrescrever.
    output_root = output_path
    if backend == "mmpose":
        output_path = os.path.join(output_root, "Resultados_MMPose")
    elif backend == "rtmlib":
        output_path = os.path.join(output_root, "Resultados_RTMLib")
    else:
        output_path = os.path.join(output_root, "Resultados_MediaPipe")
    os.makedirs(output_path, exist_ok=True)

    if bm is None:
        messagebox.showerror("Erro", "Massa corporal (BM) é obrigatória."); return False
    
    # Garante thigh: usa valor fornecido ou calcula como leg × 0.53
    if thigh_m is None and leg is not None:
        thigh_m = leg * 0.53
    
    if thigh_m is None:
        messagebox.showerror("Erro", "Comprimento da coxa é obrigatório (ou forneça o comprimento total da perna)."); return False

    # ── 1. Captura primeiro frame para calibração (sempre frame 0) ──
    cap = cv2.VideoCapture(video_path)
    fps_vid = cap.get(cv2.CAP_PROP_FPS)
    # NÃO fazer seek aqui — primeiro frame é sempre frame 0 para calibração
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Erro", "Não foi possível ler o primeiro frame."); return False

    fs = fs_req if fs_req and fs_req > 0 else fps_vid

    # ── 2. Calibração por cones (base + topo, qtd. livre) ──
    cone_px, cone_real, cone_scale_y = select_cones_opencv(first_frame)
    if cone_px is None:
        messagebox.showwarning("Calibração", "Calibração por cones ignorada — escala uniforme será usada.")

    # ── 2b. ROI para MMPose: automática visual, manual ou frame inteiro ──
    mmpose_roi = None
    roi_info = {"roi_mode": mmpose_roi_mode, "status": "nao_aplicavel"}
    if backend == "mmpose":
        if mmpose_roi_mode in ("auto_track", "auto_person", "auto_static"):
            messagebox.showinfo(
                "MMPose Auto-Person",
                "O script tentara escolher automaticamente a pessoa principal/mais proxima.\n\n"
                "Na janela seguinte, o restante do video ficara fosco e a pessoa escolhida ficara em destaque.\n"
                "ENTER/ESPAÇO aceita | N/P troca candidato | M usa ROI manual | Q cancela."
            )
            mmpose_roi, roi_info = select_auto_person_roi_mmpose(
                video_path,
                start_sec=t_ini if t_ini and t_ini > 0 else None,
                end_sec=t_fim,
                model_alias=mmpose_model,
                device=mmpose_device,
            )
            if mmpose_roi is None:
                messagebox.showwarning(
                    "MMPose Auto-Person",
                    "Seleção automática falhou/cancelou. Vamos cair para ROI manual."
                )
                roi_frame = get_frame_at_time(video_path, t_ini if t_ini is not None else 0.0)
                if roi_frame is None:
                    roi_frame = first_frame
                mmpose_roi = select_subject_roi_opencv(roi_frame)
                roi_info = {"roi_mode": "manual_roi", "status": "fallback_pos_auto_person"}
        elif mmpose_roi_mode == "manual_roi":
            roi_frame = get_frame_at_time(video_path, t_ini if t_ini is not None else 0.0)
            if roi_frame is None:
                roi_frame = first_frame
            messagebox.showinfo(
                "ROI MMPose",
                "Selecione uma ROI que contenha toda a trajetória da pessoa, "
                "mas exclua o máximo possível do fundo."
            )
            mmpose_roi = select_subject_roi_opencv(roi_frame)
            roi_info = {"roi_mode": "manual_roi", "status": "aceito" if mmpose_roi is not None else "sem_roi"}
        else:
            roi_info = {"roi_mode": "full_frame", "status": "frame_inteiro"}

        if mmpose_roi is None and mmpose_roi_mode != "full_frame":
            messagebox.showwarning("ROI MMPose", "ROI não selecionada. O MMPose usará o frame inteiro.")
        if mmpose_roi is not None:
            roi_info.update({"roi_x": mmpose_roi[0], "roi_y": mmpose_roi[1],
                             "roi_w": mmpose_roi[2], "roi_h": mmpose_roi[3]})
        roi_info["dynamic_roi"] = bool(mmpose_roi_mode in ("auto_track", "auto_person"))

    # ── 3. Processamento do backend de pose ──
    import time as _time
    prog_win = tk.Toplevel(root)
    prog_win.title("Processando")
    prog_win.resizable(False, False)
    prog_win.configure(bg=UI["bg"])
    apply_clinical_theme(prog_win)
    set_window_icon(prog_win)
    prog_win.grab_set()
    make_header(prog_win, "Processando vídeo", os.path.basename(video_path))

    body = ttk.Frame(prog_win, style="TFrame", padding=(18, 14))
    body.pack(fill="both", expand=True)
    stage_var = tk.StringVar(value="Preparando…")
    pct_var   = tk.StringVar(value="")
    eta_var   = tk.StringVar(value="")
    ttk.Label(body, textvariable=stage_var, style="TLabel", font=FONT_BOLD).pack(anchor="w")
    pb = ttk.Progressbar(body, mode="indeterminate", maximum=100, length=380)
    pb.pack(fill="x", pady=(8, 4))
    pb.start(12)
    info = ttk.Frame(body, style="TFrame"); info.pack(fill="x")
    ttk.Label(info, textvariable=pct_var, style="Muted.TLabel").pack(side="left")
    ttk.Label(info, textvariable=eta_var, style="Muted.TLabel").pack(side="right")
    prog_win.update()

    _t0 = [None]
    def _progress(stage, done, total):
        if _t0[0] is None:
            _t0[0] = _time.time()
        stage_var.set(stage)
        if total and total > 0:
            if str(pb["mode"]) != "determinate":
                pb.stop(); pb.config(mode="determinate")
            pct = min(100.0, 100.0 * done / max(total, 1))
            pb["value"] = pct
            pct_var.set(f"{done}/{total} frames  ·  {pct:.0f}%")
            el = _time.time() - _t0[0]
            if 0 < done < total:
                eta = el * (total - done) / done
                eta_var.set(f"~{int(eta)}s restantes")
            else:
                eta_var.set("")
        try:
            prog_win.update()
        except Exception:
            pass

    try:
        df_raw, frames, fps_vid2, vid_w, vid_h = process_video(
            video_path,
            start_sec=t_ini if t_ini and t_ini > 0 else None,
            end_sec=t_fim,
            show_window=show,
            backend=backend,
            mmpose_model=mmpose_model,
            mmpose_device=mmpose_device,
            mmpose_whole_image=mmpose_whole_image,
            mmpose_roi=mmpose_roi,
            mmpose_dynamic_roi=bool(mmpose_roi_mode in ("auto_track", "auto_person")),
            rtmlib_mode=rtmlib_mode,
            rtmlib_device=rtmlib_device,
            progress_cb=_progress)
    except Exception as e:
        prog_win.destroy()
        messagebox.showerror("Erro", f"Falha no backend {backend}:\n{e}")
        return False

    prog_win.destroy()

    if df_raw.empty:
        messagebox.showerror("Erro", f"{backend.upper()} não detectou nenhum landmark."); return False

    # ── 4. Escala pixel → metro ──
    # Calcula comprimento médio da coxa em pixels
    thigh_px = np.sqrt(
        (df_raw["quadril_dir_x_px"] - df_raw["joelho_dir_x_px"])**2 +
        (df_raw["quadril_dir_y_px"] - df_raw["joelho_dir_y_px"])**2
    ).mean()
    uy = thigh_m / thigh_px
    print(f"[INFO] Coxa média: {thigh_px:.1f} px → escala Y (fallback) = {uy:.5f} m/px")
    if cone_scale_y is not None:
        print(f"[INFO] Escala Y por cone (m/px): "
              f"{', '.join(f'{v:.5f}' for v in cone_scale_y)}")

    scale_x_func, scale_y_func = build_scale_functions(cone_px, cone_real, cone_scale_y, thigh_px, thigh_m)

    # ── 5. Pipeline de processamento ──
    df = fix_feet(df_raw)
    df = apply_scale(df, scale_x_func, scale_y_func)
    df = filter_coords(df, fs)
    df = compute_angles(df)
    df = filter_angles(df, fs)

    # ── 6. COM e trabalho mecânico ──
    x_com, y_com = estimate_com(df)
    df["x_com_m"] = x_com
    df["y_com_m"] = y_com
    mech = compute_mechanical_work(x_com, y_com, bm, fs, leg_length=leg)
    irl_str = f"{mech['IRL']:.3f}" if not np.isnan(mech["IRL"]) else "N/A (sem L_membro)"
    print(f"[INFO] Recovery={mech['Recovery_pct']:.1f}%  |  "
          f"IRL={irl_str}  |  "
          f"Wext={mech['W_ext_J_per_kg_m']:.2f} J/(kg·m)  |  "
          f"V={mech['Mean_Speed_ms']:.2f} m/s")

    # ── 7. Detecção de eventos — direção AUTO-DETECTADA pela trajetória do COM ──
    # A física vem do dado: o sentido do movimento é o sinal do deslocamento líquido do
    # COM, não uma escolha do operador. O 'walk_dir' vindo do nome do arquivo serve só
    # para um aviso de sanidade (arquivo possivelmente mal-nomeado).
    walk_dir_auto = _walk_direction(x_com)
    print(f"[INFO] Direção (auto pelo COM): {'IDA (esq->dir)' if walk_dir_auto > 0 else 'VOLTA (dir->esq)'}")
    if walk_dir is not None and walk_dir_auto != walk_dir:
        print(f"[AVISO] Direção do nome do arquivo ({'IDA' if walk_dir > 0 else 'VOLTA'}) "
              f"difere da detectada no vídeo — usando a detectada.")
    walk_dir = walk_dir_auto
    if detector == "robusto":
        evts = detect_events_robust(df, fs, x_com, walk_dir=walk_dir)
    else:
        evts = detect_events(df, fs, x_com, walk_dir=walk_dir)
    # ── 8. Revisão interativa ──
    rev = EventReviewWindow(root, df, evts, fs)
    if not rev.accepted:
        return False
    evts_ok = rev.get_events()
    # Reconstrói sinals que ficaram no dict original
    for k in ("td_sig_R","to_sig_R","td_sig_L","to_sig_L"):
        evts_ok[k] = evts[k]

    TD_R = evts_ok.get("TD_R", np.array([], dtype=int))
    TO_R = evts_ok.get("TO_R", np.array([], dtype=int))
    TD_L = evts_ok.get("TD_L", np.array([], dtype=int))
    TO_L = evts_ok.get("TO_L", np.array([], dtype=int))

    # ── 9. Espaço-temporais ──
    df_steps, df_strides, grp_step, grp_stride = build_spatiotemporal(
        df, {"TD_R": TD_R, "TO_R": TO_R, "TD_L": TD_L, "TO_L": TO_L},
        fs, x_com, bm, params.get("leg"))

    # ── 10. Angular por passada ──
    df_ang_stride = angular_per_stride(df, TD_R, fs)

    # ── 11. Eventos por passo (TD → TD contralateral) ──
    step_events = sorted(
        [
            (int(row["TD_Frame"]), int(row["TC_Contra_Frame"]))
            for _, row in df_steps.iterrows()
            if not pd.isna(row.get("TC_Contra_Frame")) and not pd.isna(row.get("TD_Frame"))
        ],
        key=lambda x: x[0],
    )

    # ── 12. Mecânica por passo ──
    df_mech_step = compute_step_mechanics(
        mech["_Ep"], mech["_Ekf"], mech["_Ekv"],
        mech["_x_com"], bm, fs, step_events,
    )
    mec_interp_step = interpolate_mechanics_per_step(
        mech["_Ep"], mech["_Ekf"], mech["_Ekv"], mech["_Emec"],
        step_events,
    )

    # ── 13. Angular por passo ──
    df_ang_step     = angular_per_step(df, step_events, fs)
    ang_interp_step = interpolate_angles_per_step(df, step_events)

    # ── 14. Salvar ──
    save_outputs(output_path, name, df,
                 df_steps, df_strides, grp_step, grp_stride,
                 df_ang_stride, df_ang_step,
                 mech, df_mech_step, mec_interp_step,
                 ang_interp_step, evts_ok, fs)
    save_quality_report(output_path, name, df_raw, evts_ok, backend=backend, detector=detector, roi_info=roi_info)

    # Exibe resumo
    speed_str = f"{mech['Mean_Speed_ms']:.2f} m/s"
    rec_str   = f"{mech['Recovery_pct']:.1f}%"
    wext_str  = f"{mech['W_ext_J_per_kg_m']:.2f} J/(kg·m)"
    wv_str    = f"{mech['Wv_J_per_kg_m']:.2f} J/(kg·m)"
    wf_str    = f"{mech['Wf_J_per_kg_m']:.2f} J/(kg·m)"
    irl_str = f"{mech['IRL']:.3f}" if not np.isnan(mech["IRL"]) else "N/A"
    lri_str = f"{mech['LRI_Tartaruga']:.1f}%" if not np.isnan(mech["LRI_Tartaruga"]) else "N/A"
    n_pass  = len(TD_R) - 1
    n_steps = len(step_events)

    messagebox.showinfo("Concluído ✓",
        f"Análise concluída!\n\n"
        f"Passadas (Dir): {n_pass}  |  Passos: {n_steps}\n"
        f"Velocidade média: {speed_str}\n"
        f"Recuperação pendular: {rec_str}\n"
        f"IRL: {irl_str}  |  LRI Tartaruga: {lri_str}\n"
        f"Wext: {wext_str}\n"
        f"Wv: {wv_str}  |  Wf: {wf_str}\n\n"
        f"Resultados salvos em:\n{output_path}")
    return True

# ===================== MAIN =====================
def main():
    root = tk.Tk()
    root.withdraw()
    root.title(APP_TITLE)
    apply_clinical_theme(root)      # tema clínico em todas as janelas ttk
    set_window_icon(root)           # ícone na barra de título / tarefas

    # 1ª execução (app empacotado): prepara a GPU — se houver placa NVIDIA, oferece
    # baixar o runtime CUDA do PyPI. Sem GPU ou se recusado, segue em CPU. Em dev,
    # o gpu_bootstrap pode não estar presente/necessário: ignora silenciosamente.
    try:
        from gpu_bootstrap import ensure_gpu_ready
        ensure_gpu_ready(ask=True)
        _enable_cuda_dlls()   # re-registra caso o runtime tenha acabado de ser baixado
    except Exception:
        pass

    messagebox.showinfo(
        "Análise de Marcha em Grama",
        "Bem-vindo!\n\n"
        "Este script analisa a caminhada em superfície de grama com:\n"
        "  • Correção de perspectiva por cones\n"
        "  • Detecção bilateral de TD/TO com revisão\n"
        "  • Ângulos bilaterais (quadril, joelho, tornozelo, pé, tronco)\n"
        "  • Trabalho mecânico externo e Recovery (Cavagna)\n"
        "  • Parâmetros espaço-temporais completos\n\n"
        "Desenvolvido por Edilson Borba | borba.edi@gmail.com")

    # Carrega Excel de dados dos sujeitos
    excel_path = filedialog.askopenfilename(
        title="Selecione o Excel com dados dos sujeitos (opcional — pode cancelar)",
        filetypes=[("Excel", "*.xlsx *.xls"), ("Todos", "*.*")])
    df_excel = None
    if excel_path:
        try:
            df_excel = pd.read_excel(excel_path)
            # forward-fill dados do sujeito
            for col in ["BM (KG)", "HEIGHT (M)",
                        "Leg Length Right (GT to ground) (m)"]:
                if col in df_excel.columns:
                    df_excel[col] = df_excel[col].ffill()
            print(f"[INFO] Excel carregado: {len(df_excel)} linhas")
        except Exception as e:
            messagebox.showwarning("Aviso", f"Não foi possível ler o Excel:\n{e}")

    output_path = filedialog.askdirectory(title="Selecione a pasta de saída")
    if not output_path:
        messagebox.showerror("Erro", "Pasta de saída não selecionada."); return

    while True:
        video_path = filedialog.askopenfilename(
            title="Selecione o vídeo para analisar",
            filetypes=[("Vídeo", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")])
        if not video_path:
            break

        run_analysis(root, video_path, output_path, df_excel)

        if not messagebox.askyesno("Próximo vídeo?",
                                   "Deseja processar outro vídeo?"):
            break

    messagebox.showinfo("Encerrado", "Sessão encerrada.\nObrigado por usar o programa!\nEdilson Borba | borba.edi@gmail.com")
    root.destroy()


def _selftest(report_path):
    """Modo de verificação do app empacotado: sem GUI, escreve um relatório e sai.
    Uso: definir AM_SELFTEST=<caminho_do_relatorio> e rodar o exe."""
    import sys, traceback
    lines = []
    try:
        import gpu_bootstrap as gb
        lines.append(f"gpu_nvidia={gb.has_nvidia_gpu()}")
        import onnxruntime as ort
        lines.append(f"onnxruntime={ort.__version__}")
        lines.append("providers=" + ",".join(ort.get_available_providers()))
        from rtmlib import BodyWithFeet  # noqa: F401
        lines.append("rtmlib_import=OK")
        lines.append("SELFTEST_OK")
    except Exception:
        lines.append("SELFTEST_FAIL")
        lines.append(traceback.format_exc())
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    _report = os.environ.get("AM_SELFTEST")
    if _report:
        _selftest(_report)
    else:
        main()
