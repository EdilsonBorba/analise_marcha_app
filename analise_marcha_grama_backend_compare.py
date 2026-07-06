# analise_marcha_grama.py
# Análise de marcha em superfície de grama com correção de perspectiva por cones
# Autor: Edilson Borba | borba.edi@gmail.com
# Parâmetros: mecânicos (trabalho externo, recovery, IRL), espaço-temporais bilaterais, angulares bilaterais

# ===================== Imports & Setup =====================
import logging, warnings, os
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import cv2
import mediapipe as mp
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


def process_video_mediapipe(video_path, start_sec=None, end_sec=None, show_window=False):
    """
    Extrai landmarks MediaPipe do vídeo (segmento start_sec–end_sec).
    Retorna DataFrame com coords em pixels e visibilidade, lista de frames, fps, w, h.
    """
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


def _best_mmpose_instance(predictions):
    """Escolhe a instância com maior confiança média/área. Robusto a variações do formato do MMPoseInferencer."""
    if predictions is None:
        return None
    # Formato comum: result['predictions'] = [[{inst1}, {inst2}, ...]]
    if isinstance(predictions, list) and len(predictions) == 1 and isinstance(predictions[0], list):
        instances = predictions[0]
    elif isinstance(predictions, list):
        instances = predictions
    else:
        return None
    if not instances:
        return None

    def score_instance(inst):
        kps = np.asarray(inst.get("keypoints", []), dtype=float)
        sc  = np.asarray(inst.get("keypoint_scores", []), dtype=float)
        mean_sc = float(np.nanmean(sc)) if sc.size else 0.0
        if kps.ndim == 2 and kps.shape[0] > 0:
            x_span = np.nanmax(kps[:, 0]) - np.nanmin(kps[:, 0])
            y_span = np.nanmax(kps[:, 1]) - np.nanmin(kps[:, 1])
            area = max(float(x_span * y_span), 1.0)
        else:
            area = 1.0
        return mean_sc * np.log1p(area)

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
                         model_alias="wholebody", device="cuda:0", use_whole_image=True):
    """
    Extrai landmarks com MMPose/RTMPose/RTMW e retorna o mesmo formato do MediaPipe.
    Recomenda-se model_alias='wholebody' para obter calcanhar e ponta do pé.
    """
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    f_start = int(start_sec * fps) if start_sec is not None else 0
    f_end   = int(end_sec   * fps) if end_sec   is not None else total
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    print(f"[INFO] Inicializando MMPose: pose2d={model_alias}, device={device}, det_model={'whole_image' if use_whole_image else 'default'}")
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

    while cap.isOpened() and fidx <= f_end:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # O inferencer aceita array de imagem. return_vis=False evita carregar imagem anotada.
            result = next(inferencer(frame, return_vis=False, show=False))
            inst = _best_mmpose_instance(result.get("predictions"))
        except Exception as e:
            print(f"[AVISO] MMPose falhou no frame {fidx}: {e}")
            inst = None

        if inst is not None:
            row = _mmpose_instance_to_row(inst, fidx)
            data.append(row)
            frames.append(frame.copy())

            if show_window:
                disp = frame.copy()
                for nm in LANDMARK_MAP.values():
                    x = row.get(f"{nm}_x_px", np.nan)
                    y = row.get(f"{nm}_y_px", np.nan)
                    v = row.get(f"{nm}_vis", 0.0)
                    if np.isfinite(x) and np.isfinite(y):
                        color = (0, 255, 0) if v >= VIS_THRESHOLD else (0, 165, 255)
                        cv2.circle(disp, (int(x), int(y)), 5, color, -1)
                cv2.imshow("Processando MMPose...", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        fidx += 1

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    return pd.DataFrame(data).reset_index(drop=True), frames, fps, w, h


def process_video(video_path, start_sec=None, end_sec=None, show_window=False,
                  backend="mediapipe", mmpose_model="wholebody", mmpose_device="cuda:0",
                  mmpose_whole_image=True):
    """Wrapper único para preservar o restante do pipeline."""
    backend = str(backend).lower().strip()
    if backend == "mediapipe":
        return process_video_mediapipe(video_path, start_sec, end_sec, show_window)
    if backend == "mmpose":
        return process_video_mmpose(video_path, start_sec, end_sec, show_window,
                                   model_alias=mmpose_model,
                                   device=mmpose_device,
                                   use_whole_image=mmpose_whole_image)
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
    win.grab_set()

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
        tk.Label(parent, text=label, anchor="w").grid(row=row, column=0,
                                                        sticky="w", padx=8, pady=3)
        var = tk.StringVar(value=str(default))
        tk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1,
                                                              padx=8, pady=3)
        return var

    frm = tk.Frame(win, padx=15, pady=10)
    frm.pack()

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

    # Direção: IDA (+1) ou VOLTA (−1)
    tk.Label(frm, text="Direção da caminhada:", anchor="w").grid(
        row=7, column=0, sticky="w", padx=8, pady=3)
    dir_var = tk.StringVar(value=auto_dir)
    dir_frame = tk.Frame(frm)
    dir_frame.grid(row=7, column=1, sticky="w")
    tk.Radiobutton(dir_frame, text="IDA  (esq→dir)", variable=dir_var,
                   value="IDA").pack(side=tk.LEFT)
    tk.Radiobutton(dir_frame, text="VOLTA (dir→esq)", variable=dir_var,
                   value="VOLTA").pack(side=tk.LEFT, padx=6)

    # Backend de pose estimation
    tk.Label(frm, text="Backend de pose:", anchor="w").grid(
        row=8, column=0, sticky="w", padx=8, pady=3)
    backend_var = tk.StringVar(value="mediapipe")
    ttk.Combobox(frm, textvariable=backend_var,
                 values=["mediapipe", "mmpose"], state="readonly", width=16).grid(
        row=8, column=1, sticky="w", padx=8, pady=3)

    tk.Label(frm, text="MMPose model alias:", anchor="w").grid(
        row=9, column=0, sticky="w", padx=8, pady=3)
    mmpose_model_var = tk.StringVar(value="wholebody")
    tk.Entry(frm, textvariable=mmpose_model_var, width=18).grid(
        row=9, column=1, sticky="w", padx=8, pady=3)

    tk.Label(frm, text="MMPose device:", anchor="w").grid(
        row=10, column=0, sticky="w", padx=8, pady=3)
    mmpose_device_var = tk.StringVar(value="cuda:0")
    tk.Entry(frm, textvariable=mmpose_device_var, width=18).grid(
        row=10, column=1, sticky="w", padx=8, pady=3)

    tk.Label(frm, text="Detector de eventos:", anchor="w").grid(
        row=11, column=0, sticky="w", padx=8, pady=3)
    detector_var = tk.StringVar(value="atual")
    ttk.Combobox(frm, textvariable=detector_var,
                 values=["atual", "robusto"], state="readonly", width=16).grid(
        row=11, column=1, sticky="w", padx=8, pady=3)

    mmpose_whole_image_var = tk.BooleanVar(value=True)
    tk.Checkbutton(frm, text="MMPose: usar imagem inteira como pessoa única",
                   variable=mmpose_whole_image_var).grid(row=12, column=0, columnspan=2, pady=2)

    show_var = tk.BooleanVar(value=False)
    tk.Checkbutton(frm, text="Mostrar vídeo durante processamento",
                   variable=show_var).grid(row=13, column=0, columnspan=2, pady=4)

    ok_flag = [False]
    def _ok():
        ok_flag[0] = True
        win.destroy()

    tk.Button(frm, text="Continuar →", bg="#2196F3", fg="white",
              font=("Arial", 11, "bold"), command=_ok, padx=10, pady=4
              ).grid(row=14, column=0, columnspan=2, pady=8)

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
        "mmpose_model": mmpose_model_var.get().strip() or "wholebody",
        "mmpose_device": mmpose_device_var.get().strip() or "cuda:0",
        "mmpose_whole_image": bool(mmpose_whole_image_var.get()),
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


def save_quality_report(output_path, name, df_raw, evts, backend="mediapipe", detector="atual"):
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

    axes[3].plot(t, mech["_vx"], lw=1, color="green")
    axes[3].set_ylabel("Vel. COM (m/s)")
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

    # Salva cada backend em subpasta separada para permitir comparação sem sobrescrever.
    output_root = output_path
    if backend == "mmpose":
        output_path = os.path.join(output_root, "Resultados_MMPose")
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

    # ── 3. Processamento do backend de pose ──
    prog_win = tk.Toplevel(root)
    prog_win.title("Processando…")
    prog_win.geometry("360x90")
    prog_win.grab_set()
    ttk.Label(prog_win, text=f"Rodando {backend.upper()}, aguarde…").pack(pady=10)
    pb = ttk.Progressbar(prog_win, mode="indeterminate")
    pb.pack(fill=tk.X, padx=20)
    pb.start()
    root.update()

    try:
        df_raw, frames, fps_vid2, vid_w, vid_h = process_video(
            video_path,
            start_sec=t_ini if t_ini and t_ini > 0 else None,
            end_sec=t_fim,
            show_window=show,
            backend=backend,
            mmpose_model=mmpose_model,
            mmpose_device=mmpose_device,
            mmpose_whole_image=mmpose_whole_image)
    except Exception as e:
        pb.stop(); prog_win.destroy()
        messagebox.showerror("Erro", f"Falha no backend {backend}:\n{e}")
        return False

    pb.stop()
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

    # ── 7. Detecção de eventos (usando walk_dir do Excel) ──
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
    save_quality_report(output_path, name, df_raw, evts_ok, backend=backend, detector=detector)

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


if __name__ == "__main__":
    main()
