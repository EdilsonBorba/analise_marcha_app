"""
app_marcha.py — Interface (shell) estilo dashboard para a Análise de Marcha.

Inspirado no app MEMH: barra lateral + telas (Home, Participantes, Rodar análise,
Visualizador). NÃO reimplementa a análise — usa o analise_marcha.py como MOTOR
(importado como biblioteca), então o script original continua intacto e utilizável.

Requisitos extras: ttkbootstrap, Pillow (já em requirements-app se empacotar este shell).
"""
import os
import sys
import json
import glob
import datetime


def _ensure_std_streams():
    """No app empacotado com PyInstaller em modo janela (console=False), sys.stdout
    e sys.stderr ficam None. Como o motor (analise_marcha) usa print() no pipeline,
    o primeiro print estoura com "'NoneType' object has no attribute 'write'".
    Aqui redirecionamos streams None para um app.log gravável (fallback: devnull),
    ANTES de importar o motor. Também serve de log para depurar na máquina do usuário."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    if getattr(sys, "frozen", False):
        base = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "AnaliseMarcha")
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    stream = None
    try:
        os.makedirs(base, exist_ok=True)
        stream = open(os.path.join(base, "app.log"), "a", buffering=1, encoding="utf-8", errors="replace")
    except Exception:
        try:
            stream = open(os.devnull, "w")
        except Exception:
            return
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


_ensure_std_streams()

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *  # noqa

try:
    from PIL import Image, ImageTk
except Exception:
    Image = ImageTk = None

import pandas as pd

# Motor de análise (script original, não modificado)
import analise_marcha as engine

THEME = "cosmo"   # tema claro/clínico (azul/branco) do ttkbootstrap
# Sidebar clara (os ícones são line-art escuro; fundo claro os deixa legíveis).
SIDEBAR_BG     = "#eef2f6"
SIDEBAR_HOVER  = "#dbe8f7"
SIDEBAR_BORDER = "#d3dce6"
SIDEBAR_CAP    = "#54636f"


def base_dir():
    """Pasta de RECURSOS embutidos (ícones). No app empacotado é o _MEIPASS."""
    return getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))


def data_dir():
    """Pasta GRAVÁVEL e persistente para dados do usuário (participantes, config,
    registro de análises). No app empacotado, base_dir() aponta para a pasta
    temporária do PyInstaller (some ao fechar), então os dados vão para
    %LOCALAPPDATA%\\AnaliseMarcha. Em dev, fica na pasta do projeto."""
    if getattr(sys, "frozen", False):
        d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "AnaliseMarcha")
    else:
        d = os.path.dirname(os.path.abspath(__file__))
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


SUBJECTS_DIR = os.path.join(data_dir(), "Sujeitos")
ICONS_DIR    = os.path.join(base_dir(), "icons")
CONFIG_PATH  = os.path.join(data_dir(), "app_marcha_config.json")
RESULTS_DIRNAME = "Resultados_RTMLib"


# ============================ Persistência ============================
def _slug(name):
    keep = "-_ áéíóúâêôãõàçÁÉÍÓÚÂÊÔÃÕÀÇ"
    return "".join(c for c in str(name).strip() if c.isalnum() or c in keep).strip() or "sujeito"


def list_subjects():
    os.makedirs(SUBJECTS_DIR, exist_ok=True)
    out = []
    for p in sorted(glob.glob(os.path.join(SUBJECTS_DIR, "*.json"))):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            d["_file"] = p
            out.append(d)
        except Exception:
            pass
    return out


def save_subject(data):
    os.makedirs(SUBJECTS_DIR, exist_ok=True)
    path = data.get("_file") or os.path.join(SUBJECTS_DIR, _slug(data.get("nome", "")) + ".json")
    payload = {k: v for k, v in data.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    return path


def delete_subject(data):
    p = data.get("_file")
    if p and os.path.exists(p):
        os.remove(p)


def _find_col(cols, *keywords):
    """Primeira coluna cujo nome (minúsculo) contém TODAS as palavras-chave."""
    for c in cols:
        cl = str(c).lower()
        if all(k in cl for k in keywords):
            return c
    return None


def _num(v):
    try:
        x = float(str(v).replace(",", "."))
        return x if x == x else None   # nan → None
    except Exception:
        return None


def _to_cm(v):
    """Converte para cm: se o valor parece estar em metros (< 3.5), multiplica por 100."""
    x = _num(v)
    if x is None:
        return ""
    return str(round(x * 100, 1)) if x < 3.5 else str(round(x, 1))


def import_subjects_from_excel(path):
    """Importa cadastros de um Excel de caracterização. Mapeia (fuzzy) as colunas:
    Subject→nome, Body Mass→massa_kg, Height→estatura_cm, Leg Length Right→perna_cm,
    thigh-length→coxa_cm. Faz MERGE com cadastros existentes (por nome). Retorna
    (importados, ignorados, aviso)."""
    xl = pd.ExcelFile(path)
    best = None
    for sh in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sh)
        cols = list(df.columns)
        name_c = next((c for c in cols if "subject" in str(c).lower()
                       and "number" not in str(c).lower()), None)
        mass_c = _find_col(cols, "mass") or _find_col(cols, "massa")
        if name_c and mass_c is not None:
            best = (df, cols, name_c, mass_c); break
    if best is None:
        return 0, 0, "Não encontrei uma aba com colunas de Nome (Subject) + Massa."

    df, cols, name_c, mass_c = best
    height_c = _find_col(cols, "height") or _find_col(cols, "estatura")
    legR_c   = _find_col(cols, "leg", "right") or _find_col(cols, "perna", "direita")
    thigh_c  = _find_col(cols, "thigh") or _find_col(cols, "coxa")

    existing = {s.get("nome", "").strip().lower(): s for s in list_subjects()}
    n_imp, n_skip = 0, 0
    for _, r in df.iterrows():
        nome = str(r.get(name_c, "")).strip()
        if not nome or nome.lower() in ("nan", "none"):
            n_skip += 1
            continue
        data = existing.get(nome.lower(), {}).copy()   # merge com o que já existe
        data["nome"] = nome
        m = _num(r.get(mass_c))
        if m is not None:
            data["massa_kg"] = str(round(m, 1))
        if height_c is not None:
            v = _to_cm(r.get(height_c));  data["estatura_cm"] = v or data.get("estatura_cm", "")
        if legR_c is not None:
            v = _to_cm(r.get(legR_c));    data["perna_cm"] = v or data.get("perna_cm", "")
        if thigh_c is not None:
            v = _to_cm(r.get(thigh_c));   data["coxa_cm"] = v or data.get("coxa_cm", "")
        data.setdefault("criado_em", datetime.date.today().strftime("%d/%m/%Y"))
        save_subject(data)
        n_imp += 1
    return n_imp, n_skip, ""


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"out_root": "", "avaliador": "", "avaliador_email": ""}


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception:
        pass


def _attach_date_mask(var):
    """Formata automaticamente uma StringVar como DD/MM/AAAA enquanto o usuário digita."""
    _busy = {"v": False}
    def on_write(*_):
        if _busy["v"]:
            return
        digits = "".join(c for c in var.get() if c.isdigit())[:8]
        out = digits[:2]
        if len(digits) > 2:
            out += "/" + digits[2:4]
        if len(digits) > 4:
            out += "/" + digits[4:8]
        if out != var.get():
            _busy["v"] = True
            var.set(out)
            _busy["v"] = False
    var.trace_add("write", on_write)


LEDGER_PATH = os.path.join(data_dir(), "analises.json")


def record_analysis(name, out_dir, metrics=None, graph=None, graph_ang=None):
    """Registra uma análise concluída (nome, data, métricas e gráficos) para o painel."""
    led = []
    if os.path.exists(LEDGER_PATH):
        try:
            with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                led = json.load(f)
        except Exception:
            led = []
    # substitui registro anterior do mesmo trial (evita duplicar ao reprocessar)
    led = [e for e in led if e.get("name") != name]
    led.append({"name": name, "out_dir": out_dir,
                "when": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
                "metrics": metrics or {}, "graph": graph or "", "graph_ang": graph_ang or ""})
    try:
        with open(LEDGER_PATH, "w", encoding="utf-8") as f:
            json.dump(led, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def load_ledger():
    if os.path.exists(LEDGER_PATH):
        try:
            with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def count_results(out_root=None):
    """Nº de análises já feitas e a última data, a partir do registro."""
    led = load_ledger()
    return len(led), (led[-1].get("when") if led else None)


def read_trial_metrics(out_dir, name):
    """Lê a linha DESTE trial no Resumo_Geral.xlsx salvo, para o app mostrar
    exatamente o que está na planilha (evita qualquer divergência app × Excel)."""
    path = os.path.join(out_dir, "Resumo_Geral.xlsx")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_excel(path, sheet_name="Resumo")
        rows = df[df["Paciente"] == name]
        if rows.empty:
            return {}
        r = rows.iloc[-1]
    except Exception:
        return {}

    def g(col):
        try:
            v = float(r[col]); return None if v != v else v
        except Exception:
            return None
    return {"vel": g("Mean_Speed_ms"), "wext": g("W_ext_J_per_kg_m"),
            "recovery": g("Recovery_pct"), "irl": g("IRL"),
            "cadencia": g("ET_Cadencia_Hz"), "comp_passo": g("ET_Comprimento_Passo_m"),
            "comp_passada": g("ET_Comprimento_Passada_m"), "tempo_passo": g("ET_Tempo_Passo_s")}


# Esqueleto simples (12 landmarks) para o replay de verificação.
SKELETON_LINKS = [
    ("ombro_esq", "ombro_dir"),
    ("ombro_esq", "quadril_esq"), ("ombro_dir", "quadril_dir"),
    ("quadril_esq", "quadril_dir"),
    ("quadril_esq", "joelho_esq"), ("joelho_esq", "tornozelo_esq"),
    ("tornozelo_esq", "calcanhar_esq"), ("tornozelo_esq", "ponta_esq"),
    ("quadril_dir", "joelho_dir"), ("joelho_dir", "tornozelo_dir"),
    ("tornozelo_dir", "calcanhar_dir"), ("tornozelo_dir", "ponta_dir"),
]
SKELETON_PTS = ["ombro_esq", "ombro_dir", "quadril_esq", "quadril_dir",
                "joelho_esq", "joelho_dir", "tornozelo_esq", "tornozelo_dir",
                "calcanhar_esq", "calcanhar_dir", "ponta_esq", "ponta_dir"]


def draw_stick(frame, row):
    """Desenha o esqueleto (12 pontos) sobre um frame BGR, a partir de uma linha do df."""
    def pt(nm):
        x = row.get(f"{nm}_x_px", float("nan")); y = row.get(f"{nm}_y_px", float("nan"))
        return (int(x), int(y)) if (x == x and y == y) else None
    for a, b in SKELETON_LINKS:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, (0, 210, 255), 2)
    for nm in SKELETON_PTS:
        p = pt(nm)
        if p:
            cv2.circle(frame, p, 4, (0, 255, 0), -1)
    return frame


# ============================ App ============================
class MarchaApp:
    SIDEBAR_W = 92
    ROW_H = 74
    ICON = 40

    def __init__(self, root: ttkb.Window):
        self.root = root
        self.root.title("Análise de Marcha — Painel")
        self.root.minsize(1120, 720)
        try:
            ico = os.path.join(base_dir(), "icone.ico")
            if os.path.exists(ico):
                self.root.iconbitmap(ico)
        except Exception:
            pass

        self.cfg = load_config()
        self._icons = {}
        self._play_gen = 0        # invalida reproduções antigas (evita loops empilhados)

        self.main = ttkb.Frame(self.root)
        self.main.pack(fill="both", expand=True)
        self.main.columnconfigure(0, weight=0, minsize=self.SIDEBAR_W)
        self.main.columnconfigure(1, weight=1)
        self.main.rowconfigure(0, weight=1)

        self.sidebar = tk.Frame(self.main, bg=SIDEBAR_BG, width=self.SIDEBAR_W,
                                highlightbackground=SIDEBAR_BORDER, highlightthickness=1)
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.grid_propagate(False)

        self.content = ttkb.Frame(self.main)
        self.content.grid(row=0, column=1, sticky="nsew")

        self._build_sidebar()
        self.show_home()

    # -------- ícones / sidebar --------
    def _icon(self, filename, size):
        key = f"{filename}:{size}"
        if key in self._icons:
            return self._icons[key]
        img = None
        p = os.path.join(ICONS_DIR, filename)
        if Image and os.path.exists(p):
            try:
                im = Image.open(p).convert("RGBA").resize((size, size), Image.LANCZOS)
                img = ImageTk.PhotoImage(im)
            except Exception:
                img = None
        self._icons[key] = img
        return img

    def _make_icon_row(self, icon_file, tip, command, bg=SIDEBAR_BG, hover=SIDEBAR_HOVER):
        row = tk.Frame(self.sidebar, bg=bg, height=self.ROW_H, width=self.SIDEBAR_W)
        row.pack(fill="x")
        row.pack_propagate(False)
        img = self._icon(icon_file, self.ICON)
        if img is not None:
            lbl = tk.Label(row, image=img, bg=bg, cursor="hand2")
        else:
            lbl = tk.Label(row, text=tip[:4], bg=bg, fg=SIDEBAR_CAP, cursor="hand2", font=("Segoe UI", 9))
        lbl.pack(expand=True)
        cap = tk.Label(row, text=tip, bg=bg, fg=SIDEBAR_CAP, font=("Segoe UI", 7, "bold"))
        cap.pack(side="bottom", pady=(0, 5))

        def enter(_):
            for w in (row, lbl, cap): w.configure(bg=hover)
        def leave(_):
            for w in (row, lbl, cap): w.configure(bg=bg)
        def click(_):
            try:
                command()
            except Exception as e:
                messagebox.showerror("Erro", f"Ocorreu um erro:\n{e}")
        for w in (row, lbl, cap):
            w.bind("<Enter>", enter); w.bind("<Leave>", leave); w.bind("<Button-1>", click)

    def _build_sidebar(self):
        tk.Frame(self.sidebar, bg=SIDEBAR_BG, height=14).pack(fill="x")
        self._make_icon_row("home.png",    "Início",        self.show_home)
        self._make_icon_row("Sujeito.png", "Participantes", self.show_subjects)
        self._make_icon_row("testes.png",  "Analisar",      self.show_run)
        self._make_icon_row("grafico.png", "Resultados",    self.show_visualizer)
        tk.Frame(self.sidebar, bg=SIDEBAR_BG).pack(fill="both", expand=True)
        self._make_icon_row("config.png",  "Config",        self.show_settings)
        self._make_icon_row("info.png",    "Sobre",         self.show_about)

    def clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _header(self, parent, title, subtitle=""):
        bar = ttkb.Frame(parent)
        bar.pack(fill="x", padx=24, pady=(20, 6))
        ttkb.Label(bar, text=title, font=("Segoe UI Semibold", 20), bootstyle="primary").pack(anchor="w")
        if subtitle:
            ttkb.Label(bar, text=subtitle, font=("Segoe UI", 10), bootstyle="secondary").pack(anchor="w")
        ttkb.Separator(parent).pack(fill="x", padx=24, pady=(6, 10))

    # ======================= HOME =======================
    def show_home(self):
        self.clear_content()
        f = ttkb.Frame(self.content)
        f.pack(fill="both", expand=True)

        self._header(f, "Análise de Marcha", "Painel do laboratório — pose por rtmlib (RTMPose Halpe-26) na GPU")

        subs = list_subjects()
        n_ana, last = count_results(self.cfg.get("out_root", ""))

        cards = ttkb.Frame(f)
        cards.pack(fill="x", padx=24, pady=6)
        for i in range(3):
            cards.columnconfigure(i, weight=1)

        def card(col, value, label, style):
            c = ttkb.Frame(cards, bootstyle=style, padding=16)
            c.grid(row=0, column=col, sticky="nsew", padx=8)
            ttkb.Label(c, text=str(value), font=("Segoe UI Semibold", 26),
                       bootstyle=f"{style}-inverse").pack(anchor="w")
            ttkb.Label(c, text=label, font=("Segoe UI", 10),
                       bootstyle=f"{style}-inverse").pack(anchor="w")

        card(0, len(subs), "Participantes cadastrados", "primary")
        card(1, n_ana, "Análises realizadas", "success")
        card(2, last if last else "—", "Última análise", "info")

        # Ações rápidas
        act = ttkb.Labelframe(f, text="Ações rápidas", padding=14)
        act.pack(fill="x", padx=24, pady=(16, 8))
        ttkb.Button(act, text="➕  Novo participante", bootstyle="primary-outline",
                    command=self.show_subject_form, width=24).pack(side="left", padx=6)
        ttkb.Button(act, text="▶  Rodar análise", bootstyle="success",
                    command=self.show_run, width=22).pack(side="left", padx=6)
        ttkb.Button(act, text="📊  Ver resultados", bootstyle="info-outline",
                    command=self.show_visualizer, width=22).pack(side="left", padx=6)

        # Participantes recentes
        if subs:
            recent = ttkb.Labelframe(f, text="Participantes", padding=8)
            recent.pack(fill="both", expand=True, padx=24, pady=8)
            cols = ("nome", "sexo", "massa_kg", "perna_cm", "criado_em")
            tv = ttkb.Treeview(recent, columns=cols, show="headings", height=8, bootstyle="primary")
            for c, t in zip(cols, ("Nome", "Sexo", "Massa (kg)", "Perna GT (cm)", "Cadastro")):
                tv.heading(c, text=t); tv.column(c, width=140, anchor="w")
            for s in subs:
                tv.insert("", "end", values=(s.get("nome", ""), s.get("sexo", ""),
                                             s.get("massa_kg", ""), s.get("perna_cm", ""),
                                             s.get("criado_em", "")))
            tv.pack(fill="both", expand=True)

    # ======================= PARTICIPANTES =======================
    def show_subjects(self):
        self.clear_content()
        f = ttkb.Frame(self.content)
        f.pack(fill="both", expand=True)
        self._header(f, "Participantes", "Cadastro de sujeitos (salvos em Sujeitos/)")

        top = ttkb.Frame(f); top.pack(fill="x", padx=24)
        ttkb.Button(top, text="➕  Novo participante", bootstyle="primary",
                    command=self.show_subject_form).pack(side="left")

        subs = list_subjects()
        wrap = ttkb.Frame(f); wrap.pack(fill="both", expand=True, padx=24, pady=10)
        cols = ("nome", "sexo", "nascimento", "massa_kg", "estatura_cm", "perna_cm", "coxa_cm")
        tv = ttkb.Treeview(wrap, columns=cols, show="headings", bootstyle="primary")
        for c, t in zip(cols, ("Nome", "Sexo", "Nascimento", "Massa (kg)", "Estatura (cm)",
                               "Perna GT (cm)", "Coxa (cm)")):
            tv.heading(c, text=t); tv.column(c, width=120, anchor="w")
        for s in subs:
            tv.insert("", "end", iid=s["_file"],
                      values=(s.get("nome", ""), s.get("sexo", ""), s.get("nascimento", ""),
                              s.get("massa_kg", ""), s.get("estatura_cm", ""),
                              s.get("perna_cm", ""), s.get("coxa_cm", "")))
        tv.pack(fill="both", expand=True, side="left")
        sb = ttkb.Scrollbar(wrap, command=tv.yview); sb.pack(side="right", fill="y")
        tv.configure(yscrollcommand=sb.set)

        bar = ttkb.Frame(f); bar.pack(fill="x", padx=24, pady=(0, 14))
        def _sel():
            iid = tv.focus()
            return next((s for s in subs if s["_file"] == iid), None)
        def _edit():
            s = _sel()
            if s: self.show_subject_form(s)
        def _del():
            s = _sel()
            if s and messagebox.askyesno("Excluir", f"Excluir participante '{s.get('nome','')}'?"):
                delete_subject(s); self.show_subjects()
        ttkb.Button(bar, text="Editar", bootstyle="secondary", command=_edit).pack(side="left", padx=4)
        ttkb.Button(bar, text="Excluir", bootstyle="danger-outline", command=_del).pack(side="left", padx=4)
        tv.bind("<Double-1>", lambda e: _edit())

    def show_subject_form(self, subject=None):
        self.clear_content()
        f = ttkb.Frame(self.content)
        f.pack(fill="both", expand=True)
        edit = subject is not None
        self._header(f, "Editar participante" if edit else "Novo participante",
                     "Dados usados para escalar a análise (massa, comprimentos)")

        body = ttkb.Frame(f, padding=(24, 6)); body.pack(fill="both", expand=True)
        fields = [
            ("nome", "Nome"), ("nascimento", "Nascimento (DD/MM/AAAA)"),
            ("sexo", "Sexo"), ("estatura_cm", "Estatura (cm)"),
            ("massa_kg", "Massa (kg)"), ("perna_cm", "Comprimento membro inf. GT (cm)"),
            ("coxa_cm", "Coxa (cm)"), ("avaliador", "Avaliador"),
            ("avaliador_email", "E-mail do avaliador"),
        ]
        vars_ = {}
        for i, (k, lbl) in enumerate(fields):
            ttkb.Label(body, text=lbl).grid(row=i, column=0, sticky="w", padx=6, pady=5)
            default = (subject or {}).get(k, "")
            if not default and k == "avaliador":
                default = self.cfg.get("avaliador", "")
            if not default and k == "avaliador_email":
                default = self.cfg.get("avaliador_email", "")
            v = tk.StringVar(value=str(default))
            if k == "sexo":
                w = ttkb.Combobox(body, textvariable=v, state="readonly", width=30,
                                  values=["Masculino", "Feminino", "Outro"])
                if not default:
                    v.set("Masculino")
            elif k == "nascimento":
                w = ttkb.Entry(body, textvariable=v, width=32)
                _attach_date_mask(v)
            else:
                w = ttkb.Entry(body, textvariable=v, width=32)
            w.grid(row=i, column=1, sticky="w", padx=6, pady=5)
            vars_[k] = v

        def _save():
            data = {k: v.get().strip() for k, v in vars_.items()}
            if not data["nome"]:
                messagebox.showwarning("Atenção", "Informe o nome do participante."); return
            if subject:
                data["_file"] = subject.get("_file")
                data["criado_em"] = subject.get("criado_em", datetime.date.today().strftime("%d/%m/%Y"))
            else:
                data["criado_em"] = datetime.date.today().strftime("%d/%m/%Y")
            save_subject(data)
            messagebox.showinfo("Salvo", f"Participante '{data['nome']}' salvo.")
            self.show_subjects()

        bar = ttkb.Frame(f, padding=(24, 8)); bar.pack(fill="x")
        ttkb.Button(bar, text="Salvar", bootstyle="success", command=_save, width=16).pack(side="left")
        ttkb.Button(bar, text="Cancelar", bootstyle="secondary-outline",
                    command=self.show_subjects, width=12).pack(side="left", padx=6)

    # ======================= RODAR ANÁLISE =======================
    def _num(self, s, default=None):
        try:
            return float(str(s).replace(",", "."))
        except Exception:
            return default

    def show_run(self):
        self.clear_content()
        f = ttkb.Frame(self.content); f.pack(fill="both", expand=True)
        self._header(f, "Rodar análise de marcha",
                     "Tudo acontece aqui: cones → processamento → resultados. Pose e direção são automáticas.")
        body = ttkb.Frame(f, padding=(24, 8)); body.pack(fill="both", expand=True)

        subs = list_subjects()
        names = [s.get("nome", "") for s in subs]
        subj_var = tk.StringVar(value=names[0] if names else "")
        mass_var, leg_var, thigh_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        video_var = tk.StringVar()
        out_var = tk.StringVar(value=self.cfg.get("out_root", ""))
        tini_var, tfim_var = tk.StringVar(value="0"), tk.StringVar(value="")

        def _fill(*_):
            s = next((x for x in subs if x.get("nome") == subj_var.get()), None)
            if s:
                mass_var.set(str(s.get("massa_kg", "")))
                lp = self._num(s.get("perna_cm", ""))
                leg_var.set(f"{lp/100:.3f}" if lp else "")
                cp = self._num(s.get("coxa_cm", ""))
                thigh_var.set(f"{cp/100:.3f}" if cp else "")
        subj_var.trace_add("write", _fill); _fill()

        r = 0
        def _row(label, var, browse=None):
            nonlocal r
            ttkb.Label(body, text=label).grid(row=r, column=0, sticky="w", pady=6, padx=6)
            e = ttkb.Entry(body, textvariable=var, width=40)
            e.grid(row=r, column=1, sticky="w", pady=6, padx=6)
            if browse:
                ttkb.Button(body, text="Escolher…", bootstyle="secondary", command=browse).grid(row=r, column=2, padx=6)
            r += 1

        ttkb.Label(body, text="Participante:").grid(row=r, column=0, sticky="w", pady=6, padx=6)
        ttkb.Combobox(body, textvariable=subj_var, values=names, state="readonly", width=38).grid(
            row=r, column=1, sticky="w", pady=6, padx=6); r += 1
        _row("Massa (kg):", mass_var)
        _row("Comprimento perna GT (m):", leg_var)
        _row("Coxa (m) [opcional]:", thigh_var)
        _row("Vídeo:", video_var, browse=lambda: video_var.set(filedialog.askopenfilename(
            title="Selecione o vídeo",
            filetypes=[("Vídeo", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")]) or video_var.get()))
        _row("Pasta de saída:", out_var, browse=lambda: out_var.set(
            filedialog.askdirectory(title="Pasta de saída") or out_var.get()))
        _row("Início (s):", tini_var)
        _row("Fim (s) [vazio = até o fim]:", tfim_var)

        def _start():
            video = video_var.get().strip()
            outdir = out_var.get().strip()
            bm = self._num(mass_var.get())
            leg = self._num(leg_var.get())
            thigh = self._num(thigh_var.get()) or (leg * 0.53 if leg else None)
            if not video or not os.path.exists(video):
                messagebox.showwarning("Atenção", "Selecione um vídeo válido."); return
            if not outdir:
                messagebox.showwarning("Atenção", "Selecione a pasta de saída."); return
            if bm is None:
                messagebox.showwarning("Atenção", "Informe a massa (kg)."); return
            if thigh is None:
                messagebox.showwarning("Atenção", "Informe a coxa (m) ou o comprimento da perna."); return
            self.cfg["out_root"] = outdir; save_config(self.cfg)
            name = os.path.splitext(os.path.basename(video))[0]
            if subj_var.get():
                name = f"{_slug(subj_var.get())}_{name}"
            cfg = {"name": name, "video": video, "outdir": outdir, "bm": bm, "leg": leg,
                   "thigh": thigh, "t_ini": self._num(tini_var.get(), 0.0),
                   "t_fim": self._num(tfim_var.get(), None)}
            self._cone_step(cfg)

        ttkb.Button(f, text="Continuar → marcar cones", bootstyle="success",
                    command=_start, width=26).pack(anchor="w", padx=30, pady=(6, 16))

    # ---------- Etapa: calibração de cones (manual, no canvas) ----------
    def _cone_step(self, cfg):
        self.clear_content()
        f = ttkb.Frame(self.content); f.pack(fill="both", expand=True)
        self._header(f, "Calibração por cones",
                     "Marque a BASE de cada cone (esq→dir) e depois o TOPO de cada um, na mesma ordem.")

        frame0 = engine.get_frame_at_time(cfg["video"],
                                          cfg["t_ini"] if cfg["t_ini"] else 0.0)
        if frame0 is None:
            messagebox.showerror("Erro", "Não foi possível ler o vídeo."); return
        H, W = frame0.shape[:2]
        maxw, maxh = 940, 520
        scale = min(maxw / W, maxh / H, 1.0)
        disp_w, disp_h = int(W * scale), int(H * scale)
        rgb = cv2.cvtColor(cv2.resize(frame0, (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        bar = ttkb.Frame(f, padding=(24, 2)); bar.pack(fill="x")
        phase = {"v": 1}  # 1 = base, 2 = topo
        bases, tops = [], []
        status = tk.StringVar(value="FASE 1/2 — clique na BASE de cada cone (esq → dir).")
        ttkb.Label(bar, textvariable=status, bootstyle="primary", font=("Segoe UI Semibold", 11)).pack(anchor="w")

        cvs = tk.Canvas(f, width=disp_w, height=disp_h, highlightthickness=1,
                        highlightbackground="#c9d3dd", cursor="crosshair")
        cvs.pack(padx=24, pady=8)
        cvs.create_image(0, 0, anchor="nw", image=photo)
        cvs.image = photo

        def _redraw():
            cvs.delete("mark")
            for i, (x, y) in enumerate(bases):
                cvs.create_oval(x*scale-5, y*scale-5, x*scale+5, y*scale+5, fill="#2ecc71", outline="", tags="mark")
                cvs.create_text(x*scale+10, y*scale-8, text=f"B{i+1}", fill="#2ecc71", tags="mark", anchor="w")
            for i, (x, y) in enumerate(tops):
                cvs.create_oval(x*scale-5, y*scale-5, x*scale+5, y*scale+5, fill="#e67e22", outline="", tags="mark")
                bx, by = bases[i]
                cvs.create_line(bx*scale, by*scale, x*scale, y*scale, fill="#e67e22", tags="mark")

        def _click(ev):
            ox, oy = ev.x/scale, ev.y/scale
            if phase["v"] == 1:
                bases.append((ox, oy))
            elif len(tops) < len(bases):
                tops.append((ox, oy))
            _redraw()
        cvs.bind("<Button-1>", _click)

        def _undo():
            if phase["v"] == 1 and bases:
                bases.pop()
            elif phase["v"] == 2 and tops:
                tops.pop()
            _redraw()
        def _clear():
            bases.clear(); tops.clear(); phase["v"] = 1
            status.set("FASE 1/2 — clique na BASE de cada cone (esq → dir).")
            _redraw()
        def _next():
            if phase["v"] == 1:
                if len(bases) < 2:
                    messagebox.showwarning("Cones", "Marque pelo menos 2 bases."); return
                phase["v"] = 2
                status.set(f"FASE 2/2 — clique no TOPO de cada cone (mesma ordem)  0/{len(bases)}.")
            _redraw()
        def _confirm():
            if len(bases) < 2:
                messagebox.showwarning("Cones", "Marque pelo menos 2 cones."); return
            if len(tops) != len(bases):
                messagebox.showwarning("Cones", f"Falta marcar o topo de alguns cones ({len(tops)}/{len(bases)})."); return
            b = np.array(bases, float); t = np.array(tops, float)
            order = np.argsort(b[:, 0]); b = b[order]; t = t[order]
            cone_px = b[:, 0]
            cone_real = np.arange(len(cone_px)) * engine.CONE_SPACING_M
            h_px = np.abs(b[:, 1] - t[:, 1]); h_px[h_px == 0] = np.nan
            cone_scale_y = engine.CONE_HEIGHT_M / h_px
            self._processing_step(cfg, (cone_px, cone_real, cone_scale_y))
        def _skip():
            if messagebox.askyesno("Pular calibração",
                                   "Sem cones, a escala usa só a coxa (menos precisa). Continuar assim?"):
                self._processing_step(cfg, (None, None, None))

        def _update_topo_count(*_):
            if phase["v"] == 2:
                status.set(f"FASE 2/2 — clique no TOPO de cada cone (mesma ordem)  {len(tops)}/{len(bases)}.")
        cvs.bind("<ButtonRelease-1>", _update_topo_count, add="+")

        btns = ttkb.Frame(f, padding=(24, 6)); btns.pack(fill="x")
        ttkb.Button(btns, text="Desfazer", bootstyle="secondary", command=_undo).pack(side="left", padx=3)
        ttkb.Button(btns, text="Limpar", bootstyle="secondary-outline", command=_clear).pack(side="left", padx=3)
        ttkb.Button(btns, text="Fase topo →", bootstyle="info", command=_next).pack(side="left", padx=3)
        ttkb.Button(btns, text="Confirmar e processar", bootstyle="success", command=_confirm).pack(side="left", padx=12)
        ttkb.Button(btns, text="Pular (sem cones)", bootstyle="warning-outline", command=_skip).pack(side="right", padx=3)

    # ---------- Etapa: processamento (vídeo ao vivo + barra, EM THREAD) ----------
    def _processing_step(self, cfg, cones):
        self.clear_content()
        f = ttkb.Frame(self.content); f.pack(fill="both", expand=True)
        self._header(f, "Processando", os.path.basename(cfg["video"]))

        vlabel = ttkb.Label(f, anchor="center")
        vlabel.pack(padx=24, pady=8)
        stage_var = tk.StringVar(value="Preparando…")
        ttkb.Label(f, textvariable=stage_var, font=("Segoe UI Semibold", 11)).pack(anchor="w", padx=26)
        pb = ttkb.Progressbar(f, mode="determinate", maximum=100, length=760, bootstyle="success-striped")
        pb.pack(fill="x", padx=26, pady=(4, 2))
        detail_var = tk.StringVar(value="")
        ttkb.Label(f, textvariable=detail_var, bootstyle="secondary").pack(anchor="w", padx=26)
        self._proc_img = None

        import threading, time as _t
        # Estado compartilhado worker→UI. Os callbacks (rodados na thread) só ESCREVEM
        # aqui; toda mexida no tkinter fica na thread principal (no poll via after).
        state = {"frame": None, "stage": "", "done": 0, "total": 0,
                 "finished": False, "error": None, "result": None, "t0": None}

        def on_frame(bgr, done, total):
            state["frame"] = bgr
        def on_progress(stage, done, total):
            state["stage"], state["done"], state["total"] = stage, done, total

        def worker():
            try:
                state["result"] = self._extract(cfg, on_frame, on_progress)
            except Exception as e:
                import traceback; traceback.print_exc()
                state["error"] = str(e)
            finally:
                state["finished"] = True

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            if not f.winfo_exists():
                return
            bgr = state["frame"]
            if bgr is not None:
                try:
                    hh, ww = bgr.shape[:2]; s = min(760 / ww, 420 / hh)
                    rgb = cv2.cvtColor(cv2.resize(bgr, (int(ww*s), int(hh*s))), cv2.COLOR_BGR2RGB)
                    self._proc_img = ImageTk.PhotoImage(Image.fromarray(rgb))
                    vlabel.configure(image=self._proc_img)
                except Exception:
                    pass
            if state["stage"]:
                stage_var.set(state["stage"])
            if state["total"]:
                if state["t0"] is None:
                    state["t0"] = _t.time()
                d, tot = state["done"], state["total"]
                pct = min(100.0, 100.0 * d / max(tot, 1)); pb["value"] = pct
                el = _t.time() - state["t0"]
                eta = f" · ~{int(el*(tot-d)/d)}s restantes" if 0 < d < tot else ""
                detail_var.set(f"{d}/{tot} frames · {pct:.0f}%{eta}")
            if state["finished"]:
                if state["error"]:
                    messagebox.showerror("Erro no processamento", state["error"]); self.show_run(); return
                r = state["result"]
                if not r or r[0] is None or r[0].empty or not r[1]:
                    messagebox.showerror("Erro", "Nenhum landmark detectado no vídeo."); self.show_run(); return
                df_raw, frames, fps = r
                self._trim_step(cfg, cones, df_raw, frames, fps); return
            self.root.after(40, poll)
        self.root.after(40, poll)

    # ---------- Extração (detecção) e análise, separadas p/ permitir corte ----------
    def _extract(self, cfg, on_frame=None, on_progress=None):
        df_raw, frames, fps, w, h = engine.process_video(
            cfg["video"],
            start_sec=cfg["t_ini"] if cfg["t_ini"] else None,
            end_sec=cfg["t_fim"], backend="rtmlib",
            progress_cb=on_progress, frame_cb=on_frame)
        return df_raw.reset_index(drop=True), frames, fps

    def _analyze(self, cfg, cones, df_raw, frames, fps, on_progress=None):
        """Roda TODA a análise (escala → eventos → mecânica → salvar) sobre os
        frames dados. Re-executável em qualquer faixa (corte)."""
        cone_px, cone_real, cone_scale_y = cones
        bm, leg, thigh = cfg["bm"], cfg["leg"], cfg["thigh"]
        df_raw = df_raw.reset_index(drop=True)
        if df_raw.empty:
            raise RuntimeError("Faixa selecionada sem dados.")

        if on_progress: on_progress("Calibrando e calculando…", 0, 0)
        thigh_px = np.sqrt((df_raw["quadril_dir_x_px"] - df_raw["joelho_dir_x_px"])**2 +
                           (df_raw["quadril_dir_y_px"] - df_raw["joelho_dir_y_px"])**2).mean()
        sx, sy = engine.build_scale_functions(cone_px, cone_real, cone_scale_y, thigh_px, thigh)

        df = engine.fix_feet(df_raw)
        df = engine.apply_scale(df, sx, sy)
        df = engine.filter_coords(df, fps)
        df = engine.compute_angles(df)
        df = engine.filter_angles(df, fps)

        x_com, y_com = engine.estimate_com(df)
        df["x_com_m"] = x_com; df["y_com_m"] = y_com
        mech = engine.compute_mechanical_work(x_com, y_com, bm, fps, leg_length=leg)
        walk_dir = engine._walk_direction(x_com)
        evts = engine.detect_events_robust(df, fps, x_com, walk_dir=walk_dir)

        out_dir = os.path.join(cfg["outdir"], RESULTS_DIRNAME)
        os.makedirs(out_dir, exist_ok=True)
        res = {"cfg": cfg, "df": df, "df_raw": df_raw, "frames": frames, "mech": mech,
               "evts": evts, "fps": fps, "x_com": x_com, "out_dir": out_dir,
               "name": cfg["name"], "walk_dir": walk_dir}
        self._finalize_and_save(res)
        return res

    def _run_pipeline(self, cfg, cones, on_frame=None, on_progress=None):
        """Conveniência (faixa inteira) — usado em testes."""
        df_raw, frames, fps = self._extract(cfg, on_frame, on_progress)
        if df_raw.empty:
            raise RuntimeError("Nenhum landmark detectado no vídeo.")
        return self._analyze(cfg, cones, df_raw, frames, fps, on_progress)

    # ---------- Etapa: revisão + corte da faixa (esqueleto + barras) ----------
    def _trim_step(self, cfg, cones, df_raw, frames, fps):
        self.clear_content()
        f = ttkb.Frame(self.content); f.pack(fill="both", expand=True)
        self._header(f, "Revisão e corte",
                     "Confira o esqueleto e, se quiser, corte o começo/fim onde ele se perde. "
                     "A análise usará SÓ a faixa selecionada.")
        disp_df = self._smoothed_px_df(df_raw, fps)
        N = len(frames)

        lbl = ttkb.Label(f, anchor="center"); lbl.pack(padx=24, pady=(6, 4))
        self._trim_img = None

        def show_frame(idx):
            idx = int(max(0, min(N - 1, idx)))
            frame = frames[idx].copy()
            draw_stick(frame, disp_df.iloc[idx])
            hh, ww = frame.shape[:2]; s = min(820 / ww, 430 / hh)
            rgb = cv2.cvtColor(cv2.resize(frame, (int(ww*s), int(hh*s))), cv2.COLOR_BGR2RGB)
            self._trim_img = ImageTk.PhotoImage(Image.fromarray(rgb))
            lbl.configure(image=self._trim_img)

        ctrl = ttkb.Frame(f, padding=(24, 2)); ctrl.pack(fill="x")
        start_var, end_var = tk.IntVar(value=0), tk.IntVar(value=N - 1)
        info = tk.StringVar()

        def upd_info():
            i0, i1 = start_var.get(), end_var.get()
            dur = (i1 - i0 + 1) / fps if fps else 0
            info.set(f"Faixa selecionada: frame {i0} → {i1}   ({i1 - i0 + 1} frames · {dur:.1f}s)")

        row1 = ttkb.Frame(ctrl); row1.pack(fill="x", pady=2)
        ttkb.Label(row1, text="Início", width=8).pack(side="left")
        s_start = ttkb.Scale(row1, from_=0, to=N - 1, value=0)
        s_start.pack(side="left", fill="x", expand=True, padx=8)
        row2 = ttkb.Frame(ctrl); row2.pack(fill="x", pady=2)
        ttkb.Label(row2, text="Fim", width=8).pack(side="left")
        s_end = ttkb.Scale(row2, from_=0, to=N - 1, value=N - 1)
        s_end.pack(side="left", fill="x", expand=True, padx=8)

        def on_start(v):
            i0 = int(float(v)); start_var.set(i0)
            if i0 > end_var.get():
                end_var.set(i0); s_end.set(i0)
            show_frame(i0); upd_info()

        def on_end(v):
            i1 = int(float(v)); end_var.set(i1)
            if i1 < start_var.get():
                start_var.set(i1); s_start.set(i1)
            show_frame(i1); upd_info()
        s_start.configure(command=on_start)
        s_end.configure(command=on_end)

        ttkb.Label(ctrl, textvariable=info, bootstyle="secondary").pack(anchor="w", pady=(4, 0))
        upd_info()

        bar = ttkb.Frame(f, padding=(24, 8)); bar.pack(fill="x")
        ttkb.Button(bar, text="▶  Reproduzir faixa", bootstyle="info",
                    command=lambda: self._play_range(frames, disp_df, start_var.get(),
                                                     end_var.get(), fps, lbl)).pack(side="left", padx=4)
        ttkb.Button(bar, text="↺  Faixa toda", bootstyle="secondary",
                    command=lambda: (start_var.set(0), end_var.set(N - 1),
                                     s_start.set(0), s_end.set(N - 1),
                                     show_frame(0), upd_info())).pack(side="left", padx=4)
        ttkb.Button(bar, text="✓  Confirmar faixa e analisar", bootstyle="success",
                    command=lambda: self._confirm_trim(cfg, cones, df_raw, frames, fps,
                                                       start_var.get(), end_var.get())).pack(side="right", padx=4)
        show_frame(0)

    def _play_range(self, frames, disp_df, i0, i1, fps, lbl):
        i0, i1 = int(i0), int(i1)
        if i1 <= i0:
            return
        delay = max(int(1000.0 / (fps or 30)), 15)
        self._play_gen += 1
        gen = self._play_gen
        st = {"i": i0}

        def step():
            if gen != self._play_gen or not lbl.winfo_exists() or st["i"] > i1:
                return
            frame = frames[st["i"]].copy()
            draw_stick(frame, disp_df.iloc[st["i"]])
            hh, ww = frame.shape[:2]; s = min(820 / ww, 430 / hh)
            rgb = cv2.cvtColor(cv2.resize(frame, (int(ww*s), int(hh*s))), cv2.COLOR_BGR2RGB)
            self._trim_img = ImageTk.PhotoImage(Image.fromarray(rgb))
            lbl.configure(image=self._trim_img)
            st["i"] += 1
            self.root.after(delay, step)
        self.root.after(delay, step)

    def _confirm_trim(self, cfg, cones, df_raw, frames, fps, i0, i1):
        i0, i1 = int(i0), int(i1)
        if i1 - i0 + 1 < 15:
            messagebox.showwarning("Faixa curta",
                                   "Selecione uma faixa maior (mínimo ~15 frames) para a análise ser confiável.")
            return
        sub_df = df_raw.iloc[i0:i1 + 1].reset_index(drop=True)
        sub_frames = frames[i0:i1 + 1]
        self.clear_content()
        f = ttkb.Frame(self.content); f.pack(fill="both", expand=True)
        self._header(f, "Calculando…", cfg["name"])
        pb = ttkb.Progressbar(f, mode="indeterminate", length=420, bootstyle="success-striped")
        pb.pack(padx=26, pady=20); pb.start(12); self.root.update()
        try:
            res = self._analyze(cfg, cones, sub_df, sub_frames, fps)
        except Exception as e:
            import traceback; traceback.print_exc()
            pb.stop(); messagebox.showerror("Erro na análise", str(e)); self.show_run(); return
        pb.stop()
        self._show_results(res)

    def _finalize_and_save(self, res):
        """Espaço-temporais + mecânica por passo + salvamento (mesmas pastas do motor)."""
        df, evts, fps = res["df"], res["evts"], res["fps"]
        bm, leg = res["cfg"]["bm"], res["cfg"]["leg"]
        x_com, mech = res["x_com"], res["mech"]
        TD_R = evts.get("TD_R", np.array([], int)); TO_R = evts.get("TO_R", np.array([], int))
        TD_L = evts.get("TD_L", np.array([], int)); TO_L = evts.get("TO_L", np.array([], int))

        df_steps, df_strides, grp_step, grp_stride = engine.build_spatiotemporal(
            df, {"TD_R": TD_R, "TO_R": TO_R, "TD_L": TD_L, "TO_L": TO_L}, fps, x_com, bm, leg)
        df_ang_stride = engine.angular_per_stride(df, TD_R, fps)
        step_events = sorted(
            [(int(row["TD_Frame"]), int(row["TC_Contra_Frame"]))
             for _, row in df_steps.iterrows()
             if not pd.isna(row.get("TC_Contra_Frame")) and not pd.isna(row.get("TD_Frame"))],
            key=lambda x: x[0])
        df_mech_step = engine.compute_step_mechanics(
            mech["_Ep"], mech["_Ekf"], mech["_Ekv"], mech["_x_com"], bm, fps, step_events)
        mec_interp_step = engine.interpolate_mechanics_per_step(
            mech["_Ep"], mech["_Ekf"], mech["_Ekv"], mech["_Emec"], step_events)
        df_ang_step = engine.angular_per_step(df, step_events, fps)
        ang_interp_step = engine.interpolate_angles_per_step(df, step_events)

        evts_ok = {k: evts[k] for k in evts}
        engine.save_outputs(res["out_dir"], res["name"], df, df_steps, df_strides,
                            grp_step, grp_stride, df_ang_stride, df_ang_step,
                            mech, df_mech_step, mec_interp_step, ang_interp_step, evts_ok, fps)
        engine.save_quality_report(res["out_dir"], res["name"], res["df_raw"], evts_ok,
                                   backend="rtmlib", detector="robusto", roi_info=None)
        res["n_pass"] = max(len(TD_R) - 1, 0)
        res["n_steps"] = len(step_events)
        # Métricas LIDAS DO EXCEL salvo deste trial → app mostra idêntico à planilha.
        m = read_trial_metrics(res["out_dir"], res["name"])
        m["passadas"] = res["n_pass"]
        m["passos"] = res["n_steps"]
        res["metrics"] = m
        # Caminhos EXATOS dos gráficos deste trial (nada de casar por prefixo).
        gdir = os.path.join(res["out_dir"], "Graficos")
        res["graph_mech"] = os.path.join(gdir, f"{res['name']}_Sinais_Mecanica.png")
        res["graph_ang"] = os.path.join(gdir, f"{res['name']}_Angulos_Bilaterais.png")
        res["graph"] = res["graph_mech"]
        record_analysis(res["name"], res["out_dir"], res["metrics"],
                        res["graph_mech"], res["graph_ang"])

    @staticmethod
    def _metric_rows(m):
        def s(v, fmt):
            return fmt.format(v) if isinstance(v, (int, float)) else "—"
        return [
            ("Velocidade média", s(m.get("vel"), "{:.2f} m/s")),
            ("Cadência", s(m.get("cadencia"), "{:.2f} Hz")),
            ("Comprimento de passo", s(m.get("comp_passo"), "{:.2f} m")),
            ("Comprimento de passada", s(m.get("comp_passada"), "{:.2f} m")),
            ("Tempo de passo", s(m.get("tempo_passo"), "{:.2f} s")),
            ("Passadas (Dir)", str(m.get("passadas", "—"))),
            ("Passos", str(m.get("passos", "—"))),
            ("Trabalho mecânico (Wext)", s(m.get("wext"), "{:.2f} J/(kg·m)")),
            ("Recuperação pendular", s(m.get("recovery"), "{:.1f} %")),
            ("IRL", s(m.get("irl"), "{:.3f}")),
        ]

    def _metrics_and_graph(self, parent, metrics, graphs):
        """Painel padrão: tabela de dados principais (do Excel do trial) + gráfico,
        com seletor Mecânico/Angular. `graphs` pode ser um caminho ou um dict
        {rótulo: caminho}."""
        if isinstance(graphs, str) or graphs is None:
            graphs = {"Gráfico": graphs or ""}
        wrap = ttkb.Frame(parent, padding=(24, 6)); wrap.pack(fill="both", expand=True)
        left = ttkb.Labelframe(wrap, text="Dados principais", padding=12)
        left.pack(side="left", fill="y")
        for i, (k, v) in enumerate(self._metric_rows(metrics or {})):
            ttkb.Label(left, text=k, bootstyle="secondary").grid(row=i, column=0, sticky="w", padx=6, pady=5)
            ttkb.Label(left, text=v, font=("Segoe UI Semibold", 12)).grid(row=i, column=1, sticky="e", padx=10, pady=5)

        right = ttkb.Frame(wrap); right.pack(side="left", fill="both", expand=True, padx=(14, 0))
        keys = [k for k, p in graphs.items() if p and os.path.exists(p)]
        sel = tk.StringVar(value=keys[0] if keys else "")
        if len(keys) > 1:
            segbar = ttkb.Frame(right); segbar.pack(anchor="w", pady=(0, 6))
            ttkb.Label(segbar, text="Gráfico: ").pack(side="left")
            for k in keys:
                ttkb.Radiobutton(segbar, text=k, value=k, variable=sel,
                                 bootstyle="primary-toolbutton",
                                 command=lambda: render()).pack(side="left", padx=2)
        holder = ttkb.Label(right, anchor="center"); holder.pack(fill="both", expand=True)
        self._panel_img = None

        def render():
            p = graphs.get(sel.get(), "")
            if p and os.path.exists(p) and Image:
                try:
                    im = Image.open(p); im.thumbnail((780, 560), Image.LANCZOS)
                    self._panel_img = ImageTk.PhotoImage(im)
                    holder.configure(image=self._panel_img, text="")
                    return
                except Exception:
                    pass
            holder.configure(image="", text="Sem gráfico para exibir.")
        render()

    # ---------- Etapa: resultados (gráfico + dados) ----------
    def _show_results(self, res):
        self.clear_content()
        f = ttkb.Frame(self.content); f.pack(fill="both", expand=True)
        self._header(f, "Resultados", res["name"])
        self._metrics_and_graph(f, res.get("metrics"),
                                {"Mecânico": res.get("graph_mech") or res.get("graph"),
                                 "Angular": res.get("graph_ang")})

        bar = ttkb.Frame(f, padding=(24, 8)); bar.pack(fill="x")
        ttkb.Button(bar, text="▶  Rever vídeo com esqueleto", bootstyle="info",
                    command=lambda: self._replay_skeleton(res)).pack(side="left", padx=4)
        ttkb.Button(bar, text="✍  Marcar TD e TO manualmente", bootstyle="primary-outline",
                    command=lambda: self._manual_review(res)).pack(side="left", padx=4)
        ttkb.Button(bar, text="＋  Nova análise", bootstyle="success", command=self.show_run).pack(side="right", padx=4)

    def _manual_review(self, res):
        """Abre a revisão de TD/TO do motor; se aceita, recalcula e re-salva."""
        try:
            rev = engine.EventReviewWindow(self.root, res["df"], res["evts"], res["fps"])
        except Exception as e:
            messagebox.showerror("Erro", str(e)); return
        finally:
            try:
                self.root.style.theme_use(THEME)
            except Exception:
                pass
        if not getattr(rev, "accepted", False):
            return
        ok = rev.get_events()
        for k in ("td_sig_R", "to_sig_R", "td_sig_L", "to_sig_L"):
            if k in res["evts"]:
                ok[k] = res["evts"][k]
        res["evts"] = ok
        self._finalize_and_save(res)
        messagebox.showinfo("Atualizado", "Eventos revisados e resultados re-salvos.")
        self._show_results(res)

    def _smoothed_px_df(self, df_raw, fps):
        """Suaviza os landmarks em PIXELS com o mesmo Butterworth do motor (3 Hz,
        fase zero), só para o esqueleto do replay ficar limpo — não afeta a análise."""
        df = df_raw.copy()
        cutoff = getattr(engine, "CUTOFF_HZ", 3.0)
        n = len(df)
        for nm in SKELETON_PTS:
            for ax in ("x", "y"):
                col = f"{nm}_{ax}_px"
                if col not in df.columns:
                    continue
                s = df[col].ffill().bfill()
                if s.notna().sum() >= 10 and n >= 15:
                    try:
                        df[col] = engine.butter_lp(s.values, cutoff, fps)
                        continue
                    except Exception:
                        pass
                df[col] = s.values
        return df

    # ---------- Replay de verificação (vídeo + esqueleto, velocidade normal) ----------
    def _replay_skeleton(self, res):
        frames = res.get("frames") or []
        if "df_disp" not in res and res.get("df_raw") is not None:
            res["df_disp"] = self._smoothed_px_df(res["df_raw"], res.get("fps") or 30.0)
        df = res.get("df_disp")
        if not frames or df is None or len(df) == 0:
            messagebox.showinfo("Replay", "Sem frames para reproduzir."); return
        self.clear_content()
        f = ttkb.Frame(self.content); f.pack(fill="both", expand=True)
        self._header(f, "Verificação — vídeo com esqueleto",
                     f"{res['name']} · velocidade normal · confira se o rastreamento pegou o sujeito certo")
        lbl = ttkb.Label(f, anchor="center"); lbl.pack(padx=24, pady=8)
        ctrl = ttkb.Frame(f, padding=(24, 6)); ctrl.pack(fill="x")

        fps = res.get("fps") or 30.0
        delay = max(int(1000.0 / fps), 15)
        n = min(len(frames), len(df))
        self._play_gen += 1
        gen = self._play_gen
        st = {"i": 0, "stop": False, "img": None}

        def finish():
            ttkb.Button(ctrl, text="Assistir novamente", bootstyle="info",
                        command=lambda: self._replay_skeleton(res)).pack(side="left", padx=4)
            ttkb.Button(ctrl, text="Voltar aos resultados", bootstyle="success",
                        command=lambda: self._show_results(res)).pack(side="left", padx=4)

        def step():
            if gen != self._play_gen or not lbl.winfo_exists():   # substituído / saiu da tela
                return
            if st["stop"] or st["i"] >= n:
                if not st["stop"]:
                    finish()
                return
            i = st["i"]
            frame = frames[i].copy()
            draw_stick(frame, df.iloc[i])
            hh, ww = frame.shape[:2]
            s = min(840 / ww, 470 / hh)
            rgb = cv2.cvtColor(cv2.resize(frame, (int(ww * s), int(hh * s))), cv2.COLOR_BGR2RGB)
            st["img"] = ImageTk.PhotoImage(Image.fromarray(rgb))
            lbl.configure(image=st["img"])
            st["i"] += 1
            self.root.after(delay, step)

        def _stop():
            st["stop"] = True
            finish()
        ttkb.Button(ctrl, text="Parar", bootstyle="secondary", command=_stop).pack(side="left", padx=4)
        self.root.after(delay, step)

    # ======================= VISUALIZADOR (por análise) =======================
    def show_visualizer(self):
        self.clear_content()
        f = ttkb.Frame(self.content); f.pack(fill="both", expand=True)
        self._header(f, "Resultados", "Análises realizadas — selecione uma para ver os dados e o gráfico")

        led = load_ledger()
        if not led:
            ttkb.Label(f, text="Nenhuma análise ainda. Vá em Analisar para rodar a primeira.",
                       padding=24, bootstyle="secondary").pack(anchor="w")
            return

        items = list(reversed(led))   # mais recentes no topo
        wrap = ttkb.Frame(f, padding=(24, 8)); wrap.pack(fill="both", expand=True)
        left = ttkb.Frame(wrap); left.pack(side="left", fill="y")
        detail = ttkb.Frame(wrap); detail.pack(side="left", fill="both", expand=True, padx=(12, 0))

        lb = tk.Listbox(left, width=36, height=24, activestyle="none",
                        font=("Segoe UI", 10), highlightthickness=1)
        lb.pack(side="left", fill="y")
        sb = ttkb.Scrollbar(left, command=lb.yview); sb.pack(side="left", fill="y")
        lb.configure(yscrollcommand=sb.set)
        for e in items:
            lb.insert("end", f" {e.get('name','?')}   ·  {e.get('when','')}")

        def show(idx):
            for w in detail.winfo_children():
                w.destroy()
            e = items[idx]
            self._metrics_and_graph(detail, e.get("metrics", {}),
                                    {"Mecânico": e.get("graph", ""),
                                     "Angular": e.get("graph_ang", "")})

        def on_sel(_):
            sel = lb.curselection()
            if sel:
                show(sel[0])
        lb.bind("<<ListboxSelect>>", on_sel)
        lb.selection_set(0); show(0)

    # ======================= CONFIG / SOBRE =======================
    def show_settings(self):
        self.clear_content()
        f = ttkb.Frame(self.content); f.pack(fill="both", expand=True)
        self._header(f, "Configurações")
        body = ttkb.Frame(f, padding=(24, 8)); body.pack(fill="x")

        out_var = tk.StringVar(value=self.cfg.get("out_root", ""))
        av_var = tk.StringVar(value=self.cfg.get("avaliador", ""))
        em_var = tk.StringVar(value=self.cfg.get("avaliador_email", ""))
        rows = [("Pasta de saída padrão:", out_var, True),
                ("Avaliador:", av_var, False),
                ("E-mail do avaliador:", em_var, False)]
        for i, (lbl, var, browse) in enumerate(rows):
            ttkb.Label(body, text=lbl).grid(row=i, column=0, sticky="w", padx=6, pady=6)
            ttkb.Entry(body, textvariable=var, width=44).grid(row=i, column=1, sticky="w", padx=6, pady=6)
            if browse:
                ttkb.Button(body, text="…", width=3, bootstyle="secondary",
                            command=lambda v=var: v.set(filedialog.askdirectory() or v.get())
                            ).grid(row=i, column=2)

        def _save():
            self.cfg.update({"out_root": out_var.get().strip(),
                             "avaliador": av_var.get().strip(),
                             "avaliador_email": em_var.get().strip()})
            save_config(self.cfg)
            messagebox.showinfo("Configurações", "Salvo.")
        ttkb.Button(f, text="Salvar", bootstyle="success", command=_save, width=16).pack(anchor="w", padx=30, pady=10)

        # ---- Importar cadastros de um Excel de caracterização ----
        imp = ttkb.Labelframe(f, text="Importar participantes (Excel)", padding=12)
        imp.pack(fill="x", padx=30, pady=(6, 12))
        ttkb.Label(imp, bootstyle="secondary", text=(
            "Carrega uma planilha e cadastra os participantes automaticamente. Colunas usadas: "
            "Subject (nome), Body Mass (massa), Height (estatura), Leg Length Right (perna) e "
            "thigh-length (coxa).")).pack(anchor="w", pady=(0, 8))

        def _import():
            path = filedialog.askopenfilename(
                title="Selecione o Excel de caracterização",
                filetypes=[("Excel", "*.xlsx *.xls"), ("Todos", "*.*")])
            if not path:
                return
            try:
                n_imp, n_skip, warn = import_subjects_from_excel(path)
            except Exception as e:
                messagebox.showerror("Erro na importação", str(e)); return
            if warn:
                messagebox.showwarning("Importação", warn); return
            messagebox.showinfo("Importação concluída",
                                f"{n_imp} participante(s) importado(s)/atualizado(s).\n"
                                f"{n_skip} linha(s) ignorada(s) (sem nome).")
        ttkb.Button(imp, text="📂  Escolher Excel e importar", bootstyle="primary",
                    command=_import).pack(anchor="w")

    def show_about(self):
        self.clear_content()
        f = ttkb.Frame(self.content); f.pack(fill="both", expand=True)
        self._header(f, "Sobre")
        body = ttkb.Frame(f, padding=(30, 8)); body.pack(fill="both", expand=True)

        title = "Análise de Marcha em Grama"
        paras = [
            "Estimativa de pose 2D baseada em modelos de visão computacional executados "
            "localmente via ONNX Runtime com aceleração por GPU. O pipeline utiliza detecção "
            "corporal por RTMDet/YOLOX e estimação de pontos anatômicos por RTMPose no formato "
            "Halpe-26, permitindo a identificação quadro a quadro de segmentos corporais "
            "relevantes para a análise biomecânica da marcha.",
            "O sistema realiza rastreamento automático do participante ao longo da trajetória, "
            "com filtragem de indivíduos parados, elementos de fundo e detecções espúrias. A "
            "direção do deslocamento é identificada automaticamente, reduzindo a necessidade de "
            "intervenção manual durante o processamento.",
            "A análise fornece parâmetros espaço-temporais, angulares bilaterais e variáveis "
            "mecânicas da marcha, incluindo estimativas de trabalho externo e recuperação "
            "mecânica. Também permite ajuste da faixa analisada e revisão manual dos eventos "
            "detectados, garantindo maior controle sobre a qualidade dos dados processados.",
            "Desenvolvido com cuidado para apoiar a análise dos dados do doutorado de Henrique "
            "Leal, com uso gratuito para essa finalidade acadêmica e científica.",
        ]

        # wraplength inicial evita o "balão": sem ele, cada parágrafo pede a largura
        # do texto numa linha só (enorme) e a janela oscila tentando caber.
        W0 = 820
        lbls = []
        lbls.append(ttkb.Label(body, text=title, font=("Segoe UI Semibold", 15),
                               bootstyle="primary", justify="left", wraplength=W0))
        lbls[-1].pack(anchor="w", pady=(0, 10))
        for p in paras:
            l = ttkb.Label(body, text=p, font=("Segoe UI", 11), justify="left", wraplength=W0)
            l.pack(anchor="w", pady=(0, 10))
            lbls.append(l)
        foot = ttkb.Label(body, text="Edilson Borba — borba.edi@gmail.com",
                          font=("Segoe UI", 11), bootstyle="secondary", justify="left", wraplength=W0)
        foot.pack(anchor="w", pady=(4, 0)); lbls.append(foot)

        # Responsivo: a quebra acompanha a largura, mas só reaplica quando a largura
        # MUDA de fato (evita o loop de realimentação Configure→wraplength→Configure).
        _st = {"w": W0}
        def _rewrap(event):
            w = max(event.width - 72, 240)
            if abs(w - _st["w"]) < 8:
                return
            _st["w"] = w
            for l in lbls:
                l.configure(wraplength=w)
        body.bind("<Configure>", _rewrap)


def _selftest(report_path):
    """Verificação do app empacotado: sem GUI, escreve um relatório e sai.
    Uso: AM_SELFTEST=<arquivo> AnaliseMarcha.exe"""
    import traceback
    lines = []
    try:
        lines.append("data_dir=" + data_dir())
        import ttkbootstrap, PIL, openpyxl  # noqa: F401
        lines.append("ttkbootstrap/PIL/openpyxl=OK")
        import analise_marcha as eng  # noqa
        lines.append("engine=OK")
        import onnxruntime as ort
        lines.append("onnxruntime=" + ort.__version__)
        lines.append("providers=" + ",".join(ort.get_available_providers()))
        from rtmlib import BodyWithFeet  # noqa: F401
        lines.append("rtmlib=OK")
        lines.append("SELFTEST_OK")
    except Exception:
        lines.append("SELFTEST_FAIL")
        lines.append(traceback.format_exc())
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    root = ttkb.Window(themename=THEME)
    # 1ª execução (app empacotado): se houver GPU NVIDIA, oferece baixar o runtime
    # CUDA (do PyPI). Sem GPU ou recusado, roda em CPU. Em dev/sem o módulo, ignora.
    try:
        from gpu_bootstrap import ensure_gpu_ready
        ensure_gpu_ready(ask=True)
        engine._enable_cuda_dlls()   # re-registra caso o CUDA acabou de ser baixado
    except Exception:
        pass
    MarchaApp(root)
    root.mainloop()


if __name__ == "__main__":
    _report = os.environ.get("AM_SELFTEST")
    if _report:
        _selftest(_report)
    else:
        main()
