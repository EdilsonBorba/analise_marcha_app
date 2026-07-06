# Análise de Marcha

Ferramenta de **análise de marcha por vídeo** (biomecânica). A partir de um vídeo do
sujeito caminhando, o programa estima marcadores corporais, ângulos articulares, eventos
do ciclo da marcha (contato/saída dos pés), parâmetros **espaço-temporais** (velocidade,
cadência, comprimento de passo/passada) e o **trabalho mecânico** do centro de massa
(recuperação pendular, IRL).

A estimativa de pose 2D usa **RTMPose (Halpe-26)** via **ONNX Runtime**, com aceleração em
GPU NVIDIA quando disponível (funciona também em CPU, mais lento).

O projeto tem duas partes:
- **`app_marcha.py`** — painel gráfico (dashboard) para uso clínico/pesquisa.
- **`analise_marcha.py`** — o *motor* de análise, reutilizável como biblioteca.

---

## Baixar o aplicativo (Windows, sem instalar Python)

Se você só quer **usar** o programa, baixe a versão pronta na página de
**[Releases](../../releases)** deste repositório:

- **`AnaliseMarcha_Setup.exe`** — instalador (recomendado). Instala como um programa normal,
  cria atalhos no Menu Iniciar e na Área de Trabalho.
- **`AnaliseMarcha.exe`** — executável único (portátil), se preferir não instalar.

> Na **primeira execução**, em máquinas com GPU NVIDIA, o programa baixa uma vez os
> componentes de GPU/modelos (precisa de internet). Depois funciona offline.

---

## Rodar a partir do código-fonte

Requisitos: **Python 3.11** (Windows).

```bash
# 1. Clonar o repositório
git clone https://github.com/<seu-usuario>/analise_marcha_app.git
cd analise_marcha_app

# 2. Criar e ativar um ambiente virtual
python -m venv .venv
.venv\Scripts\activate

# 3. Instalar as dependências do aplicativo (backend rtmlib, mais leve)
pip install -r requirements-app.txt

# 4. Rodar o painel
python app_marcha.py
```

Para a instalação **completa** (inclui os backends legados de comparação — torch/mmpose/
mediapipe), use `requirements.txt` no lugar do `requirements-app.txt`. Ela é bem mais pesada
e só é necessária para os scripts experimentais
`analise_marcha_grama_backend_compare*.py`.

### GPU (opcional, recomendado)
Com uma GPU NVIDIA, o `gpu_bootstrap.py` baixa o runtime CUDA 12.9 + cuDNN 9 na primeira
execução e habilita o ONNX Runtime na GPU automaticamente. Sem GPU, o programa roda em CPU.

---

## Gerar o executável / instalador

```bash
# Instalar as dependências de build
pip install -r requirements-app.txt

# Opção A — pasta (onedir): dist/AnaliseMarcha/
pyinstaller app_marcha.spec

# Opção B — arquivo único (onefile): dist/AnaliseMarcha.exe
pyinstaller app_marcha_onefile.spec
```

Para gerar o **instalador** `Output/AnaliseMarcha_Setup.exe`, é preciso o
[Inno Setup 6](https://jrsoftware.org/isinfo.php) e a pasta `dist/AnaliseMarcha/` já gerada:

```bash
ISCC.exe installer.iss
```

---

## Privacidade dos dados

Este repositório contém **apenas código**. Nenhum vídeo ou dado de participante é
distribuído aqui. Ao usar o programa, os dados dos sujeitos (nome, biometria, resultados)
ficam **somente na máquina do usuário**, em `%LOCALAPPDATA%\AnaliseMarcha`. Trate esses
dados conforme a LGPD e a aprovação ética do seu estudo.

---

## Licença e citação

Distribuído sob a licença **MIT** (veja [`LICENSE`](LICENSE)).

Se usar este software em trabalhos acadêmicos, por favor cite-o — veja
[`CITATION.cff`](CITATION.cff). Após o arquivamento no Zenodo, o DOI aparecerá aqui.

**Autor:** Edilson Borba — borba.edi@gmail.com
