# Análise de Marcha

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21221755.svg)](https://doi.org/10.5281/zenodo.21221755)

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

## Como usar

O app é um painel com barra lateral: **Início · Participantes · Analisar · Resultados ·
Config · Sobre**. O fluxo de uma análise é:

### 1. Cadastrar o participante
Na aba **Participantes**, clique em **Novo participante** e preencha os dados usados para
escalar a análise: nome, estatura, **massa (kg)**, **comprimento do membro inferior (GT, cm)**
e, opcionalmente, **coxa (cm)**. Clique em **Salvar**.

![Cadastro de participante](docs/img/01-participantes.png)

### 2. Configurar a análise
Na aba **Analisar**, selecione o **participante** (massa e comprimentos são preenchidos
automaticamente), escolha o **vídeo** da caminhada, a **pasta de saída** e, se quiser, recorte
por tempo em **Início (s)** / **Fim (s)**. Clique em **Continuar → marcar cones**.

![Tela de análise](docs/img/02-analisar.png)

### 3. Calibrar a escala pelos cones
Sobre o primeiro quadro do vídeo, marque a **BASE** de cada cone da esquerda para a direita
(fase 1) e depois o **TOPO** de cada um, na mesma ordem (fase 2). Isso dá a escala real
(metros/pixel). Se o vídeo não tiver cones, use **Pular** — a escala é estimada pelo
comprimento da coxa.

![Calibração por cones](docs/img/03-cones.png)

### 4. Revisar e cortar a faixa
O app mostra o vídeo com o esqueleto sobreposto. Ajuste os controles de **Início/Fim** para
manter só o trecho em que a detecção está boa e clique em **Confirmar faixa e analisar**.

![Revisão e corte](docs/img/04-corte.png)

### 5. Ver os resultados
Ao final, a tela **Resultados** mostra as métricas (velocidade, cadência, comprimento de
passo/passada, trabalho mecânico, recuperação pendular, IRL) e os gráficos (mecânico e
angular). As planilhas (`.xlsx`) e os gráficos (`.png`) são salvos na pasta de saída, dentro
de `Resultados_RTMLib/`. Análises anteriores ficam acessíveis na aba **Resultados**.

![Resultados](docs/img/05-resultados.png)

---

## O que é salvo

Cada análise grava os resultados na **pasta de saída** que você escolheu, dentro de uma
subpasta **`Resultados_RTMLib/`**. Todas as planilhas trazem uma coluna `Paciente` para
facilitar juntar vários sujeitos. Os arquivos por trial recebem o prefixo do nome
(`{participante}_{video}`).

### Estrutura das pastas

```
<pasta de saída>/Resultados_RTMLib/
├── Resumo_Geral.xlsx            (acumulativo — todos os trials juntos)
├── Marcadores/
├── Angulares_Gerais/
├── Angulares_Interpolados/
├── Angulares_Interpolados_Passo/
├── Espaco_Temporais/
├── Mecanica/
├── Eventos/
├── Qualidade/
└── Graficos/
```

### O que cada arquivo contém

| Pasta / arquivo | Conteúdo |
|---|---|
| **`Resumo_Geral.xlsx`** (raiz) | Planilha **acumulativa** entre análises. Abas: `Resumo` (uma linha por trial), `Medias_SD` (média ± desvio-padrão por passo) e `Por_Passo` (tabela mestre: uma linha por passo com espaço-temporais + mecânica + angular). É o arquivo principal para comparar sujeitos/condições. |
| **`Marcadores/`** | Coordenadas dos marcadores corporais quadro a quadro, em **pixels** (`_x_px`, `_y_px`), em **metros** (`_x_m`, `_y_m`) e a visibilidade (`_vis`) de cada ponto. |
| **`Angulares_Gerais/`** | Ângulos articulares filtrados quadro a quadro (quadril, joelho, tornozelo, pé e tronco), aba `Dados_Filtrados`. |
| **`Angulares_Interpolados/`** | Curvas angulares normalizadas em **0–100% do ciclo** da marcha, uma coluna por passada. |
| **`Angulares_Interpolados_Passo/`** | Idem, mas normalizado por **passo** (contato → contato contralateral). |
| **`Espaco_Temporais/`** | Parâmetros espaço-temporais. Abas: `Passos`, `Passadas`, `Medias_Passos`, `Medias_Passadas`, `Angular_Por_Passada`, `Angular_Por_Passo` (velocidade, cadência, comprimentos, tempos, ROM). |
| **`Mecanica/`** | Trabalho mecânico do centro de massa. Abas: `Resumo` (Wext, Wv, Wf, Recovery, IRL, LRI, velocidade média), `Series_Temporais` (energias Ep/Ek/Emec, posição e velocidade do COM quadro a quadro), `Mecanica_Por_Passo` e `Series_Interp_Passo`. |
| **`Eventos/`** | Eventos do ciclo por lado: contato inicial (`TD`) e retirada do pé (`TO`), em **frame** e em **tempo (s)**. |
| **`Qualidade/`** | Relatório de qualidade da detecção. Abas: `Landmarks` (quão bem cada ponto foi detectado) e `Eventos`. Útil para checar a confiabilidade do trial. |
| **`Graficos/`** | Dois PNGs por trial: `{nome}_Sinais_Mecanica.png` (energias/velocidade do COM) e `{nome}_Angulos_Bilaterais.png` (ângulos dos dois lados). São os mesmos gráficos exibidos na tela de Resultados. |

> **Dados do participante e do app** (cadastro de sujeitos, configurações e o registro de
> análises) ficam em `%LOCALAPPDATA%\AnaliseMarcha` — separados da pasta de saída dos
> resultados.

---

## Rodar a partir do código-fonte

Requisitos: **Python 3.11** (Windows).

```bash
# 1. Clonar o repositório
git clone https://github.com/EdilsonBorba/analise_marcha_app.git
cd analise_marcha_app

# 2. Criar e ativar um ambiente virtual
python -m venv .venv
.venv\Scripts\activate

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Rodar o painel
python app_marcha.py
```

### GPU (opcional, recomendado)
Com uma GPU NVIDIA, o `gpu_bootstrap.py` baixa o runtime CUDA 12.9 + cuDNN 9 na primeira
execução e habilita o ONNX Runtime na GPU automaticamente. Sem GPU, o programa roda em CPU.

---

## Gerar o executável / instalador

```bash
# Instalar as dependências de build
pip install -r requirements.txt

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
