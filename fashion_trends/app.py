import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image

# =====================
# PATHS (robustos p/ Streamlit Cloud)
# =====================
BASE_DIR = Path(__file__).resolve().parent  # pasta do app.py (fashion_trends/)
SAVEDMODEL_DIR = BASE_DIR / "model_saved"  # pasta com saved_model.pb e variables/
LABELS_PATH = BASE_DIR / "model" / "labels_runtime.json"
SPLITS_CSV = BASE_DIR / "data" / "splits.csv"

# =====================
# MAPA subclasse -> macro
# =====================
STYLE_TO_MACRO = {
    "casual": "CASUAL_BASICO",
    "comfortable": "CASUAL_BASICO",
    "basic": "CASUAL_BASICO",
    "classic": "CASUAL_BASICO",
    "denim": "CASUAL_BASICO",

    "chic": "CHIC_ROMANTIC",
    "romantic": "CHIC_ROMANTIC",
    "elegant": "CHIC_ROMANTIC",
    "preppy": "CHIC_ROMANTIC",

    "trendy": "TRENDY_ECLECTIC",
    "eclectic": "TRENDY_ECLECTIC",
    "urban": "TRENDY_ECLECTIC",

    "bohemian": "EXPRESSIVE",
    "rocker": "EXPRESSIVE",
    "sexy": "EXPRESSIVE",
}

# =====================
# TRADUÇÕES (UI)
# =====================
MACRO_PT = {
    "CASUAL_BASICO": "Casual / Básico",
    "CHIC_ROMANTIC": "Chique / Romântico",
    "TRENDY_ECLECTIC": "Moderno / Eclético",
    "EXPRESSIVE": "Expressivo / Autoral",
}

STYLE_PT = {
    "casual": "Casual",
    "basic": "Básico",
    "comfortable": "Confortável",
    "denim": "Jeans",
    "classic": "Clássico",

    "chic": "Chique",
    "romantic": "Romântico",
    "elegant": "Elegante",
    "preppy": "Social Jovem",

    "urban": "Urbano",
    "trendy": "Tendência",
    "eclectic": "Eclético",

    "bohemian": "Boho",
    "rocker": "Rocker",
    "sexy": "Sensual",
}

# traduções simples de tags comuns do dataset
TAG_PT = {
    "jeans": "jeans",
    "denim": "jeans",
    "boots": "botas",
    "sneakers": "tênis",
    "blazer": "blazer",
    "dress": "vestido",
    "jacket": "jaqueta",
    "coat": "casaco",
    "skirt": "saia",
    "shirt": "camisa",
    "tshirt": "camiseta",
    "t-shirt": "camiseta",
    "heels": "salto",
    "bag": "bolsa",
    "scarf": "cachecol",
}

def tr_tag(t: str) -> str:
    t = (t or "").strip().lower()
    return TAG_PT.get(t, t)

# =====================
# THEME
# =====================
def inject_dark_blue_theme():
    st.markdown("""
    <style>
    .stApp {
      background: linear-gradient(180deg, #0b1e33 0%, #081526 60%, #050b14 100%);
      color: #e6f0ff;
      font-family: 'Inter', sans-serif;
    }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    h1 { color: #e6f0ff; font-weight: 800; letter-spacing: 0.5px; }
    h2, h3 { color: #c7ddff; font-weight: 650; }
    p, li, span { color: #d6e6ff; font-size: 0.95rem; }

    [data-testid="stAlert"],
    section[data-testid="stFileUploader"] {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(100,170,255,0.18);
      border-radius: 18px;
      padding: 14px;
      margin-bottom: 1rem;
    }

    [data-testid="stFileUploader"] {
      background: rgba(255,255,255,0.06);
      border: 1px dashed rgba(120,180,255,0.35);
      border-radius: 20px;
    }

    .stButton > button {
      background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%);
      color: white;
      border-radius: 16px;
      border: none;
      padding: 0.6rem 1.2rem;
      font-weight: 700;
      letter-spacing: 0.3px;
      box-shadow: 0 6px 20px rgba(14,165,233,0.25);
    }
    .stButton > button:hover { filter: brightness(1.1); transform: translateY(-1px); }

    img {
      border-radius: 18px;
      border: 1px solid rgba(120,180,255,0.25);
      box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }

    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# =====================
# LOADERS
# =====================
@st.cache_resource
def load_infer_fn():
    """
    Carrega um SavedModel (mais robusto para deploy).
    Retorna a assinatura de inferência "serving_default".
    """
    sm = tf.saved_model.load(str(SAVEDMODEL_DIR))
    if "serving_default" not in sm.signatures:
        # fallback: pega a primeira assinatura existente
        first_key = list(sm.signatures.keys())[0]
        return sm.signatures[first_key]
    return sm.signatures["serving_default"]

@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_tag_stats():
    """
    Tags típicas por macro e por style, a partir do splits.csv.
    Isso serve como 'explicação semântica' sem Grad-CAM.
    """
    df = pd.read_csv(SPLITS_CSV)

    if "tags_clean" not in df.columns:
        df["tags_clean"] = ""

    df["tags_clean"] = df["tags_clean"].fillna("").astype(str)

    def split_tags(s: str):
        s = (s or "").strip().lower()
        if not s or s == "null":
            return []
        return [t.strip() for t in s.split(",") if t.strip()]

    df["tags_list"] = df["tags_clean"].apply(split_tags)

    macro_tags = {}
    style_tags = {}

    for macro, g in df.groupby("macro"):
        c = {}
        for tags in g["tags_list"]:
            for t in tags:
                c[t] = c.get(t, 0) + 1
        macro_tags[macro] = sorted(c.items(), key=lambda x: x[1], reverse=True)

    for style, g in df.groupby("style"):
        c = {}
        for tags in g["tags_list"]:
            for t in tags:
                c[t] = c.get(t, 0) + 1
        style_tags[style] = sorted(c.items(), key=lambda x: x[1], reverse=True)

    return macro_tags, style_tags

# =====================
# PREPROCESS (OpenCV)
# =====================
def preprocess_image(pil_img, img_size: int):
    """
    Mantém visão global do outfit:
    - crop leve (não corta demais)
    - resize
    - leve ajuste de contraste/brilho
    Retorna:
      img_f: float32 0..255 para o modelo
      img_u8: uint8 para exibir
    """
    rgb = np.array(pil_img.convert("RGB"))
    h, w = rgb.shape[:2]

    y1, y2 = int(0.05 * h), int(0.95 * h)
    x1, x2 = int(0.10 * w), int(0.90 * w)

    rgb = rgb[y1:y2, x1:x2]
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # ajuste leve
    rgb = cv2.convertScaleAbs(rgb, alpha=1.05, beta=2)

    return rgb.astype(np.float32), rgb.astype(np.uint8)

# =====================
# PREDICTION (SavedModel)
# =====================
def predict(infer_fn, img_0_255):
    """
    infer_fn: assinatura do SavedModel
    img_0_255: float32 [H,W,3] 0..255
    Retorna: (macro_probs, style_probs)
    """
    x = np.expand_dims(img_0_255, 0).astype(np.float32)
    out = infer_fn(tf.convert_to_tensor(x))

    # SavedModel geralmente retorna dict {output_name: tensor}
    if isinstance(out, dict):
        keys = list(out.keys())
        vals = [out[k] for k in keys]
    else:
        # caso raro
        vals = list(out)

    # Esperamos duas saídas (macro e style)
    if len(vals) < 2:
        raise RuntimeError(
            f"Saída inesperada do SavedModel. "
            f"Esperava 2 tensores (macro, style), recebi {len(vals)}."
        )

    macro_p = vals[0].numpy()[0]
    style_p = vals[1].numpy()[0]
    return macro_p, style_p

def tr_macro(name: str) -> str:
    return MACRO_PT.get(name, name)

def tr_style(name: str) -> str:
    return STYLE_PT.get(name, name)

# =====================
# APP
# =====================
st.set_page_config(page_title="Outfit Classifier", layout="centered")
inject_dark_blue_theme()

st.title("👕 Classificador de Outfit (Macro + Subclasse)")
st.caption("Upload de uma foto de outfit completo. O modelo retorna Estilo Principal (Macro) + Subestilo (Subclasse) e mostra evidências semânticas (tags do dataset).")

# Checagens básicas
if not SAVEDMODEL_DIR.exists():
    st.error(f"SavedModel não encontrado em: {SAVEDMODEL_DIR}")
    st.info("Você precisa subir a pasta model_saved/ (saved_model.pb + variables/) no repo.")
    st.stop()

if not (SAVEDMODEL_DIR / "saved_model.pb").exists():
    st.error(f"Arquivo saved_model.pb não encontrado em: {SAVEDMODEL_DIR / 'saved_model.pb'}")
    st.stop()

if not LABELS_PATH.exists():
    st.error(f"Arquivo de labels não encontrado em: {LABELS_PATH}")
    st.stop()

if not SPLITS_CSV.exists():
    st.warning(f"Não achei {SPLITS_CSV}. Vou rodar sem as tags típicas (explicação semântica).")

infer_fn = load_infer_fn()
labels = load_labels()

IMG_SIZE = int(labels.get("img_size", 192))
MACROS = labels["macro_names"]
STYLES = labels["style_names"]

macro_tags, style_tags = ({}, {})
if SPLITS_CSV.exists():
    macro_tags, style_tags = load_tag_stats()

file = st.file_uploader("📸 Envie uma imagem", type=["jpg", "jpeg", "png"])

if file:
    pil = Image.open(file)

    st.subheader("Imagem original")
    st.image(pil, width="stretch")

    img_f, img_u8 = preprocess_image(pil, IMG_SIZE)

    st.subheader("Imagem analisada (preprocess OpenCV)")
    st.image(img_u8, width="stretch")

    macro_p, style_p = predict(infer_fn, img_f)

    # Macro
    macro_id = int(np.argmax(macro_p))
    macro_name = MACROS[macro_id]
    macro_conf = float(macro_p[macro_id])

    # Subclasse coerente (restringe à macro prevista)
    style_mask = np.array([1.0 if STYLE_TO_MACRO.get(s) == macro_name else 0.0 for s in STYLES], dtype=np.float32)
    masked = style_p * style_mask
    style_id = int(np.argmax(masked)) if masked.sum() > 0 else int(np.argmax(style_p))
    style_name = STYLES[style_id]
    style_conf = float(style_p[style_id])

    # Exibição traduzida
    macro_pt = tr_macro(macro_name)
    style_pt = tr_style(style_name)

    st.success(f"🧠 Estilo principal: **{macro_pt}**  —  **{macro_conf:.2%}**")
    st.info(f"🏷️ Subestilo: **{style_pt}**  —  **{style_conf:.2%}**")

    # Top-3 macro
    st.subheader("Top-3 Estilos Principais (Macro)")
    topm = np.argsort(macro_p)[::-1][:3]
    for i in topm:
        st.write(f"- {tr_macro(MACROS[i])}: {float(macro_p[i]):.2%}")

    # Top-5 styles dentro da macro
    st.subheader("Top-5 Subestilos (dentro do estilo principal previsto)")
    candidates = [(i, float(masked[i])) for i in range(len(STYLES)) if style_mask[i] > 0]
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]
    if candidates:
        for i, p in candidates:
            st.write(f"- {tr_style(STYLES[i])}: {p:.2%}")
    else:
        st.write("(Sem subclasses mapeadas para essa macro no STYLE_TO_MACRO)")

    # Evidência semântica (tags típicas)
    st.subheader("🧩 Evidências semânticas (tags típicas do dataset)")
    if macro_tags:
        mt = macro_tags.get(macro_name, [])
        st.write(f"**Macro ({macro_pt}) — tags mais comuns:**")
        if mt:
            tags = [tr_tag(t) for t, _ in mt[:12]]
            st.write(", ".join(tags))
        else:
            st.write("(sem tags suficientes)")

    if style_tags:
        stt = style_tags.get(style_name, [])
        st.write(f"**Subclasse ({style_pt}) — tags mais comuns:**")
        if stt:
            tags = [tr_tag(t) for t, _ in stt[:12]]
            st.write(", ".join(tags))
        else:
            st.write("(sem tags suficientes)")

else:
    st.info("👆 Envie uma imagem para começar.")
