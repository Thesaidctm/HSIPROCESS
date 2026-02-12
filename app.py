import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from spectral import open_image

st.set_page_config(page_title="ENVI HSI Viewer", layout="wide")

@st.cache_data(show_spinner=False)
def load_envi(hdr_path: str):
    """
    Load an ENVI cube from a .hdr file using Spectral Python (SPy).
    Returns:
        cube_memmap: numpy memmap with shape (rows, cols, bands)
        meta: dict metadata from header
        wls: np.ndarray of wavelengths (float) or None
    """
    img = open_image(hdr_path)
    cube_memmap = img.open_memmap()  # memmap: avoids loading the entire cube at once
    meta = img.metadata

    wls = None
    if "wavelength" in meta:
        try:
            wls = np.array(meta["wavelength"], dtype=float)
        except Exception:
            # sometimes it comes as strings; try casting element-wise
            try:
                wls = np.array([float(x) for x in meta["wavelength"]], dtype=float)
            except Exception:
                wls = None

    return cube_memmap, meta, wls

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def make_rgb(cube: np.ndarray, r: int, g: int, b: int, clip_percent: float = 1.0) -> np.ndarray:
    """
    Build a pseudo-RGB image from cube bands (r,g,b), using robust percentile clipping.
    """
    R = cube[:, :, r].astype(np.float32)
    G = cube[:, :, g].astype(np.float32)
    B = cube[:, :, b].astype(np.float32)

    def robust_scale(ch: np.ndarray) -> np.ndarray:
        if clip_percent > 0:
            lo = np.nanpercentile(ch, clip_percent)
            hi = np.nanpercentile(ch, 100 - clip_percent)
            ch = np.clip(ch, lo, hi)
        return normalize01(ch)

    rgb = np.dstack([robust_scale(R), robust_scale(G), robust_scale(B)])
    return rgb

st.title("Interface para Cubos Hiperespectrais ENVI (HDR/RAW)")

with st.sidebar:
    st.header("1) Importação")
    hdr_file = st.file_uploader("Selecione o arquivo .hdr", type=["hdr"])
    st.caption("Deixe o .raw/.dat na mesma pasta do .hdr (com o nome esperado pelo header).")

    st.header("2) Visualização")
    clip_p = st.slider("Contraste (clipping %)", 0.0, 5.0, 1.0, 0.1)

    st.header("3) Exportação")
    export_prefix = st.text_input("Prefixo para exportar CSV", value="resultado")

if hdr_file is None:
    st.info("Faça upload do seu .hdr na barra lateral para começar.")
    st.stop()

# Save .hdr to disk so SPy can resolve the binary file path referenced by the header.
tmp_dir = os.path.join(os.getcwd(), ".streamlit_tmp")
os.makedirs(tmp_dir, exist_ok=True)

hdr_path = os.path.join(tmp_dir, hdr_file.name)
with open(hdr_path, "wb") as f:
    f.write(hdr_file.getbuffer())

try:
    cube, meta, wls = load_envi(hdr_path)
except Exception as e:
    st.error("Não consegui abrir o ENVI. Verifique se o .raw/.dat está acessível e coerente com o .hdr.")
    st.exception(e)
    st.stop()

rows, cols, bands = cube.shape
st.success(f"Cube carregado: {rows} x {cols} x {bands} (linhas x colunas x bandas)")

colA, colB = st.columns([1.2, 1.0], gap="large")

with colB:
    st.subheader("Metadados")
    keys = ["lines", "samples", "bands", "interleave", "data type", "byte order"]
    st.write({k: meta.get(k) for k in keys if k in meta})
    if wls is not None and len(wls) == bands:
        st.write(f"Wavelengths: {len(wls)} valores (ex.: {wls[0]:.1f} … {wls[-1]:.1f} nm)")
    else:
        st.warning("Não encontrei 'wavelength' coerente no header. Vou trabalhar por índice de banda.")

with colA:
    st.subheader("Pseudo-RGB")
    if wls is not None and len(wls) == bands:
        wl_min, wl_max = float(np.min(wls)), float(np.max(wls))
        r_wl = st.slider("R (nm)", wl_min, wl_max, float(wls[int(0.75 * (bands-1))]))
        g_wl = st.slider("G (nm)", wl_min, wl_max, float(wls[int(0.50 * (bands-1))]))
        b_wl = st.slider("B (nm)", wl_min, wl_max, float(wls[int(0.25 * (bands-1))]))

        r = int(np.argmin(np.abs(wls - r_wl)))
        g = int(np.argmin(np.abs(wls - g_wl)))
        b = int(np.argmin(np.abs(wls - b_wl)))

        st.caption(f"Bandas: R={r} (~{wls[r]:.1f} nm), G={g} (~{wls[g]:.1f} nm), B={b} (~{wls[b]:.1f} nm)")
    else:
        r = st.slider("Banda R (índice)", 0, bands-1, int(0.75 * (bands-1)))
        g = st.slider("Banda G (índice)", 0, bands-1, int(0.50 * (bands-1)))
        b = st.slider("Banda B (índice)", 0, bands-1, int(0.25 * (bands-1)))

    rgb = make_rgb(cube, r, g, b, clip_percent=clip_p)
    st.image(rgb, caption="Pseudo-RGB (normalizado)")

    st.markdown("### Inspeção de pixel")
    c1, c2, _ = st.columns([1, 1, 1])
    with c1:
        x = st.number_input("Coluna (x)", min_value=0, max_value=cols-1, value=int(cols/2), step=1)
    with c2:
        y = st.number_input("Linha (y)", min_value=0, max_value=rows-1, value=int(rows/2), step=1)

    spectrum = np.array(cube[int(y), int(x), :], dtype=np.float32)

    fig = plt.figure()
    if wls is not None and len(wls) == bands:
        plt.plot(wls, spectrum)
        plt.xlabel("Wavelength (nm)")
    else:
        plt.plot(np.arange(bands), spectrum)
        plt.xlabel("Índice da banda")
    plt.ylabel("Intensidade (a.u.)")
    plt.title(f"Espectro do pixel (x={int(x)}, y={int(y)})")
    st.pyplot(fig, clear_figure=True)

    if st.button("Exportar espectro do pixel (CSV)"):
        df = pd.DataFrame({
            "band": np.arange(bands),
            "wavelength_nm": wls if (wls is not None and len(wls) == bands) else np.nan,
            "value": spectrum
        })
        out_path = os.path.join(tmp_dir, f"{export_prefix}_pixel_x{int(x)}_y{int(y)}.csv")
        df.to_csv(out_path, index=False)
        st.success(f"Exportado: {out_path}")

st.divider()

st.subheader("ROI simples (retângulo) e espectro médio")
st.caption("MVP: ROI retangular. Depois você pode evoluir para ROI desenhada (polígono/brush) e máscaras.")

cR1, cR2, cR3, cR4 = st.columns(4)
with cR1:
    x0 = st.number_input("x0", 0, cols-1, 0, 1)
with cR2:
    x1 = st.number_input("x1", 0, cols-1, min(cols-1, 50), 1)
with cR3:
    y0 = st.number_input("y0", 0, rows-1, 0, 1)
with cR4:
    y1 = st.number_input("y1", 0, rows-1, min(rows-1, 50), 1)

x0, x1 = int(min(x0, x1)), int(max(x0, x1))
y0, y1 = int(min(y0, y1)), int(max(y0, y1))

roi = cube[y0:y1+1, x0:x1+1, :].astype(np.float32)
roi_mean = np.nanmean(roi.reshape(-1, bands), axis=0)

fig2 = plt.figure()
if wls is not None and len(wls) == bands:
    plt.plot(wls, roi_mean)
    plt.xlabel("Wavelength (nm)")
else:
    plt.plot(np.arange(bands), roi_mean)
    plt.xlabel("Índice da banda")
plt.ylabel("Intensidade (a.u.)")
plt.title(f"Espectro médio ROI: x[{x0}:{x1}], y[{y0}:{y1}]")
st.pyplot(fig2, clear_figure=True)

if st.button("Exportar espectro médio da ROI (CSV)"):
    df = pd.DataFrame({
        "band": np.arange(bands),
        "wavelength_nm": wls if (wls is not None and len(wls) == bands) else np.nan,
        "mean_value": roi_mean
    })
    out_path = os.path.join(tmp_dir, f"{export_prefix}_roi_x{x0}-{x1}_y{y0}-{y1}.csv")
    df.to_csv(out_path, index=False)
    st.success(f"Exportado: {out_path}")
