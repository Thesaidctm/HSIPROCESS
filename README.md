# ENVI HSI Streamlit GUI

Interface simples (MVP) para abrir cubos hiperespectrais **ENVI** (`.hdr` + `.raw/.dat`), visualizar **pseudo-RGB**, inspecionar **espectro de pixel** e exportar resultados em **CSV**.

## Requisitos
- Python 3.10+ (recomendado)
- Pacotes em `requirements.txt`

## Instalação
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Rodar
```bash
streamlit run app.py
```

## Observações importantes (ENVI)
- Faça upload do **`.hdr`** no app.
- O arquivo binário correspondente (`.raw` / `.dat`) precisa estar **na mesma pasta** e com o nome que o header espera.
- O app usa `open_memmap()` (memmap) para reduzir uso de RAM.

## Próximos upgrades (ideias)
- Clique direto na imagem para pegar (x,y)
- ROI desenhada (polígono/brush) e máscaras
- Pré-processamentos (SNV/MSC/SG/derivadas)
- PCA/PLS e mapas por pixel
