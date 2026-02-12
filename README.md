# ENVI HSI Streamlit GUI

Interface simples (MVP) para abrir cubos hiperespectrais **ENVI** (`.hdr` + `.raw/.dat`), visualizar **pseudo-RGB**, inspecionar **espectro de pixel** e exportar resultados em **CSV**.

## Selections (ROIs)
- A barra lateral possui a seção **Selections** para criar e gerenciar múltiplas ROIs retangulares.
- Cada seleção armazena `id`, `nome` (`capture.001`, `capture.002`, ...), `cor` (hex), `tipo` (`rect`) e coordenadas (`x0`, `y0`, `x1`, `y1`).
- É possível selecionar ROI ativa, renomear, deletar e alternar visibilidade (ícone de olho).
- O estado das seleções é persistido em `st.session_state` durante a sessão do app.
- A ROI ativa também pode ser desenhada diretamente na imagem com um retângulo interativo por dois cliques na imagem.
- A imagem pseudo-RGB mostra contorno colorido de cada ROI visível e um pequeno "carimbo" preenchido com a cor da seleção.

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

## Execução rápida
```bash
pip install -r requirements.txt && streamlit run app.py
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
