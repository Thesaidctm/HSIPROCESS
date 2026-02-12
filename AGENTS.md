# AGENTS.md

## Overview
Este repositório é um MVP em **Streamlit** para:
- abrir cubos hiperespectrais ENVI (`.hdr` + `.raw/.dat`),
- visualizar pseudo-RGB,
- inspecionar espectro por pixel,
- exportar dados em CSV.

Objetivo do agente: manter foco no MVP, com mudanças pequenas, claras e fáceis de manter.

## Dev Commands
Use os comandos abaixo para validar execução local:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Comando único equivalente:

```bash
pip install -r requirements.txt && streamlit run app.py
```

## Code Style
- **Não adicionar features fora do solicitado.**
- Manter a UI simples em Streamlit (sem complexidade desnecessária).
- Quando fizer sentido, separar lógica de processamento em `src/` e deixar `app.py` mais enxuto.
- Adicionar docstrings e comentários curtos para facilitar manutenção.
- Se adicionar dependências, atualizar `requirements.txt`.
- Sempre atualizar `README.md` quando novas funcionalidades forem adicionadas.

## Review Checklist
Antes de concluir uma alteração, validar:
- Escopo respeitado (sem feature creep).
- UI continua simples e funcional no Streamlit.
- Lógica de processamento extraída para `src/` quando aplicável.
- Docstrings/comentários curtos adicionados em trechos novos ou alterados.
- `requirements.txt` atualizado se houve nova dependência.
- `README.md` atualizado se houve nova funcionalidade.
- App executa com:
  - `pip install -r requirements.txt`
  - `streamlit run app.py`
  - ou `pip install -r requirements.txt && streamlit run app.py`
