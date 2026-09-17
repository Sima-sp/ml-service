# Atalhos do projeto. Uso: make <alvo>   (make ajuda lista tudo)
PY ?= python
INICIO ?= 2017-01-01
FIM ?= 2025-12-31
VERSAO ?= v1

.PHONY: ajuda ambiente testar lint checar-fontes sintetico dataset treinar servir docker demo

ajuda:
	@echo "ambiente       cria .venv e instala requirements.txt"
	@echo "testar         roda a suíte pytest"
	@echo "lint           ruff"
	@echo "checar-fontes  testa se as fontes públicas respondem nesta rede"
	@echo "sintetico      pipeline completo com dados SINTÉTICOS (teste de fumaça)"
	@echo "dataset        monta data/processed/dataset_\$$(VERSAO).parquet com os dados reais"
	@echo "treinar        treina/compara modelos e salva models/*_\$$(VERSAO).*"
	@echo "servir         sobe a API em http://localhost:5000 (docs em /docs)"
	@echo "docker         build + run do container"
	@echo "demo           envia os cenários de demonstração para a API"

ambiente:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

testar:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src app tests scripts

checar-fontes:
	$(PY) scripts/checar_fontes.py

sintetico:
	$(PY) -m src.dataset --versao sintetico --sintetico
	$(PY) -m src.train --dataset sintetico --versao sintetico

dataset:
	$(PY) -m src.dataset --versao $(VERSAO) --inicio $(INICIO) --fim $(FIM)

treinar:
	$(PY) -m src.train --dataset $(VERSAO) --versao $(VERSAO)

servir:
	VERSAO_MODELO=$(VERSAO) $(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload

docker:
	docker compose up --build

demo:
	$(PY) scripts/demo_cenarios.py --url http://localhost:5000
