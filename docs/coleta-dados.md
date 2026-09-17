# Roteiro da coleta de dados

> **Onde rodar:** na máquina de desenvolvimento, dentro da pasta `ml-service` com o ambiente
> virtual ativo. As fontes foram verificadas em 16/09/2026.

## Portão de decisão — respondido em 16/09/2026

| Pergunta (briefing §13) | Resposta com dados | Decisão |
|---|---|---|
| O CGE tem hora de início e dá para raspar vários anos? | **Sim.** `GET alagamentos.jsp?dataBusca=DD/MM/AAAA&enviaBusca=Buscar`, mesma estrutura HTML de 2011 a 2026 (ex.: 10/02/2016 = 8 pontos, 11/03/2019 = 58, 24/01/2011 = 7). O total "N pts." da página bate com os pontos extraídos. | CGE é o **rótulo principal** |
| O GeoSampa serve de rótulo complementar? | **Não pelo WFS.** `risco_ocorrencia_alagamento` tem 197 registros e `risco_ocorrencia_inundacao` tem 260, **só de 2026** e **só com data** (`2026-04-01Z`). | Entram com `hora_conhecida=False`, **fora do rótulo**, e servem só para validação espacial (R16) |
| O Open-Meteo histórico bate com o ao vivo? | **Sim, com modelo fixo.** O `best_match` ao vivo para SP é o **ECMWF IFS 9 km**: mesma grade (−23,5149; −46,6105) e mesmos valores de `models=ecmwf_ifs`. O arquivo tem `ecmwf_ifs` desde **01/01/2017**; antes disso é ERA5. A API de previsões históricas (`best_match`) devolve os **mesmos valores** do arquivo (2017+). | `OPEN_METEO_MODELO="ecmwf_ifs"`, dataset **a partir de 2017** (R17). A chuva prevista vira só teto teórico (R1) |
| Há positivos suficientes? | **Sim.** CGE 2011–2025: 12.621 alagamentos (cidade) e 1.832 na Zona Norte. No período do dataset: treino 2017–22 = 5.008 (ZN 807), validação 2023 = 822 (ZN 100), teste 2024–25 = 1.333 (ZN 205). 39 % intransitáveis; duração mediana de 76 min. | Seguir com R = 300 m. Os positivos finais dependem de quantos eventos caem perto dos pontos escolhidos |

## Passo 1 — CGE (≈ 5.500 dias; com pausa de 1,5 s leva ≈ 2h30; retomável)

Coletamos desde **2011**. Os eventos de 2011–2016 não viram linhas do dataset (não há chuva
IFS), mas melhoram a escolha dos pontos e a `freq_hist_alagamento`.

```bash
source .venv/bin/activate
cp .env.example .env            # CONTATO_EMAIL
python -m src.collect.cge_alagamentos baixar  --inicio 2024-01-01 --fim 2024-01-31   # teste rápido
python -m src.collect.cge_alagamentos extrair --inicio 2024-01-01 --fim 2024-01-31
python -m src.collect.cge_alagamentos baixar  --inicio 2011-01-01 --fim 2025-12-31   # completo
python -m src.collect.cge_alagamentos extrair --inicio 2011-01-01 --fim 2025-12-31
```

Se aparecer o aviso "a página anuncia N pontos, mas M foram extraídos", anote a data: é sinal
de que o HTML do site mudou e os seletores de `src/collect/cge_alagamentos.py` precisam de ajuste.

## Passo 2 — GeoSampa (nomes conferidos)

```bash
python -m src.collect.geosampa_wfs baixar --camada geoportal:subprefeitura geoportal:drenagem \
    geoportal:risco_ocorrencia_alagamento geoportal:risco_ocorrencia_inundacao
python -m src.collect.geosampa_wfs extrair --arquivo data/raw/geosampa/risco_ocorrencia_alagamento.geojson \
    data/raw/geosampa/risco_ocorrencia_inundacao.geojson
```

- `subprefeitura`: 32 polígonos com `nm_subprefeitura` e `nm_regiao_05` (a zona oficial).
- `drenagem`: 28.366 trechos de cursos d'água (abertos e canalizados), usados em `dist_corrego_m`.
- Opcional: `geoportal:risco_hidrologico` (793 áreas de risco), candidata a feature fixa.

## Passo 3 — geocodificação, eventos e pontos

```bash
python -m src.pontos geocodificar     # Nominatim, 1 req/s, com cache; demora na 1ª vez
python -m src.pontos unificar         # imprime eventos por ano: cidade × Zona Norte
python -m src.pontos selecionar       # data/processed/pontos.csv
```

Formatos de referência do CGE já tratados:
- via: "AV OTAVIANO ALVES DE LIMA" → **cruzamento**;
- número: "461", "ALTURA DO N. 66", "ALTURA DO NÚMERO 196" → **numero**;
- lugar: → **referencia**.

Abreviações sem ponto (AV, R, AL, PC, PTE, CARD, CAP, STO, ENGA...) são expandidas. O que não
for localizado com precisão fica como `logradouro` e sai dos pontos e dos rótulos.

## Passo 4 — chuva e dataset

```bash
python -m src.collect.open_meteo --pontos data/processed/pontos.csv --inicio 2017-01-01 --fim 2025-12-31
python -m src.dataset --versao v1
```

As primeiras 72 h de 2017 saem do dataset automaticamente (não há janela completa).
Depois, abra `notebooks/03_dataset.ipynb` e preencha o `DATASET.md`.

## Rede bloqueada?

Se alguma fonte não responder, baixe os arquivos por outra conexão e coloque-os em
`data/raw/` com os mesmos nomes. As etapas de extração funcionam offline.
