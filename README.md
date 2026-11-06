# SIMA-SP — Serviço de IA (previsão de alagamentos)

Serviço Python do **SIMA-SP** (Sistema Inteligente de Monitoramento de Alagamentos em São Paulo,
TCC). Ele estima a **probabilidade de alagamento nas próximas 3 horas** para cada ponto
monitorado (bueiro, boca de lobo, via crítica) e devolve um **nível de risco**
(BAIXO / MEDIO / ALTO / CRITICO). O backend Java chama o serviço por HTTP.

> Documento-base: `BRIEFING_IA_SIMA-SP.md`. Este README explica **como o código implementa o
> briefing**. Os pontos em que a implementação diverge do briefing estão em
> [`docs/recomendacoes.md`](docs/recomendacoes.md). **Leia esse arquivo antes de coletar os dados.**

---

## Sumário

1. [Estado atual](#1-estado-atual)
2. [Visão geral](#2-visão-geral)
3. [Estrutura do repositório](#3-estrutura-do-repositório)
4. [Convenções de código](#4-convenções-de-código)
5. [Instalação](#5-instalação)
6. [Pipeline passo a passo](#6-pipeline-passo-a-passo)
7. [Módulos](#7-módulos)
8. [Como o treinamento funciona](#8-como-o-treinamento-funciona)
9. [Serviço HTTP (API)](#9-serviço-http-api)
10. [Tempo real](#10-tempo-real)
11. [Testes](#11-testes)
12. [Docker](#12-docker)
13. [Limitações (para o TCC)](#13-limitações-para-o-tcc)
14. [Documentação complementar](#14-documentação-complementar)

---

## 1. Estado atual

| Parte | Situação |
|---|---|
| Estrutura, configuração, Docker, Makefile | ✅ pronto |
| Coletores (CGE, GeoSampa, Open-Meteo, boletins) | ✅ **conferidos na fonte real em 16/09/2026** (HTML do CGE 2011–2026, camadas do GeoSampa, modelos do Open-Meteo) |
| Pontos recorrentes, rótulo, features, dataset | ✅ pronto e testado com dados sintéticos |
| Baseline, regressão logística, LightGBM, Random Forest, calibração, limiares, métricas | ✅ pronto e testado com dados sintéticos |
| Ajuste pelo sensor | ✅ pronto e testado |
| API FastAPI (`/health`, `/model-info`, `/predict`, `/predict/batch`, `/predict/regioes`) | ✅ pronta; sobe em **modo regras** enquanto não há modelo |
| **Dados reais** (`data/`) | 🔄 em coleta: CGE e GeoSampa baixados; geocodificação em andamento (ver [`docs/coleta-dados.md`](docs/coleta-dados.md)) |
| Modelo v1 treinado (`models/`) | ⏳ depende do dataset |
| Limiares do baseline iguais aos do `RiskClassificationService` | ⏳ copiar do código Java (`src/baseline.py`) |
| LSTM (extra) | 📝 esqueleto em `notebooks/05_lstm_extra.ipynb` |

Situação em 17/09/2026: 111 testes passando, pipeline completo (dataset → treino → API)
validado com dados sintéticos e serviço rodando em Docker (`make docker`, `/health` = 200).

---

## 2. Visão geral

```
                 TREINO (uma vez, offline)                          PRODUÇÃO (tempo real)
┌───────────────────────────────────────────────────┐   ┌──────────────────────────────────────────┐
│ CGE alagamentos ─┐                                │   │ Backend Java (@Scheduled 10 min          │
│ GeoSampa ────────┼─► eventos ─► pontos recorrentes│   │   ou leitura nova do sensor)             │
│                  │              │                 │   │        │ POST /predict/batch             │
│ Open-Meteo ──────┼──────────────┼─► dataset       │   │        ▼                                 │
│ (chuva horária)  │              │   ponto × hora  │   │ Serviço Python                           │
│ GeoSampa hidro/  │              │       │         │   │  ├─ Open-Meteo ao vivo (cache 15 min)    │
│ subprefeituras ──┘              ▼       ▼         │   │  ├─ src/features.py  ◄── MESMO CÓDIGO ──┼──┐
│                         src/features.py ──► train │   │  ├─ modelo + calibração                  │  │
│                                           │       │   │  ├─ limiares → nível                     │  │
│                                           ▼       │   │  └─ ajuste do sensor (regra)             │  │
│                             models/flood_lgbm_v1  ├───┼─► resposta: 72 % · ALTO · até 14:50      │  │
└───────────────────────────────────────────────────┘   └──────────────────────────────────────────┘  │
                         ▲                                                                           │
                         └───────────────────────────────────────────────────────────────────────────┘
```

**Definições (briefing, seção 3):**

| Conceito | Definição no código |
|---|---|
| Unidade | 1 linha = 1 ponto × 1 hora cheia `t` (UTC) |
| Rótulo `alagou_3h` | alagamento registrado a ≤ `RAIO_ROTULO_M` (300 m) com início em **(t, t+3h]** |
| Janela | 3 h, fixa (`JANELA_PREVISAO_HORAS`) |
| Saída | `probabilidadeAlagamento` (0–1, calibrada) + `nivelRisco` |
| Medido × previsto | `nivelAgua ≥ 100 %` é **medição** (`medicaoTransbordando`); a probabilidade é **previsão** |

---

## 3. Estrutura do repositório

```
ml-service/
├── README.md                  ← você está aqui
├── DATASET.md                 ← ficha do dataset (preencher após a coleta)
├── Makefile                   ← atalhos: make testar | sintetico | dataset | treinar | servir | docker
├── Dockerfile                 ← imagem enxuta do serviço (python:3.12-slim)
├── docker-compose.yml         ← sobe só o ml-service (porta 5000)
├── requirements.txt           ← ambiente completo (coleta, geo, notebooks, testes)
├── requirements-servico.txt   ← só o que a imagem de produção precisa
├── requirements-extra.txt     ← + PyTorch para a LSTM (extra)
├── pyproject.toml             ← config do pytest e do ruff
├── .env.example               ← variáveis de ambiente comentadas (copie para .env)
│
├── src/                       ← PIPELINE (treino) + código compartilhado com o serviço
│   ├── config.py              ← TODOS os parâmetros do problema num lugar só
│   ├── tempo.py               ← UTC ↔ hora local, hora de referência
│   ├── collect/               ← coletores das fontes públicas
│   │   ├── http.py            ← sessão com cache em disco, novas tentativas, pausa
│   │   ├── cge_alagamentos.py ← fonte A (rótulo principal)
│   │   ├── geosampa_wfs.py    ← fonte B + hidrografia + subprefeituras
│   │   ├── open_meteo.py      ← chuva: histórico (treino) e recente (serviço)
│   │   └── cge_pluvio.py      ← boletins pluviométricos (melhoria)
│   ├── geo.py                 ← distâncias, DBSCAN, geocodificação (Nominatim)
│   ├── pontos.py              ← eventos unificados → pontos recorrentes
│   ├── labels.py              ← alagou_3h, horas já alagadas, freq. histórica
│   ├── features.py            ← ★ features — MESMO código no treino e ao vivo
│   ├── dataset.py             ← tabela ponto × hora (Parquet)
│   ├── baseline.py            ← régua de regras (≈ RiskClassificationService)
│   ├── modelos.py             ← logística, LightGBM, Random Forest
│   ├── calibracao.py          ← Platt/isotônica + escolha de limiares
│   ├── niveis.py              ← escala BAIXO/MEDIO/ALTO/CRITICO
│   ├── sensor_adjust.py       ← ajuste por regra com a leitura do sensor
│   ├── artefatos.py           ← salvar/carregar o pacote do modelo
│   ├── train.py               ← treino + comparação + relatórios
│   ├── evaluate.py            ← métricas (PR-AUC, Brier, por evento, Zona Norte), gráficos
│   └── sintetico.py           ← dados falsos SÓ para testar o pipeline
│
├── app/                       ← SERVIÇO FastAPI
│   ├── main.py                ← rotas e ciclo de vida
│   ├── schemas.py             ← contrato JSON (Pydantic)
│   ├── predictor.py           ← orquestra uma previsão
│   ├── fonte_chuva.py         ← Open-Meteo ao vivo + cache + fallback
│   ├── caracteristicas_ponto.py ← características do ponto mais próximo
│   └── configuracao.py        ← variáveis de ambiente
│
├── data/                      ← NÃO versionado (exceto a estrutura)
│   ├── raw/                   ← downloads originais (HTML, GeoJSON, parquet por célula-ano)
│   ├── interim/               ← tabelas limpas intermediárias
│   └── processed/             ← pontos.csv, dataset_v1.parquet, resumos
├── models/                    ← flood_lgbm_v1.joblib, thresholds_v1.json, metadata_v1.json, pontos_v1.csv
├── reports/                   ← métricas, tabelas e gráficos gerados pelo treino
├── notebooks/                 ← 01 inspeção · 02 pontos · 03 dataset · 04 modelos · 05 LSTM
├── scripts/                   ← checar_fontes.py, demo_cenarios.py
├── tests/                     ← pytest (111 testes)
└── docs/
    ├── recomendacoes.md       ← ★ decisões e sugestões para discutir
    ├── coleta-dados.md        ← roteiro do próximo passo
    ├── treinamento.md         ← método detalhado (para o capítulo do TCC)
    ├── tempo-real.md          ← fluxo ao vivo, cache e falhas
    ├── integracao-backend.md  ← mudanças no Java (DTOs, tabela, agendador, compose)
    └── adr/                   ← registros de decisão de arquitetura
```

---

## 4. Convenções de código

- **Tudo em português**: módulos, funções, variáveis, docstrings e comentários
  (`calcular_features_chuva`, `nivel_por_probabilidade`). A exceção são as APIs das
  bibliotecas (`fit`, `predict_proba`) e os nomes de pasta pedidos no briefing
  (`collect/`, `train.py`, `evaluate.py`, `sensor_adjust.py`).
- **Identificadores sem acento** (`nivel`, `previsao`); acentos só em textos e comentários.
- **Campos JSON** em camelCase, iguais aos DTOs do backend (`probabilidadeAlagamento`).
- **Colunas de dados** em snake_case, iguais ao briefing (`chuva_3h`, `alagou_3h`).
- Parâmetros de negócio **só** em `src/config.py`; nada de número mágico espalhado.
- Datas em **UTC** internamente (`data_hora_utc`); hora local só para calendário e fontes locais.
- Todo módulo começa com uma docstring explicando **o que faz e por quê**.
- Qualidade: `make lint` (ruff) e `make testar` (pytest) antes de cada commit.

---

## 5. Instalação

Requisitos: Python **3.11+** (3.12 recomendado), `make`, Docker (opcional).

```bash
cd ml-service
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt       # ambiente completo
cp .env.example .env                  # ajuste CONTATO_EMAIL (exigido pelo Nominatim)

make testar                           # 111 testes, ~8 s
make sintetico                        # pipeline completo com dados falsos (~1 min)
```

> `geopandas` pode exigir bibliotecas do sistema em algumas distros. No Linux Mint/Ubuntu,
> `pip install geopandas` já traz as rodas (wheels) prontas. Se falhar:
> `sudo apt install libgdal-dev`.

---

## 6. Pipeline passo a passo

Cada etapa lê o que a anterior gravou em disco, então você pode parar e retomar a qualquer
momento. Roteiro detalhado, com o portão de decisão, em [`docs/coleta-dados.md`](docs/coleta-dados.md).

| # | Comando | Gera |
|---|---|---|
| 0 | `make checar-fontes` | diagnóstico de rede |
| 1 | `python -m src.collect.cge_alagamentos baixar --inicio 2011-01-01 --fim 2025-12-31` | `data/raw/cge_alagamentos/html/` |
| 2 | `python -m src.collect.cge_alagamentos extrair --inicio ... --fim ...` | `data/interim/eventos_cge.parquet` |
| 3 | `python -m src.collect.geosampa_wfs baixar --camada geoportal:subprefeitura geoportal:drenagem geoportal:risco_ocorrencia_alagamento geoportal:risco_ocorrencia_inundacao` | `data/raw/geosampa/*.geojson` |
| 4 | `python -m src.collect.geosampa_wfs extrair --arquivo data/raw/geosampa/risco_ocorrencia_alagamento.geojson data/raw/geosampa/risco_ocorrencia_inundacao.geojson` | `data/interim/eventos_geosampa.parquet` (só validação: 2026, sem hora) |
| 5 | — | — |
| 6 | `python -m src.pontos geocodificar` | `data/interim/eventos_cge_geocodificados.parquet` |
| 7 | `python -m src.pontos unificar` | `data/interim/eventos.parquet` + contagem cidade × ZN |
| 8 | `python -m src.pontos selecionar` | `data/processed/pontos.csv` |
| 9 | `python -m src.collect.open_meteo --pontos data/processed/pontos.csv --inicio 2017-01-01 --fim 2025-12-31` | `data/interim/chuva_horaria.parquet` |
| 10 | `python -m src.dataset --versao v1` | `data/processed/dataset_v1.parquet` + resumo |
| 11 | `python -m src.train --dataset v1 --versao v1` | `models/*_v1.*` + `reports/` |
| 12 | `make servir` → http://localhost:5000/docs | API |

---

## 7. Módulos

### `src/config.py`
Fonte única dos parâmetros: janela (3 h), raio do rótulo (300 m), raio do DBSCAN (200 m),
período, divisão temporal, taxa de subamostragem, limiares, regras do sensor, validade da
previsão, URLs. Os itens marcados `DECISÃO EM ABERTO` dependem da análise dos dados.

### `src/tempo.py`
Converte hora local → UTC tratando o horário de verão (até 2019) e define a **hora de
referência**: uma previsão às 14:20 usa `t = 14:00`, porque o valor "14:00" do Open-Meteo é a
chuva de 13:00 a 14:00 (último intervalo completo).

### `src/collect/`
Padrão **baixar → extrair**: o bruto fica em `data/raw` e nunca é baixado duas vezes.
- `cge_alagamentos.py`: raspa a página por data. Guarda o HTML de cada dia e extrai horário
  de início e fim, logradouro, sentido, referência e se a via está transitável. Atravessar a
  meia-noite é tratado. ⚠️ Conferir `METODO_HTTP`, `montar_parametros()` e as classes CSS na
  aba Rede.
- `geosampa_wfs.py`: lista camadas (`GetCapabilities`), baixa com paginação no sistema nativo
  e converte para WGS84. Mapeia colunas por listas de candidatos (conferir no Dicionário de
  Dados) e remove duplicatas (~50 m, mesma hora).
- `open_meteo.py`: histórico (arquivo ou previsões históricas) e recente (`past_hours` +
  `forecast_hours`). Agrupa pontos em **células de 0,05°** e pede a chuva no centro da
  célula, igual no treino e ao vivo. O download é retomável (um arquivo por célula × ano).
  Também tem `comparar_historico_e_ao_vivo()` para o portão de decisão.
- `cge_pluvio.py`: leitor genérico dos boletins (baixe pelo navegador). É melhoria opcional.

### `src/geo.py`
Haversine, busca por raio (BallTree), DBSCAN com métrica haversine, zona a partir da
subprefeitura, distância a linhas e ponto em polígono (geopandas, EPSG:31983), e o
**Geocodificador** (Nominatim, 1 req/s, cache em JSON). A estratégia do geocodificador é:
cruzamento das vias → lugar de referência → centro da via (precisão baixa).

### `src/pontos.py`
Une CGE + GeoSampa, descarta coordenadas imprecisas ou fora do município, atribui zona e conta
**eventos por ano, cidade × Zona Norte** (tarefa de comprovação do briefing). Depois agrupa
com DBSCAN, escolhe os N mais recorrentes + todos da Zona Norte (**só com eventos até 2022**,
R10) e adiciona distância ao córrego, elevação e célula de chuva.

### `src/labels.py`
`rotular_ponto()` usa `searchsorted` (vetorizado) para marcar `alagou_3h`, `ja_alagado` e
`evento_alvo_id`. `taxa_historica_anual()` calcula `freq_hist_alagamento` só com anos
anteriores (R4).

### `src/features.py` ★
Coração da consistência treino × produção. `calcular_features_chuva()` recebe uma série
horária e devolve todas as janelas (1, 3, 6, 24 e 72 h, pico em 24 h, horas desde a última
chuva). Janelas incompletas viram `NaN`, nunca uma soma subestimada. `features_ao_vivo()` é o
que o serviço chama: usa **as mesmas funções** e devolve uma linha. Um teste garante que a
linha ao vivo é idêntica à do dataset.

### `src/dataset.py`
Monta a tabela ponto × hora (detalhes na seção 8) e grava Parquet + resumo JSON + a tabela de
frequência histórica + a lista de pontos da versão.

### `src/baseline.py`
Classificador por limiares de chuva acumulada: é a régua do TCC e o modo regras do serviço.
⚠️ **Os limiares são provisórios**: copie os do `RiskClassificationService.java`.

### `src/modelos.py`
- **Regressão logística**: `log1p` na chuva, padronização, one-hot na zona, imputação pela
  mediana (com indicador de ausência).
- **LightGBM**: parada antecipada pela PR-AUC da validação; `zona` como categoria nativa.
- **Random Forest**: alternativa, com o mesmo pré-processamento sem padronização.
- Nenhum usa `class_weight`: o desbalanceamento é tratado pela subamostragem com peso e pela
  calibração.

### `src/calibracao.py`
`Calibrador` (Platt/sigmoide ou isotônica; `auto` escolhe pela quantidade de positivos) e
`escolher_limiares()` (metas de precisão/recall na validação com orçamento de alertas, sempre
com MEDIO < ALTO < CRITICO).

### `src/niveis.py` e `src/sensor_adjust.py`
Escala única de risco e a regra do sensor: ≥100 % = transbordando (medido, CRITICO); ≥80 %
sobe um nível; lixo ≥60 % com chuva sobe um nível; as regras acumulam, nunca descem e têm teto
em CRITICO.

### `src/artefatos.py`
`PacoteModelo` junta estimador, calibrador, features, limiares e metadados; salva e carrega os
quatro arquivos de `models/`.

### `src/train.py` e `src/evaluate.py`
Ver seção 8.

### `app/`
Ver seções 9 e 10.

---

## 8. Como o treinamento funciona

Versão resumida. O método completo, para o capítulo do TCC, está em
[`docs/treinamento.md`](docs/treinamento.md).

1. **Dataset** (`src/dataset.py`)
   - Para cada célula de chuva, calcula as features uma vez; para cada ponto da célula, junta
     calendário (hora local), rótulo, características fixas e `freq_hist_alagamento` do ano.
   - **Remove** as horas em que o ponto já está alagado (não há o que prever) e as horas sem
     72 h completas de chuva.
   - **Divisão temporal** pelo ano local: treino 2017–2022, validação 2023, teste 2024–2025.
   - **Subamostragem só no treino** (R2): mantém todos os positivos e todas as horas com
     chuva nas últimas 24 h, e sorteia 7 % das horas secas negativas, com
     `peso_amostra = 1/0,07`.

2. **Modelos** (`src/train.py`), na ordem do briefing:
   1. baseline de regras (sem treino);
   2. regressão logística;
   3. LightGBM (principal): até 3000 árvores, parando quando a PR-AUC da validação não
      melhora por 150 rodadas;
   4. Random Forest (opcional); LSTM no notebook 05 (extra).

3. **Calibração** na validação (prevalência real). `auto` escolhe isotônica com ≥ 300
   positivos e sigmoide com menos.

4. **Limiares dos níveis**: fixos do briefing (`--limiares fixos`) ou por metas na validação
   (`--limiares metas`, R3). As metas combinam recall/precisão com um **orçamento de alertas**
   (no máximo 5 % das horas em ≥MEDIO e 1 % em ≥ALTO). Os limiares são salvos em
   `thresholds_v1.json` com o diagnóstico.

5. **Avaliação** no teste, geral e **Zona Norte**:
   - por hora: PR-AUC (principal), ROC-AUC, Brier (com a referência "sempre a prevalência"),
     precisão e recall em ≥MEDIO, ≥ALTO e CRITICO;
   - por evento: % de alagamentos com aviso ≥ALTO antes do início, antecedência média e taxa
     de alarme falso por ponto-dia.

6. **Saídas**:
   - `models/flood_lgbm_v1.joblib`, `thresholds_v1.json`, `metadata_v1.json`, `pontos_v1.csv`;
   - `reports/metricas_v1.json`, `comparacao_v1.md`, `curva_pr_v1.png`,
     `calibracao_v1.png` e `importancia_features_v1.csv`.

**Vazamento de dados — o que o código impede:**
- nenhuma feature usa valores posteriores a `t` (teste `test_sem_vazamento_do_futuro`);
- `chuva_prevista_3h` fica bloqueada com a reanálise (`train.py` recusa);
- frequência histórica só com anos anteriores;
- divisão temporal, pontos escolhidos só com o período de treino, limiares escolhidos só na
  validação.

```bash
python -m src.train --dataset v1 --versao v1                                  # padrão
python -m src.train --dataset v1 --versao v1 --limiares metas                 # R3
python -m src.train --dataset v1 --versao v1 --modelos baseline,logistica,lightgbm,random_forest
python -m src.train --dataset v1prev --versao v1prev --usar-chuva-prevista     # R1 (previsões históricas)
```

---

## 9. Serviço HTTP (API)

`uvicorn app.main:app --port 5000` · Swagger em **http://localhost:5000/docs**

| Rota | Uso |
|---|---|
| `GET /health` | `{"status":"ok","modeloVersao":"v1","origem":"MODELO",...}` |
| `GET /model-info` | features, período, divisão, calibração, limiares, métricas de teste, versões das libs |
| `POST /predict` | uma previsão |
| `POST /predict/batch` | lista de requisições → lista de resultados **na mesma ordem**; um item com erro não derruba o lote |
| `GET /predict/regioes` | risco por subprefeitura (maior risco entre os pontos da região), para o aviso de região do app — ver ADR 0005 |
| `POST /predict/regioes` | o mesmo, com `chuvaHoraria` informada e aplicada a todos os pontos — só para simulação/demonstração; a resposta sai com `fonteChuva = "INFORMADA"` |

### Requisição (compatível com o payload antigo)

```json
{
  "sensorId": 12,
  "latitude": -23.5152,
  "longitude": -46.5841,
  "timestamp": "2026-09-16T14:20:00-03:00",
  "nivelAgua": 82.0,
  "porcentagemLixo": 64.0,
  "chuvaMm": 30.0,
  "chuva": { "chuva_1h": 12.4, "chuva_3h": 30.0, "chuva_24h": 41.2 },
  "chuvaHoraria": null
}
```

- Só `latitude` e `longitude` são obrigatórios.
- De onde vem a chuva, em ordem de prioridade:
  1. `chuvaHoraria` (≥ 72 valores, o último é a hora de referência) → caminho exato, ideal para a demo;
  2. Open-Meteo/cache, com `chuva` (se vier) sobrescrevendo os acumulados;
  3. só `chuva` → acumulados completados por aproximação;
  4. só `chuvaMm` (legado) → tratado como `chuva_1h`.
- Sem nenhuma chuva → **503** `CHUVA_INDISPONIVEL`, e o backend usa as regras.

### Resposta

```json
{
  "sensorId": 12,
  "probabilidadeAlagamento": 0.72,
  "nivelRisco": "CRITICO",
  "nivelRiscoModelo": "ALTO",
  "janelaHoras": 3,
  "geradaEm": "2026-09-16T14:20:03-03:00",
  "validaAte": "2026-09-16T14:50:03-03:00",
  "horaReferencia": "2026-09-16T14:00:00-03:00",
  "medicaoTransbordando": false,
  "ajusteSensorAplicado": true,
  "motivosAjuste": ["nivelAgua 82% >= 80%"],
  "semLeituraSensor": false,
  "fonteChuva": "OPEN_METEO",
  "chuvaRecente3hMm": 30.0,
  "chuvaPrevista3hMm": 4.2,
  "origem": "MODELO",
  "modeloVersao": "v1",
  "pontoId": "P0042",
  "distanciaPontoM": 35.2,
  "avisos": []
}
```

Campos além do briefing (todos aditivos; ver [`docs/adr/0001-contrato-predict.md`](docs/adr/0001-contrato-predict.md)):
`nivelRiscoModelo`, `horaReferencia`, `motivosAjuste`, `semLeituraSensor`,
`chuvaRecente3hMm`, `chuvaPrevista3hMm`, `origem`, `pontoId`, `distanciaPontoM`, `avisos`.

| `fonteChuva` | Significado |
|---|---|
| `OPEN_METEO` | baixada agora ou cache com < 15 min |
| `CACHE` | Open-Meteo falhou; usada chuva com < 60 min |
| `INFORMADA` | veio no request (`chuvaHoraria`/`chuva`) |
| `INFORMADA_LEGADO` | só `chuvaMm` |

| Código de erro | HTTP | Quando |
|---|---|---|
| `CHUVA_INDISPONIVEL` | 503 | sem Open-Meteo, sem cache e sem chuva informada |
| `CHUVA_FORA_DA_JANELA` | 422 | `timestamp` antigo demais para a série ao vivo (~4 dias) |
| `LOTE_GRANDE_DEMAIS` | 413 | mais de 500 itens no lote |
| (validação Pydantic) | 422 | campos fora do intervalo |

Cenários prontos: `make demo` (seco, chuva forte, bueiro entupido, água alta, transbordando,
manutenção, chuva real).

---

## 10. Tempo real

Resumo (completo em [`docs/tempo-real.md`](docs/tempo-real.md)):
- o modelo é carregado **uma vez** na inicialização;
- a chuva é cacheada **por célula** por 15 min, e o lote usa **uma** chamada ao Open-Meteo;
- se o Open-Meteo falhar, usa a chuva com < 60 min (`CACHE`); sem isso, 503 e o backend usa as regras;
- `validaAte = geradaEm + 30 min`: passado isso, o app mostra "previsão desatualizada";
- sensor INACTIVE/MAINTENANCE: o backend manda sem `nivelAgua`/`porcentagemLixo` e a resposta
  vem com `semLeituraSensor = true`;
- sem modelo treinado: **modo regras** (`origem = REGRAS`, probabilidade `null`).

---

## 11. Testes

```bash
make testar            # ou: python -m pytest -v
```

Só com o ambiente do serviço: `pip install -r requirements-servico.lock -r requirements-teste.txt`.
O CI (`.github/workflows/ci.yml`) roda a suíte assim em todo PR e confere o build da imagem Docker.

| Arquivo | Cobre |
|---|---|
| `test_features.py` | janelas, NaN em buracos, **sem vazamento**, horário de verão, ao vivo = treino, resumo parcial |
| `test_labels.py` | janela (t, t+3h], horas alagadas, eventos sobrepostos, raio, frequência histórica |
| `test_dataset.py` | divisão temporal, subamostragem só no treino, pesos, evento-alvo |
| `test_sensor_adjust.py` | limites 79,9/80/100 %, lixo com e sem chuva, acumulação, teto, sem leitura |
| `test_niveis_calibracao.py` | limites dos níveis, calibração melhora Brier, limiares ordenados, orçamento de alertas |
| `test_fonte_chuva.py` | cache por célula, 1 chamada por lote, fallback < 60 min, falha total |
| `test_api.py` | schema, payload antigo e novo, lote, 503, legado, modo regras, acento configurável |
| `test_baseline_artefatos.py` | baseline e ida-e-volta do pacote |
| `test_coleta.py` | parser do CGE (HTML de exemplo, meia-noite, horário de verão), GeoSampa em UTM → WGS84, zonas, distância ao córrego, DBSCAN/seleção, geocodificador (cruzamento/referência/logradouro) |

Nenhum teste acessa a internet: a chuva e as respostas do Nominatim são simuladas e o modelo de
teste é treinado em segundos com dados sintéticos. Os testes geoespaciais são pulados se
`geopandas`/`shapely` não estiverem instalados.

---

## 12. Docker

```bash
make docker                        # docker compose up --build  → http://localhost:5000/health
docker build -t sima-ml-service .  # só a imagem
```

- A imagem usa `requirements-servico.txt` (sem geopandas/jupyter) e copia `src/`, `app/` e `models/`.
- Roda com usuário não-root e tem `HEALTHCHECK` em `/health`.
- `models/` também é montado como volume somente leitura: dá para trocar o modelo sem rebuild.
- Para o compose do projeto inteiro, veja [`docs/integracao-backend.md`](docs/integracao-backend.md).

---

## 13. Limitações (para o TCC)

- **Rótulo proxy**: alagamento registrado ≠ bueiro transbordando.
- **Viés das fontes**: o CGE cobre vias principais (CET) e o GeoSampa depende de chamados; a
  hora do GeoSampa é a do registro.
- **Chuva de modelo/reanálise** com resolução espacial limitada (~9–25 km): pancadas muito
  localizadas se perdem.
- **Sem chuva futura** no modelo principal: pouca antecedência em tempestades que ainda não
  começaram (R1).
- **Geocodificação** de endereços do CGE tem erro, que o filtro de precisão reduz mas não elimina.
- **Sensor entra por regra**, não por aprendizado (não há histórico).
- **Sensores simulados**: sem validação de campo.
- Com dados reais acumulados, dá para retreinar com o rótulo verdadeiro.

---

## 14. Documentação complementar

| Documento | Conteúdo |
|---|---|
| [`docs/recomendacoes.md`](docs/recomendacoes.md) | ★ decisões em aberto e sugestões (R1–R15) |
| [`docs/coleta-dados.md`](docs/coleta-dados.md) | roteiro da coleta + portão de decisão |
| [`docs/treinamento.md`](docs/treinamento.md) | método completo |
| [`docs/tempo-real.md`](docs/tempo-real.md) | fluxo ao vivo, cache, falhas, cota |
| [`docs/integracao-backend.md`](docs/integracao-backend.md) | DTOs Java, tabela `prediction`, agendador, alertas, compose |
| [`docs/adr/`](docs/adr) | 0001 contrato · 0002 fonte de chuva e tempo · 0003 estratégia de treino · 0004 modo regras |
| [`DATASET.md`](DATASET.md) | ficha do dataset |

## Como o grupo trabalha

- Ninguém envia direto para a `main`: cada mudança vai numa branch e entra por pull request.
- O PR só pode ser juntado com o CI passando (job `ci`: a suíte pytest).
- O merge é sempre *squash* (um commit por PR) e a branch é apagada depois.
- Mudança no contrato do `/predict` vira uma ADR em `docs/adr/`.
- O Dependabot cuida só das GitHub Actions e da imagem base. Bibliotecas Python são atualizadas à mão, junto com um novo treino (R14).
