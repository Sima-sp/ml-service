# DATASET — `dataset_v1`

> Ficha do dataset. **Preencher depois da coleta** com os números de
> `data/processed/dataset_v1_resumo.json` e do notebook `03_dataset.ipynb`.
> Campos `<...>` estão pendentes.

## 1. Identificação

| Item | Valor |
|---|---|
| Arquivo | `data/processed/dataset_v1.parquet` |
| Gerado em | `<data>` por `python -m src.dataset --versao v1` |
| Unidade | 1 linha = 1 ponto monitorado × 1 hora cheia (UTC) |
| Período | 2017-01-01 a `<2025-12-31>` (hora local); eventos do CGE desde 2011 para pontos e frequência |
| Janela do rótulo | 3 h — início em (t, t+3h] |
| Raio do rótulo (R) | `<300>` m |
| Fonte da chuva | Open-Meteo arquivo, modelo `ecmwf_ifs` (9 km), o mesmo do serviço ao vivo |

## 2. Origem dos dados

| Fonte | Acesso | Período coletado | Registros brutos | Após filtros |
|---|---|---|---|---|
| CGE — alagamentos | raspagem de `cgesp.org/v3/alagamentos.jsp` em `<data>` | `<>` | `<>` | `<>` |
| GeoSampa — alagamento + inundação (SIGRC) | WFS `risco_ocorrencia_alagamento` (197) e `risco_ocorrencia_inundacao` (260) | só 2026, só data | 457 | fora do rótulo (validação espacial) |
| Open-Meteo | `<archive | historical-forecast>`, `<n>` células × `<n>` anos | `<>` | — | — |
| GeoSampa — drenagem (cursos d'água) | WFS `drenagem` (28.366 trechos) | — | — | — |
| GeoSampa — subprefeituras | WFS `subprefeitura` (32, com `nm_regiao_05`) | — | — | — |
| Elevação | Open-Meteo Elevation (MDE 90 m) | — | — | — |

### Geocodificação do CGE

| Precisão | Registros | % |
|---|---|---|
| numero ("Referência: 461") | `<>` | `<>` |
| cruzamento | `<>` | `<>` |
| referencia | `<>` | `<>` |
| logradouro (descartado) | `<>` | `<>` |
| falhou | `<>` | `<>` |

## 3. Eventos por ano — cidade × Zona Norte

(saída de `python -m src.pontos unificar`)

Contagem direta do CGE (zona conforme o próprio CGE agrupa), antes da geocodificação:

| Ano | Centro | Leste | Norte | Oeste | Sudeste | Sul | Total | % Norte |
|---|---|---|---|---|---|---|---|---|
| 2011 | 119 | 243 | 151 | 419 | 121 | 162 | 1.215 | 12 % |
| 2012 | 108 | 164 | 121 | 369 | 94 | 132 | 988 | 12 % |
| 2013 | 113 | 130 | 74 | 234 | 69 | 107 | 727 | 10 % |
| 2014 | 47 | 144 | 72 | 147 | 58 | 136 | 604 | 12 % |
| 2015 | 154 | 199 | 140 | 243 | 101 | 145 | 982 | 14 % |
| 2016 | 162 | 227 | 162 | 183 | 68 | 140 | 942 | 17 % |
| 2017 | 103 | 206 | 190 | 202 | 59 | 132 | 892 | 21 % |
| 2018 | 115 | 131 | 104 | 156 | 52 | 70 | 628 | 17 % |
| 2019 | 224 | 237 | 131 | 312 | 71 | 140 | 1.115 | 12 % |
| 2020 | 167 | 193 | 189 | 275 | 78 | 110 | 1.012 | 19 % |
| 2021 | 100 | 114 | 98 | 124 | 58 | 112 | 606 | 16 % |
| 2022 | 138 | 176 | 95 | 170 | 79 | 97 | 755 | 13 % |
| 2023 | 128 | 255 | 100 | 152 | 53 | 134 | 822 | 12 % |
| 2024 | 127 | 226 | 98 | 135 | 50 | 65 | 701 | 14 % |
| 2025 | 84 | 212 | 107 | 142 | 45 | 42 | 632 | 17 % |
| **Total** | 1.889 | 2.857 | 1.832 | 3.263 | 1.056 | 1.724 | **12.621** | 15 % |

- Zona Norte por subprefeitura (2011–2025): Santana 540, Vila Maria/Vila Guilherme 393,
  Casa Verde 392, Freguesia do Ó 184, Pirituba/Jaraguá 171, Jaçanã/Tremembé 77, Perus 75.
- 39,2 % dos registros são intransitáveis. Duração: mediana de 76 min, p75 de 166 min, máximo
  de 1.439 min. Durações acima de 12 h são tratadas como erro (`DURACAO_MAXIMA_EVENTO_H`).
- A Zona Norte tem só ~15 % dos eventos: reforça a decisão de treinar com a cidade inteira.
- Há queda de registros a partir de 2021 (mudança na operação da CET?). A prevalência do teste
  deve ser menor que a do treino (limitação a declarar).

## 4. Pontos monitorados

- Agrupamento: DBSCAN, raio `<200>` m, mínimo `<3>` eventos, eventos até `<2022>`.
- Seleção: `<100>` mais recorrentes da cidade + todos da Zona Norte → **`<N>` pontos**
  (`<N_ZN>` na Zona Norte), em `<n>` células de chuva.
- Arquivo: `data/processed/pontos.csv`.

## 5. Filtros e exclusões

| Etapa | Linhas |
|---|---|
| ponto × hora bruto | `<>` |
| − horas já alagadas | `<>` |
| − horas sem 72 h de chuva completas | `<>` |
| − subamostragem de horas secas (treino, taxa `<0,07>`) | `<>` |
| **final** | `<>` |

## 6. Contagens por divisão

| Split | Anos | Linhas | Positivos | Prevalência | Eventos distintos | Linhas ZN | Positivos ZN |
|---|---|---|---|---|---|---|---|
| treino | ≤ 2022 | | | (real ponderada: ) | | | |
| validação | 2023 | | | | | | |
| teste | 2024–2025 | | | | | | |

## 7. Dicionário de colunas

| Coluna | Tipo | Descrição |
|---|---|---|
| `ponto_id` | category | id do ponto (`P0001`…) — ver `pontos.csv` |
| `data_hora_utc` | datetime (UTC) | hora de referência `t` |
| `ano` | int16 | ano **local** de `t` (base da divisão) |
| `split` | category | treino / validacao / teste |
| `zona` | str | CENTRO / LESTE / NORTE / OESTE / SUL |
| `chuva_1h` … `chuva_72h` | float32 (mm) | soma da chuva nas últimas 1/3/6/24/72 h até `t` |
| `max_chuva_1h_24h` | float32 (mm) | maior chuva horária nas últimas 24 h |
| `horas_desde_ultima_chuva` | float32 (h) | horas desde a última hora ≥ 0,2 mm (teto 72) |
| `chuva_prevista_3h` | float32 (mm) | soma de t+1..t+3 — **com a fonte `arquivo` é chuva observada no futuro: não usar como feature** |
| `hora_dia`, `mes`, `dia_semana`, `estacao_chuvosa` | int8 | calendário local |
| `lat`, `lon` | float32 | centro do ponto |
| `elevacao` | float32 (m) | MDE 90 m |
| `dist_corrego_m` | float32 (m) | distância ao curso d'água mais próximo |
| `freq_hist_alagamento` | float32 (eventos/ano) | eventos a ≤ R m em anos **anteriores** ao da linha; NaN no 1º ano |
| `alagou_3h` | int8 | **rótulo** |
| `evento_alvo_id` | str | 1º evento que torna a linha positiva (métricas por evento) |
| `peso_amostra` | float32 | 1/taxa nas horas secas subamostradas; 1 no resto |

Arquivos irmãos: `dataset_v1_resumo.json` (contagens), `dataset_v1_freq_hist.parquet`
(frequência por ponto × ano, inclui o ano de produção), `dataset_v1_pontos.csv`.

## 8. Limitações conhecidas

- Rótulo proxy (alagamento registrado ≠ bueiro transbordando).
- CGE focado em vias principais; GeoSampa com hora de registro e possível atraso/duplicata.
- Geocodificação aproximada (filtrada por precisão).
- Chuva de modelo com resolução de alguns quilômetros.
- `<outras observadas na coleta>`
