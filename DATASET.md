# DATASET — `dataset_v1`

> Ficha do dataset. Números de `data/processed/dataset_v1_resumo.json` (gerado em 17/09/2026).

## 1. Identificação

| Item | Valor |
|---|---|
| Arquivo | `data/processed/dataset_v1.parquet` |
| Gerado em | 17/09/2026 por `python -m src.dataset --versao v1` (8.081.784 linhas) |
| Unidade | 1 linha = 1 ponto monitorado × 1 hora cheia (UTC) |
| Período | 2017-01-01 a 2025-12-31 (hora local); eventos do CGE desde 2011 para pontos e frequência |
| Janela do rótulo | 3 h — início em (t, t+3h] |
| Raio do rótulo (R) | 300 m |
| Fonte da chuva | Open-Meteo arquivo, modelo `ecmwf_ifs` (9 km), o mesmo do serviço ao vivo |

## 2. Origem dos dados

| Fonte | Acesso | Período coletado | Registros brutos | Após filtros |
|---|---|---|---|---|
| CGE — alagamentos | raspagem de `cgesp.org/v3/alagamentos.jsp` em 16/09/2026 | 2011–2025 | 12.621 | 9.528 (geocodificação precisa) |
| GeoSampa — alagamento + inundação (SIGRC) | WFS `risco_ocorrencia_alagamento` (197) e `risco_ocorrencia_inundacao` (260) | só 2026, só data | 457 | fora do rótulo (validação espacial) |
| Open-Meteo | arquivo (`archive`), 27 células × 9 anos | 2017–2025 | — | — |
| GeoSampa — drenagem (cursos d'água) | WFS `drenagem` (28.366 trechos) | — | — | — |
| GeoSampa — subprefeituras | WFS `subprefeitura` (32, com `nm_regiao_05`) | — | — | — |
| Elevação | Open-Meteo Elevation (MDE 90 m) | — | — | — |

### Geocodificação do CGE

| Precisão | Registros | % |
|---|---|---|
| cruzamento | 8.212 | 65,1 % |
| logradouro_curto (via de até ~600 m) | 833 | 6,6 % |
| numero ("Referência: 461") | 448 | 3,5 % |
| referencia | 35 | 0,3 % |
| **aproveitados** | **9.528** | **75,5 %** |
| logradouro (descartado — via longa) | 2.743 | 21,7 % |
| falhou | 350 | 2,8 % |

A 1ª rodada (sem os tratamentos de nomes descritos em `docs/coleta-dados.md`) aproveitava 50 %.
Os descartes são quase todos avenidas longas sem cruzamento (Marginais, Nações Unidas...),
que ficam sub-representadas nos rótulos.

## 3. Eventos por ano — cidade × Zona Norte

(saída de `python -m src.pontos unificar`, só eventos com geocodificação precisa)

| Ano | Cidade | Zona Norte | % Norte |
|---|---|---|---|
| 2011 | 982 | 121 | 12 % |
| 2012 | 775 | 102 | 13 % |
| 2013 | 555 | 44 | 8 % |
| 2014 | 471 | 51 | 11 % |
| 2015 | 707 | 103 | 15 % |
| 2016 | 632 | 102 | 16 % |
| 2017 | 640 | 121 | 19 % |
| 2018 | 440 | 73 | 17 % |
| 2019 | 837 | 115 | 14 % |
| 2020 | 745 | 142 | 19 % |
| 2021 | 461 | 71 | 15 % |
| 2022 | 618 | 80 | 13 % |
| 2023 | 659 | 82 | 12 % |
| 2024 | 569 | 90 | 16 % |
| 2025 | 437 | 90 | 21 % |
| **Total** | **9.528** | **1.387** | 15 % |

Na Zona Norte, 1.387 dos 1.832 registros (76 %) foram aproveitados — a mesma taxa da cidade.
Os 419 eventos do GeoSampa (2026, sem hora) ficam fora.

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

- Agrupamento: DBSCAN, raio 200 m, mínimo 3 eventos, eventos até 2022 (comparação de alternativas em `docs/coleta-dados.md`).
- Seleção: 100 mais recorrentes da cidade + todos da Zona Norte → **137 pontos**
  (55 na Zona Norte, 24 Leste, 24 Sul, 23 Oeste, 11 Centro), em 27 células de chuva.
- Eventos por ponto: mediana 27 (3 a 314). Distância ao curso d'água: mediana 45 m.
  Elevação: 716 a 841 m. 32 pontos sem bairro (campo só descritivo).
- Arquivo: `data/processed/pontos.csv`.

## 5. Filtros e exclusões

| Etapa | Linhas |
|---|---|
| ponto × hora bruto (137 × 78.889 h) | 10.807.793 |
| − horas já alagadas | 5.402 |
| − horas sem 72 h de chuva completas (início de 2017) | 9.854 |
| − subamostragem de horas secas (treino, taxa 0,07) | 2.710.753 |
| **final** | **8.081.784** |

## 6. Contagens por divisão

| Split | Anos | Linhas | Positivos | Prevalência | Eventos distintos | Linhas ZN | Positivos ZN |
|---|---|---|---|---|---|---|---|
| treino | 2017–2022 | 4.480.201 | 6.540 | 0,146 % na amostra (real ponderada: 0,091 %) | 2.120 | 1.805.605 | 1.474 |
| validação | 2023 | 1.199.399 | 1.054 | 0,088 % | 335 | 481.710 | 190 |
| teste | 2024–2025 | 2.402.184 | 1.472 | 0,061 % | 483 | 964.469 | 394 |

Um evento vira até 3 linhas positivas (as 3 horas anteriores) em cada ponto a ≤ 300 m, por
isso há mais positivos do que eventos distintos.

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
- Avenidas longas sem cruzamento (Marginais, Nações Unidas, Alcântara Machado) ficam de fora
  por falta de posição precisa (22 % dos registros).
- **Prevalência cai no tempo:** 0,091 % no treino, 0,088 % na validação e 0,061 % no teste
  (menos registros do CGE em 2024–2025). A calibração ajustada em 2023 tende a superestimar
  um pouco no teste; o Brier e a curva de calibração do teste devem ser lidos com isso em mente.
- Poucos positivos na Zona Norte na validação (190 linhas, de poucas dezenas de eventos):
  as métricas por zona na validação são instáveis; o teste (394) é mais confiável.
