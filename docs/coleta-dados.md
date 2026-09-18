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

Níveis de precisão (do melhor para o pior):

| Precisão | Como é obtida | Entra nos pontos/rótulos? |
|---|---|---|
| `numero` | "461", "ALTURA DO N. 66", "PROX AO NUMERO 1100" → endereço "via, número" | sim |
| `cruzamento` | referência é uma via → ponto onde as duas se cruzam | sim |
| `referencia` | referência é um lugar a até 1 km da via | sim |
| `logradouro_curto` | via inteira com até ~600 m (o centro fica a ≤ 300 m de qualquer ponto dela) | sim |
| `logradouro` | centro de uma via longa — posição sem valor | não |
| `falhou` | via não encontrada | não |

Tratamentos aplicados aos nomes (montados a partir dos endereços reais que falharam):
- abreviações com e sem ponto, inclusive coladas: `AV CD.DE FRONTIN`, `AV NSRA.DO O`,
  `R. BR. DO. BANANAL`, `TN JORN ...`, `ES ...`, `VEL ...`, `PÇ. ...`;
- sufixos de pista: `MARGINAL TIETE CBAS CENTRAL` → `MARGINAL TIETE`;
- nomes duplos: `PTE DA CASA VERDE-JORN.WALTER ABRAHAO` → tenta cada metade;
- variantes sem títulos e iniciais (`Ponte ROSSI ZUCCOLO`), aceitas só se cruzarem a outra via;
- avenidas longas vêm cortadas em 10 trechos: sem cruzamento, a busca é refeita com 40;
- prefixos da referência: `ALTURA DO`, `PROX AO`, `100M ANTES DA`...

Resultado da 1ª rodada (17/09/2026, 12.621 registros): cruzamento 5.883, logradouro 5.283,
falhou 977, numero 400, referencia 78 → 50 % aproveitáveis.

Resultado da 2ª rodada (17/09/2026, com os tratamentos acima): cruzamento 8.212,
logradouro 2.743, logradouro_curto 833, numero 448, falhou 350, referencia 35 →
9.528 aproveitáveis (**75 %**). Os 22 % em `logradouro` são quase todos avenidas longas sem
cruzamento (Marginais, Nações Unidas...), que ficam sub-representadas nos rótulos (R11).

### Escolha dos pontos (17/09/2026)

Simulação com os eventos da 2ª rodada, contando os alagamentos a até 300 m de algum ponto:

| Agrupamento | Mín. eventos | N cidade | Pontos (ZN) | Treino | Validação | Teste | ZN treino/val/teste |
|---|---|---|---|---|---|---|---|
| 150 m | 3 | 100 | 141 (58) | 2.552 | 375 | 534 | 562 / 71 / 153 |
| **200 m** | **3** | **100** | **138 (55)** | **2.432** | **374** | **548** | **555 / 69 / 149** |
| 200 m | 5 | 100 | 120 (37) | 2.436 | 368 | 543 | 513 / 65 / 144 |
| 300 m | 3 | 100 | 124 (43) | 1.960 | 340 | 438 | 406 / 46 / 91 |
| 200 m | 3 | 150 | 180 (55) | 2.668 | 416 | 597 | 555 / 69 / 149 |

Decisão: manter o padrão (agrupamento 200 m, mínimo 3 eventos, 100 da cidade + todos da
Zona Norte, rótulo a 300 m). São 138 pontos em 27 células de chuva. Agrupar a 300 m junta
pontos distintos e perde eventos; 150 pontos da cidade quase não acrescentam validação/teste.

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
