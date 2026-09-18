# Método de treinamento e avaliação

Texto-base para o capítulo de metodologia do TCC. Os números entre `<>` são preenchidos
depois do treino com os dados reais (ver `reports/metricas_v1.json`).

## 1. Formulação

Classificação binária por **ponto monitorado × hora**. Para o ponto `p` e a hora cheia `t`
(UTC), o alvo é:

> `alagou_3h(p, t) = 1` se houve alagamento oficialmente registrado a até `R = 300 m` de `p`
> com início em `(t, t + 3h]`.

Horas em que `p` já estava alagado (`inicio ≤ t < fim`) são **removidas**: nelas não há o que
prever. Sem horário de término, assume-se uma duração de 2 h.

O rótulo é um **proxy** do evento operacional (o bueiro transbordar), porque não existe
histórico de bueiros.

## 2. Dados

| Fonte | Papel | Observação |
|---|---|---|
| CGE — alagamentos | **único rótulo** (2011–hoje; eventos antes de 2017 só para pontos e frequência) | tem hora de início; vias principais (CET) |
| GeoSampa — ocorrências (SIGRC) | só validação espacial | no WFS: só 2026 e só data |
| Open-Meteo (`ecmwf_ifs`, 9 km) | chuva horária | mesmo modelo e mesma grade no treino e ao vivo |
| GeoSampa — hidrografia, subprefeituras | características do ponto | |
| Open-Meteo Elevation (MDE 90 m) | elevação | |

**Recorte:** treino com a **cidade inteira** e avaliação também **recortada para a Zona Norte**.
Filtrar só a Zona Norte reduziria os positivos para `<N_ZN>` (contra `<N_cidade>`) e pioraria o
modelo.

## 3. Pontos monitorados

Os eventos com coordenada confiável (oficial, cruzamento ou referência) são agrupados por
DBSCAN (raio 200 m, mínimo de 3 eventos). São escolhidos os 100 agrupamentos mais recorrentes
da cidade mais todos os da Zona Norte, usando **só eventos até 2022**, para não escolher pontos
com base no período de teste.

## 4. Features

| Grupo | Features | Observação |
|---|---|---|
| Chuva (passado) | `chuva_1h/3h/6h/24h/72h`, `max_chuva_1h_24h`, `horas_desde_ultima_chuva` | janelas exigem horas completas |
| Tempo | `hora_dia`, `mes`, `dia_semana`, `estacao_chuvosa` | hora local |
| Ponto | `lat`, `lon`, `elevacao`, `dist_corrego_m`, `zona`, `freq_hist_alagamento` | frequência só com anos anteriores |
| (experimental) | `chuva_prevista_3h` | só com previsões históricas (R1) |

Todas as features vêm de `src/features.py`, que o serviço também usa. Um teste automatizado
garante que a linha calculada ao vivo é igual à do dataset.

## 5. Divisão e balanceamento

- **Temporal**: treino 2017–2022, validação 2023, teste 2024–2025 (o modelo ECMWF IFS do Open-Meteo começa em 2017).
- **Treino**: todas as horas positivas e todas as horas com chuva nas últimas 24 h; das horas
  secas negativas, uma amostra aleatória de 7 %, com peso `1/0,07` para preservar a
  distribuição.
- **Validação e teste**: completos (prevalência real de `<prev>`).

## 6. Modelos

1. **Baseline de regras**: limiares de chuva acumulada iguais aos do `RiskClassificationService`.
2. **Regressão logística**: `log1p` na chuva, padronização, one-hot na zona.
3. **LightGBM** (principal): taxa de aprendizado 0,03, 31 folhas, mínimo de 100 amostras por
   folha, subamostragem de linhas e colunas de 80 %, parada antecipada pela PR-AUC da validação
   (paciência de 150 rodadas).
4. **Random Forest** (alternativa) e **LSTM** (extra, notebook 05).

## 7. Calibração e níveis

A pontuação de cada modelo é calibrada na validação: regressão isotônica se houver pelo menos
300 positivos, Platt (sigmoide) se houver menos. Os níveis BAIXO/MEDIO/ALTO/CRITICO vêm de
limiares sobre a probabilidade calibrada:
- **fixos** (briefing): 0,15 / 0,40 / 0,70;
- **por metas** na validação: MEDIO captura 80 % dos positivos, ALTO captura 50 %, CRITICO tem
  precisão ≥ 50 %, respeitando um orçamento de alertas (≤ 5 % das horas em ≥MEDIO e ≤ 1 % em
  ≥ALTO).

O teste **nunca** é usado para escolher nada.

## 8. Métricas

- **Por hora**: PR-AUC (principal; referência = prevalência), ROC-AUC, Brier (referência =
  `prev·(1−prev)`) e precisão/recall em ≥MEDIO, ≥ALTO e CRITICO.
- **Por evento**: fração dos alagamentos com aviso ≥ALTO em alguma das 3 h anteriores
  (qualquer ponto a até R metros), antecedência média e taxa de alarme falso por ponto-dia.
- **Calibração**: curva por decis (probabilidade média × frequência observada).
- Todas também para a **Zona Norte**.

## 9. Ajuste pelo sensor

Regra não treinada, aplicada sobre o nível do modelo: água ≥ 100 % = transbordando (medição,
CRITICO); água ≥ 80 % sobe um nível; lixo ≥ 60 % com chuva sobe um nível. O ajuste nunca desce
o nível e tem teto em CRITICO. A probabilidade não muda. A regra é validada com o simulador.

## 10. Reprodutibilidade

- Semente fixa (`SEMENTE = 42`), inclusive na subamostragem por ponto (crc32 do id).
- `metadata_v1.json` guarda período, divisão, fonte da chuva, calibração, métricas e versões
  das bibliotecas.
- Todo dado bruto fica em `data/raw` com cache; o pipeline pode ser refeito do zero com os
  comandos do README.

## 11. Resultados do `v1` (17/09/2026)

Resumo em `docs/recomendacoes.md` (R18). Detalhes em `reports/comparacao_v1.md` e
`reports/metricas_v1.json`.

- LightGBM no teste (2024–25): PR-AUC 0,010 (prevalência 0,061 %), ROC-AUC 0,93; em ALTO,
  precisão 1,6 % e recall 22 %; 31 % dos eventos avisados com 2,4 h de antecedência.
- Zona Norte: ROC-AUC 0,95; 29 % dos eventos avisados com 2,6 h.
- Limiares por metas: MEDIO 0,0041, ALTO 0,0119, CRITICO 0,25.
- Features mais importantes: `chuva_3h` (34 %), `freq_hist_alagamento` (13 %), `mes` (12 %),
  `hora_dia` (8 %), `chuva_6h` (7 %).
- A calibração isotônica devolve poucos valores distintos (69 no teste): várias horas
  compartilham a mesma probabilidade. O orçamento de alertas trata esses empates
  (`calibracao.piso_orcamento`).
