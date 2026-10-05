# Recomendações e decisões para discutir

Pontos em que a implementação diverge do briefing, ajustes propostos e decisões ainda em
aberto. Cada item diz **o que já está implementado** e **o que ainda precisa ser decidido pelo
grupo**. A ordem é por impacto no resultado do TCC.

> Legenda: ✅ implementado como padrão · 🔀 implementado, mas desligado (depende de decisão)
> · ❓ decisão pendente, nada implementado

---

## R1 — Chuva **prevista** para as próximas 3 h (maior impacto) 🔀 — *revisado em 16/09*

**Problema.** Todas as features de chuva do briefing olham para o passado (até `t`). Em São
Paulo, a maior parte dos alagamentos vem de pancadas curtas (1–3 h). Muitas vezes, na hora `t`,
a tempestade que vai causar o alagamento em `t+1h` **ainda não começou**. O modelo só consegue
"ver" o evento quando a chuva já está caindo.

**O que a verificação mostrou.** A API "Historical Forecast" do Open-Meteo (`best_match`)
devolve **exatamente os mesmos valores** do arquivo de 2017 em diante. Ela junta as primeiras
horas de cada rodada do modelo, então é praticamente a chuva que de fato ocorreu. Usar essa
série como "chuva prevista" no treino daria um resultado **otimista demais**, porque ao vivo a
previsão para t+1..t+3 erra mais.

**Proposta.**
- **v1 (principal)**: só chuva passada, como no briefing.
- **Experimento "teto teórico"**: `--usar-chuva-prevista --permitir-oraculo`. Mostra quanto uma
  previsão perfeita das próximas 3 h acrescentaria. Deve ser apresentado **como limite
  superior**, não como desempenho real.
- **Opcional (se sobrar tempo)**: um limite inferior realista com a "Previous Runs API" do
  Open-Meteo (previsões emitidas com 1 dia de antecedência). A verdade fica entre os dois.
- **Ao vivo**, a resposta já traz `chuvaPrevista3hMm` como informação para o app.

**Ilustração (dados sintéticos, não é resultado).** No teste de fumaça, a PR-AUC foi de 0,0007
(só passado) para 0,057 com a chuva futura perfeita.

## R2 — Subamostrar **só o treino** ✅

O briefing sugere subamostrar as horas secas e depois "recalibrar corrigindo a taxa". Eu
subamostro **apenas o treino** (com `peso_amostra = 1/taxa`) e deixo validação e teste com
**todas** as horas. Assim:
- PR-AUC, precisão e Brier da validação/teste refletem a prevalência real, sem correção manual;
- a calibração é feita direto na validação, sem fórmula de ajuste de prevalência.

Custo: validação + teste ficam com ~2–3 milhões de linhas para ~100 pontos. Isso cabe
tranquilamente em memória e o LightGBM prevê isso em segundos.

---

## R3 — Limiares 0,15 / 0,40 / 0,70 **nunca** disparam (confirmado no v1) ✅/❓

Com prevalência na casa de **0,05 % por ponto-hora**, uma probabilidade **calibrada** de 40 %
é raríssima, mesmo durante um temporal. Com os limiares fixos, o app mostraria "BAIXO" quase
sempre, inclusive em dias de alagamento. No teste sintético, a maior probabilidade calibrada
foi 0,11 e **todas** as 526 mil horas de teste ficaram em BAIXO com os limiares fixos.

**Proposta:**
1. escolher os limiares na validação por metas (`python -m src.train --limiares metas`).
   As metas padrão são: MEDIO captura 80 % dos positivos, ALTO captura 50 % e CRITICO tem
   precisão de 50 %. Elas vêm com um **orçamento de alertas**: no máximo 5 % das horas em
   ≥MEDIO e 1 % em ≥ALTO. Sem esse orçamento, um modelo fraco atinge o recall pedido
   colocando um terço das horas em ALTO (aconteceu no teste sintético). Os valores ficam em
   `config.METAS_LIMIARES`;
2. no app, o **nível** é a informação principal. A porcentagem fica em segundo plano, com uma
   explicação curta (ex.: "8 % — cerca de 150× o normal para este ponto"). Sem isso, "8 % · ALTO"
   parece contraditório.

**Confirmado com os dados reais (17/09/2026):** no `v1` com limiares fixos, o LightGBM não pôs
nenhuma hora em ALTO (a maior probabilidade calibrada do teste foi 0,25). O `v1` passou a ser
treinado com `--limiares metas` (ver R18).

**Decisão pendente:** como o app mostra a porcentagem; se o backend adota os limiares do
`thresholds_v1.json`.

---

## R4 — `freq_hist_alagamento` só com anos **anteriores** ✅

O briefing diz "calcular só no período de treino". Isso evita vazamento para o teste, mas ainda
deixa cada linha de treino enxergar **o próprio evento** dentro da frequência. Com poucos
eventos por ponto, isso infla a importância dessa feature. A implementação usa uma janela expansiva:
a linha do ano `Y` usa só eventos de anos `< Y`, e em produção vale todo o histórico
(`docs/adr/0003`).

---

## R5 — O ajuste do sensor muda o **nível**, não a probabilidade ✅

Se `nivelAgua ≥ 100 %`, o nível vira CRITICO, mas a probabilidade continua sendo a do modelo
(ex.: 12 %). Mostrar "12 % · CRÍTICO" confunde. Por isso a resposta ganhou:
- `nivelRiscoModelo` (antes do sensor) e `motivosAjuste` (por que subiu);
- `medicaoTransbordando` para o app trocar o número por **"Transbordando agora (medido)"**.

**Para o app:** quando `medicaoTransbordando = true`, esconder a porcentagem.

---

## R6 — Níveis **sem acento** no JSON (`MEDIO`, `CRITICO`) ✅/❓

O documento de contexto fala em "ALTO ou CRITICO", o que sugere que o enum Java não tem acento.
O briefing escreve "MÉDIO/CRÍTICO". O padrão adotado é sem acento (acento em nome de enum Java e
em JSON costuma dar problema de encoding). Se o `RiskLevel` do backend tiver acento, basta
`NIVEL_RISCO_COM_ACENTO=true`. **Confirme no código Java.**

---

## R7 — Serviço sobe em **modo regras** quando não há modelo ✅

Sem `models/flood_lgbm_v1.joblib`, o serviço responde com o baseline de regras,
`origem: "REGRAS"`, `modeloVersao: "regras-v0"` e **`probabilidadeAlagamento: null`**. Assim
o backend (dias 14–15) pode ser integrado **antes** do modelo ficar pronto.

**Para o backend:** o DTO precisa aceitar `null` (`Double`, não `double`).

---

## R8 — O backend **não** deve mandar chuva no fluxo normal ✅/❓

Pela regra de ouro, a chuva usada ao vivo tem que ser calculada pelo mesmo código e da mesma
fonte do treino. Então o **serviço Python é o dono da chuva**:
- fluxo normal: o backend manda só lat/lon + leitura do sensor;
- `chuvaHoraria` (série de 72 h) e `chuva` (resumo) servem para o simulador e para a demo
  (ex.: "chuva forte" num dia seco);
- `chuvaMm` (legado) tem significado ambíguo. Hoje ele só é usado como `chuva_1h` se não
  houver nenhuma outra fonte, e na regra do lixo.

**Decisão pendente:** o que `chuvaMm` significa no backend (chuva da última hora? acumulado?), ou
depreciar o campo.

---

## R9 — Frequência do agendador × atualização do Open-Meteo ✅

O Open-Meteo atualiza **de hora em hora**, então recalcular a cada 10 min não traz chuva nova.
O que muda é a leitura do sensor, e essa já dispara recálculo por leitura. Por isso o serviço tem:
- cache de chuva de **15 min por célula** (~5,5 km). Pontos vizinhos compartilham a chamada;
- `/predict/batch` busca **todas as células numa única chamada** HTTP.

Com isso, 10 min continua ok e a cota gratuita (~10 mil chamadas/dia) fica longe do limite.

---

## R10 — Escolher os pontos só com eventos **até 2022** ✅

Se a escolha dos pontos usar 2024–2025, os pontos já são, por construção, os que alagaram no
período de teste. Isso favorece o modelo no teste (viés de seleção). `ANO_LIMITE_SELECAO_PONTOS`
controla essa escolha.

---

## R11 — Geocodificação do CGE: cuidado com avenidas longas ❓

"Av. Marginal Tietê" geocodificada pelo nome cai no **meio** da avenida, a quilômetros do ponto
real. O `Geocodificador` tenta, em ordem: **cruzamento** com a via de referência, depois o
**lugar de referência**, e só por último o centro da via (`precisao_geocod = logradouro`).
Por padrão, só entram nos pontos e rótulos as precisões `coordenada_oficial` (GeoSampa),
`cruzamento` e `referencia`.

Na página real, muitas referências são **números** ("461", "ALTURA DO N. 66"). Para esses casos
o geocodificador procura o endereço "via, número" (precisão `numero`).

Na 1ª rodada, 50 % dos registros ficaram com boa precisão. A 2ª rodada acrescenta a precisão
`logradouro_curto` (vias de até ~600 m, cujo centro fica dentro do raio R), normaliza as
abreviações coladas do CGE e amplia a busca de avenidas longas (detalhes em
`docs/coleta-dados.md`). O GeoSampa não resolve isso (R16).

**Viés a declarar no TCC:** os descartes se concentram em avenidas longas sem cruzamento
identificável (Marginais, Nações Unidas, Alcântara Machado). Esses corredores ficam
sub-representados nos pontos escolhidos.

---

## R12 — Horários em **UTC** internamente ✅

São Paulo teve horário de verão até fev/2019 (horas repetidas e horas que não existem). Tudo é
guardado em UTC e convertido para hora local só nas features de calendário e na leitura do CGE.

---

## R13 — Sugestões para o backend (Java) ❓

Detalhes em `docs/integracao-backend.md`. Resumo:
- tabela `prediction` com colunas além das do briefing: `probabilidade` **nula**,
  `nivel_risco_modelo`, `hora_referencia`, `ajuste_sensor_aplicado`, `sem_leitura_sensor`,
  `reading_id` (nulo) e índice `(sensor_id, gerada_em)`. Criar via **Flyway**;
- `Alert` ganha `origem` (**PREVISTO** × **MEDIDO**), para separar "vai alagar" de "está
  transbordando", como o app já faz;
- o endpoint público do mapa devolve a última previsão **com status** (`VALIDA`,
  `DESATUALIZADA`, `SEM_PREVISAO`) em vez de esconder o ponto;
- timeout de 2 s no `HttpFloodPredictionGateway` e *circuit breaker* simples (depois de N falhas,
  vai direto para as regras por 1 min).

---

## R14 — Versões de biblioteca iguais no treino e no serviço ✅

O joblib do LightGBM/scikit-learn só é garantido na **mesma versão**. Feito em 20/09/2026:

- `requirements-servico.lock` — versões exatas do serviço (36 pacotes, só o fecho de
  `requirements-servico.txt`). É o que o `Dockerfile` instala;
- `requirements.lock` — o ambiente completo de treino (`pip freeze`), para recriar o `.venv`;
- `metadata_v1.json` registra as versões usadas no treino.

**Regra:** o `.lock` e o `.joblib` andam juntos. Atualizou uma biblioteca → treina de novo e gera
os dois arquivos de novo (`pip freeze > requirements.lock`; o do serviço é o subconjunto).

---

## R15 — Coleta local, com cache e retomada ✅

A coleta roda na máquina de desenvolvimento (`make checar-fontes` primeiro). Todo download fica
em `data/raw` e é retomável: uma queda de rede no meio da raspagem não obriga a baixar de novo.
O HTML do CGE e os nomes das camadas do GeoSampa foram conferidos nas fontes reais em 16/09/2026;
se o site mudar, o extrator avisa quando o total anunciado pela página não bate.

---

## R16 — GeoSampa **não** serve como rótulo (pelo WFS) ✅

Conferido em 16/09/2026: as camadas `risco_ocorrencia_alagamento` (197) e
`risco_ocorrencia_inundacao` (260) têm **só 2026** e **só a data**, sem hora. A série "2013 até
hoje" citada no catálogo não está no WFS. Por isso:
- o **CGE é a única fonte de rótulo horário** (coletado desde 2011);
- as ocorrências do GeoSampa entram com `hora_conhecida = False` e o `dataset.py` as **ignora**
  no rótulo e na frequência. Elas servem para checar se os pontos escolhidos fazem sentido
  (validação espacial de 2026).
- Se a série antiga existir para download no catálogo, dá para reavaliar. Mesmo assim ela não
  tem hora, então só serviria para um rótulo diário (Plano B).

Limitação para o TCC: o rótulo fica restrito ao que a CET registra (vias principais).

---

## R17 — Modelo de chuva fixo (`ecmwf_ifs`) e dataset a partir de **2017** ✅

Conferido em 16/09/2026:
- o `best_match` do Open-Meteo **ao vivo** para São Paulo é o **ECMWF IFS 9 km** (mesma grade e
  mesmos valores de `models=ecmwf_ifs`);
- no **arquivo**, `best_match` é ERA5 até 2016 e IFS a partir de 01/01/2017 (em jan/2017 o ERA5
  dá 290 mm e o IFS dá 344 mm).

Misturar os dois modelos no treino criaria uma quebra artificial na série. Por isso
`OPEN_METEO_MODELO = "ecmwf_ifs"` foi fixado no treino e ao vivo, e o período do dataset passou
a 2017–2025 (treino 2017–2022). Os eventos do CGE de 2011–2016 continuam sendo usados para
escolher os pontos e calcular a frequência histórica.

---

## R18 — Resultado do `v1`: o limite é a chuva, não o modelo ✅/❓

Treino de 17/09/2026 (`python -m src.train --dataset v1 --versao v1 --limiares metas`):

| Teste 2024–25 | PR-AUC | ROC-AUC | ALTO: precisão / recall | Eventos avisados (ALTO) | Antecedência |
|---|---|---|---|---|---|
| regras (baseline) | 0,002 | 0,59 | 0,2 % / 0,3 % | 0,4 % | — |
| regressão logística | 0,011 | 0,92 | 1,6 % / 18 % | 27 % | 2,3 h |
| **LightGBM** | 0,010 | **0,93** | 1,6 % / 22 % | **31 %** | 2,4 h |
| LightGBM, Zona Norte | 0,016 | 0,95 | 2,6 % / 20 % | 29 % | 2,6 h |

Prevalência do teste: 0,061 %. Limiares escolhidos na validação: MEDIO 0,0041 (4,4 % das
horas), ALTO 0,0119 (0,96 %), CRITICO 0,25 (precisão de 50 % inalcançável).

**Leitura:**
- o modelo **ordena bem** (ROC 0,93; nas horas em ALTO o risco é ~25× o normal) e é muito
  melhor que as regras fixas;
- a **precisão por ponto-hora é baixa** (~2 %): 96 % dos ponto-dias com ALTO não têm
  alagamento registrado;
- **a chuva é o gargalo:** em 21 % das horas positivas o IFS 9 km marca < 0,2 mm nas 3 h
  anteriores. Mesmo usando a chuva *observada* das 3 h seguintes (teto teórico,
  `--usar-chuva-prevista --permitir-oraculo`), a PR-AUC só vai de 0,010 para 0,014 e os eventos
  avisados de 31 % para 36 %. Previsão melhor (R1) ajuda pouco com esse modelo de chuva;
- **agregando, o alerta fica útil** (mesmo modelo, maior probabilidade do grupo, teste):

| Unidade do alerta | Prevalência | PR-AUC | 2 % das horas com maior risco: precisão / recall |
|---|---|---|---|
| ponto × hora | 0,06 % | 0,010 | — |
| ponto × dia | 0,5 % | 0,046 | 4 % / 32 % |
| subprefeitura × hora | 0,26 % | 0,038 | 4,5 % / 36 % |
| zona × hora | 1,0 % | 0,091 | 10 % / 40 % |
| cidade × hora | 3,0 % | 0,234 | 33 % / 26 % |

- `dia_semana` e `hora_dia` têm peso alto: parte é chuva de verão à tarde, parte é viés do
  registro (o CGE registra mais em dias úteis, com trânsito). Declarar como limitação.

**Propostas:**
1. **App:** tratar o nível como "risco acima do normal", não como previsão certa; mostrar a
   porcentagem com a comparação ("~25× o normal") ou não mostrar (ligado a R3).
2. **Alerta regional:** além do nível por ponto, um aviso por zona/cidade ("risco elevado de
   alagamentos na Zona Norte nas próximas 3 h"), que acerta bem mais. Não muda o contrato do
   `/predict`; seria um endpoint novo (decisão do grupo — ADR).
3. **Melhor chuva (maior ganho esperado):** os pluviômetros do CGE (`src/collect/cge_pluvio.py`)
   medem a chuva real perto dos pontos. Exige raspagem ao vivo das estações (seção 5.2).
   Alternativa: radar meteorológico, mais trabalhoso.
4. Ajuste de hiperparâmetros do LightGBM: ganho esperado pequeno perto dos itens acima.

**Decidido em 20/09/2026 (item 2):** o alerta de região sai por **subprefeitura**, com a regra
de cobertura mínima de 3 pontos e limiares próprios (ADR 0005). Medido no teste, por
subprefeitura: ALTO acerta 8,0 % e pega 15 % das horas com alagamento; MEDIO acerta 4,8 % e pega
42 %. Por ponto, o ALTO acerta 1,6 %. Detalhe por região em `reports/regioes_v1.csv`.

Implementado: `src/regioes.py`, `GET /predict/regioes` no serviço e
`GET /previsoes/regioes` no backend. O alerta por ponto **continua** sendo a base do mapa e do
desvio de rota; a região é camada de contexto.

**Decisão pendente:** item 3 (pluviômetros do CGE).

---

## Trabalho futuro (para o TCC)

- Os relatos do cidadão no app (o design já prevê a tela) podem virar **rótulo** melhor que o
  proxy atual. Para isso falta criar o domínio `Ocorrencia` no backend.
- Com leituras reais acumuladas, retreinar com o rótulo verdadeiro (sensor saturado) e com
  `nivelAgua` como feature aprendida.