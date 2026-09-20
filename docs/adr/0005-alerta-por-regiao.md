# ADR 0005 — Alerta por subprefeitura, além do alerta por ponto

- **Status:** Aceito
- **Data:** 20/09/2026

## Contexto

O modelo `v1` separa bem hora de risco de hora normal (ROC-AUC 0,93 no teste), mas errar por
poucos quarteirões estraga o acerto no ponto: no nível ALTO, cerca de **1,6 %** dos ponto-horas
alertados terminam em alagamento registrado. A causa é a resolução da chuva — o ECMWF IFS tem
grade de 9 km e enxerga "pancada forte sobre a Zona Norte", não "vai acumular água nesta rua".
O teto foi medido: mesmo com a chuva observada das 3 h seguintes (oráculo), a PR-AUC sobe de
0,010 para 0,014 (R18).

Lido numa escala maior, o mesmo modelo acerta bem mais. No conjunto de teste, agregando pelo
**maior risco entre os pontos** de cada região:

| Unidade | Precisão do alerta | Alagamentos que o alerta pega |
|---|---|---|
| ponto × hora (ALTO) | 1,6 % | 22 % |
| subprefeitura × hora (ALTO) | **8,0 %** | 15 % |
| subprefeitura × hora (MEDIO) | 4,8 % | **42 %** |
| zona × hora | ~10 % | ~40 % |

## Decisão

1. **A unidade do alerta de região é a subprefeitura** (32 oficiais, camada `subprefeitura` do
   GeoSampa), e não a zona nem o distrito:
   - é como o **próprio CGE** agrupa os registros, então rótulo e alerta falam da mesma coisa;
   - os nomes são os que as pessoas usam ("Vila Maria / Vila Guilherme");
   - zona acerta um pouco mais, mas "Zona Norte" é grande demais para ser acionável;
   - distrito (96) deixaria a cobertura rala: 137 pontos não representam 96 regiões.
2. **Regra de cobertura:** região com menos de `MIN_PONTOS_REGIAO` (3) pontos monitorados não
   recebe nível; devolve `cobertura = "SEM_COBERTURA"`. "Sem cobertura" **não** é "sem risco", e
   a tela precisa dizer isso.
3. **Limiares próprios**, escolhidos na validação com orçamento de alertas mais apertado que o
   dos pontos (≤ 5 % das horas em ≥MEDIO, ≤ 2 % em ≥ALTO), em
   `models/thresholds_regiao_<versao>.json`. Sem esse arquivo, o endpoint responde 503 em vez de
   usar limiar de outra escala.
4. **Endpoint novo** `GET /predict/regioes` no serviço; o backend repassa em
   `GET /previsoes/regioes` com cache curto. O contrato de `/predict` **não muda**.
5. **O alerta por ponto continua sendo a base do mapa e do desvio de rota.** O alerta de região
   é uma camada de contexto, não um substituto.

## Consequências

- O app ganha um aviso que a pessoa entende ("risco elevado na Vila Maria nas próximas 3 h") com
  acerto cinco vezes maior que o do ponto.
- Seis das 32 subprefeituras não têm nenhum ponto monitorado e outras têm um ou dois; elas
  aparecem como sem cobertura. Isso é honesto, mas precisa de texto claro na tela.
- Duas escalas de limiar para manter (ponto e região). O treino grava as duas.
- A avaliação por região entra no TCC como tabela (`reports/regioes_v1.csv`), incluindo as
  regiões em que o alerta funciona mal — Pirituba/Jaraguá e Jaçanã/Tremembé quase não disparam.
