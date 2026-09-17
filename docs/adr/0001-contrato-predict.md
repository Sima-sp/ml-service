# ADR 0001 — Contrato do `POST /predict` (v2, aditivo)

- **Status:** Proposto (aguarda aprovação no backend)
- **Data:** 16/09/2026

## Contexto
O backend chama o serviço com `nivelAgua, porcentagemLixo, chuvaMm, latitude, longitude` e
espera `probabilidadeAlagamento, nivelRisco`. O briefing (seção 9) acrescenta `sensorId`,
`timestamp`, `chuva`, validade e transparência do ajuste.

## Decisão
1. Todos os campos novos são **opcionais** na entrada e **aditivos** na saída; o payload antigo
   continua válido.
2. Entrada nova: `sensorId`, `timestamp`, `chuva` (resumo) e `chuvaHoraria` (série ≥ 72 h,
   para simulador/demo).
3. Saída nova: `janelaHoras`, `geradaEm`, `validaAte`, `horaReferencia`,
   `medicaoTransbordando`, `ajusteSensorAplicado`, `motivosAjuste`, `semLeituraSensor`,
   `fonteChuva`, `chuvaRecente3hMm`, `chuvaPrevista3hMm`, `nivelRiscoModelo`, `origem`,
   `modeloVersao`, `pontoId`, `distanciaPontoM`, `avisos`.
4. `probabilidadeAlagamento` pode ser **null** (`origem = REGRAS`).
5. Níveis **sem acento** (`MEDIO`, `CRITICO`); acento configurável.
6. Erros com corpo `{codigo, mensagem}` e códigos estáveis (`CHUVA_INDISPONIVEL` = 503).
7. `POST /predict/batch` recebe uma **lista** e devolve uma lista na mesma ordem, com
   `sucesso/previsao/erro` por item.

## Consequências
- O backend precisa aceitar campos desconhecidos e `probabilidade` nula.
- O app ganha o que precisa para separar Medido × Previsto e mostrar "desatualizada".
