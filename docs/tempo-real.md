# Tempo real — como uma previsão chega ao app

```
┌─────────────┐  a cada 10 min (@Scheduled)   ┌───────────────────────┐
│  Backend    │ ───────────────────────────►  │ POST /predict/batch    │
│  Spring     │  ou leitura nova do sensor    │ (Python, porta 5000)   │
│             │ ───────────────────────────►  │ POST /predict          │
└──────┬──────┘                               └───────────┬───────────┘
       │ salva Prediction (fonte=IA)                      │ 1. células vencidas → 1 chamada Open-Meteo
       │ falhou/timeout 2 s → regras (fonte=REGRAS)       │ 2. features (src/features.py)
       ▼                                                  │ 3. modelo + calibração → nível
┌─────────────┐   GET /api/public/mapa-risco              │ 4. ajuste do sensor
│  MySQL      │ ◄──────────────── App (polling 1–2 min)   │ 5. geradaEm / validaAte (+30 min)
└─────────────┘                                           ▼
```

## Hora de referência
Previsão pedida às 14:20 → `horaReferencia = 14:00`. O valor de chuva "14:00" do Open-Meteo é o
acumulado de 13:00–14:00, o último intervalo completo. O dataset usa exatamente essa convenção.
Se a hora 14:00 ainda vier vazia, o serviço usa 13:00 e avisa.

## Cache de chuva
- Chave: célula de 0,05° (`src/collect/open_meteo.py::id_celula`), a mesma do treino.
- Validade (`CACHE_CHUVA_TTL_MIN`): 15 min. O Open-Meteo atualiza de hora em hora.
- O lote pede **todas** as células vencidas numa única chamada HTTP.
- Estimativa de cota: ~20 células × 1 chamada a cada 15 min ≈ 100 chamadas/dia, bem abaixo do
  limite gratuito (~10 mil/dia).

## Falhas e o que o app mostra

| Situação | Serviço | Backend | App |
|---|---|---|---|
| Tudo ok | 200, `fonteChuva=OPEN_METEO` | salva `fonte=IA` | "72 % · ALTO · previsto às 14:20 · próximas 3 h" |
| Open-Meteo fora, cache < 60 min | 200, `fonteChuva=CACHE` + aviso | salva `fonte=IA` | igual (pode mostrar "chuva de 40 min atrás") |
| Open-Meteo fora, sem cache | 503 `CHUVA_INDISPONIVEL` | regras, `fonte=REGRAS` | nível pelas regras, sem porcentagem |
| Serviço fora / timeout | — | regras, `fonte=REGRAS` | idem |
| Sem modelo treinado | 200, `origem=REGRAS`, prob. `null` | salva `fonte=REGRAS` | idem |
| Sensor INACTIVE/MAINTENANCE | 200, `semLeituraSensor=true` | envia sem água/lixo | "sem leitura do sensor" + previsão só por chuva |
| `nivelAgua ≥ 100` | 200, `medicaoTransbordando=true`, CRITICO | alerta MEDIDO | "Transbordando agora (medido)", sem porcentagem |
| Agora > `validaAte` | — | endpoint devolve `DESATUALIZADA` | "previsão desatualizada" no lugar do número |

## Parâmetros (variáveis de ambiente)

| Variável | Padrão | Decisão em aberto? |
|---|---|---|
| `VALIDADE_PREVISAO_MIN` | 30 | sim (briefing §17) |
| `CACHE_CHUVA_TTL_MIN` | 15 | — |
| `IDADE_MAX_CHUVA_MIN` | 60 | — |
| `TIMEOUT_OPEN_METEO_S` | 5 | — (o backend espera 2 s; com cache quente a resposta leva ms) |
| `RAIO_PONTO_CONHECIDO_M` | 500 | — |

> Atenção ao timeout: o backend espera 2 s, mas a primeira chamada com o cache frio pode
> levar mais que isso se o Open-Meteo estiver lento. O agendador deve usar um timeout maior
> para o `/predict/batch` (ex.: 10 s), e a chamada síncrona por leitura fica com 2 s. Como o
> agendador aquece o cache, a chamada por leitura quase sempre responde em milissegundos.
