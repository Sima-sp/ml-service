# ADR 0002 — Fonte de chuva, células e convenção de tempo

- **Status:** Aceito — modelo conferido em 16/09/2026
- **Data:** 16/09/2026

## Contexto
A regra de ouro exige a mesma fonte e a mesma definição de feature no treino e ao vivo.
O Open-Meteo oferece arquivo (reanálise), previsões históricas e previsão ao vivo, todos sem
chave. Documentos mais antigos citavam a WeatherAPI; o briefing trocou pelo Open-Meteo.

## Decisão
1. **Open-Meteo** é a fonte de chuva no treino e ao vivo; o CGE entra só como melhoria, se
   houver versão ao vivo.
2. A chuva é pedida no **centro de células de 0,05°**, tanto no treino quanto ao vivo. Pontos da
   mesma célula compartilham a série e o cache.
3. Tudo em **UTC** (`timezone=GMT`, `timeformat=unixtime`); a hora local só entra no calendário
   e na leitura do CGE (horário de verão até 2019).
4. **Hora de referência** = hora cheia anterior ao pedido. O valor do Open-Meteo em `T` é a
   chuva de `T−1h` a `T`.
5. Modelo fixo `models=ecmwf_ifs` (ECMWF IFS 9 km) no treino e ao vivo. Conferido: o
   `best_match` ao vivo para São Paulo devolve exatamente a mesma grade e os mesmos valores; o
   arquivo tem esse modelo a partir de 01/01/2017 (antes é ERA5). Por isso o dataset começa em 2017.

## Consequências
- A resolução espacial fica limitada à do modelo meteorológico (limitação declarada).
- A troca de fonte exige gerar outro dataset (`--fonte-chuva`), registrado no resumo e nos metadados.
