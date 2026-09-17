# ADR 0003 — Estratégia de treino sem vazamento

- **Status:** Aceito
- **Data:** 16/09/2026

## Decisão
1. **Divisão temporal**: treino ≤ 2022, validação 2023, teste 2024–2025 (ajustável).
2. **Subamostragem só no treino** (horas secas negativas, 7 %, `peso_amostra = 1/taxa`);
   validação e teste completos.
3. **Calibração e limiares** escolhidos só na validação; o teste é usado uma vez, no fim.
4. **`freq_hist_alagamento`** por ano com janela expansiva (só anos anteriores à linha);
   produção usa todo o histórico.
5. **Pontos** escolhidos só com eventos até o fim do treino.
6. **`chuva_prevista_3h`** proibida com a reanálise (seria chuva futura observada);
   permitida só com previsões históricas.
7. Treino com a **cidade inteira**; métricas também recortadas para a Zona Norte.

## Consequências
Métricas mais baixas do que uma divisão aleatória daria, mas honestas e defensáveis na banca.
