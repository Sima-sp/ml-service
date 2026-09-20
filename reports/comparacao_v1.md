# Comparação de modelos — v1

Dataset `v1` (chuva: arquivo). Treino ≤ 2022 · validação [2023] · teste [2024, 2025].

## validacao

| modelo | pr_auc | roc_auc | brier | prevalencia | precisao_ALTO | recall_ALTO | recall_eventos | antecedencia_h | taxa_alarme_falso |
|---|---|---|---|---|---|---|---|---|---|
| baseline_regras | 0.0026 | 0.5932 |  | 0.000879 | 0.0167 | 0.0066 | 0.0209 | 2.43 | 0.965 |
| logistica | 0.0187 | 0.8872 | 0.000868 | 0.000879 | 0.0238 | 0.2343 | 0.3284 | 2.37 | 0.9518 |
| lightgbm | 0.0227 | 0.9028 | 0.000866 | 0.000879 | 0.0228 | 0.2495 | 0.3373 | 2.38 | 0.9514 |

## teste

| modelo | pr_auc | roc_auc | brier | prevalencia | precisao_ALTO | recall_ALTO | recall_eventos | antecedencia_h | taxa_alarme_falso |
|---|---|---|---|---|---|---|---|---|---|
| baseline_regras | 0.0017 | 0.5892 |  | 0.000613 | 0.0019 | 0.0027 | 0.0041 | 2.0 | 0.9756 |
| logistica | 0.0111 | 0.9219 | 0.00061 | 0.000613 | 0.0163 | 0.1773 | 0.2712 | 2.31 | 0.9613 |
| lightgbm | 0.0104 | 0.9324 | 0.000612 | 0.000613 | 0.016 | 0.2154 | 0.3126 | 2.4 | 0.9625 |

## teste_zona_norte

| modelo | pr_auc | roc_auc | brier | prevalencia | precisao_ALTO | recall_ALTO | recall_eventos | antecedencia_h | taxa_alarme_falso |
|---|---|---|---|---|---|---|---|---|---|
| baseline_regras | 0.001 | 0.5838 |  | 0.000409 | 0.0 | 0.0 | 0.0 |  | 1.0 |
| logistica | 0.0122 | 0.9298 | 0.000406 | 0.000409 | 0.0276 | 0.1269 | 0.1955 | 2.46 | 0.9561 |
| lightgbm | 0.0162 | 0.9454 | 0.000405 | 0.000409 | 0.0259 | 0.198 | 0.2857 | 2.55 | 0.9488 |
