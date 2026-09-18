# Comparação de modelos — v1-metas

Dataset `v1` (chuva: arquivo). Treino ≤ 2022 · validação [2023] · teste [2024, 2025].

## validacao

| modelo | pr_auc | roc_auc | brier | prevalencia | precisao_ALTO | recall_ALTO | recall_eventos | antecedencia_h | taxa_alarme_falso |
|---|---|---|---|---|---|---|---|---|---|
| baseline_regras | 0.0026 | 0.5932 |  | 0.000879 | 0.0167 | 0.0066 | 0.0209 | 2.43 | 0.965 |
| logistica | 0.0187 | 0.8872 | 0.000868 | 0.000879 | 0.0157 | 0.297 | 0.406 | 2.32 | 0.9631 |
| lightgbm | 0.0227 | 0.9028 | 0.000866 | 0.000879 | 0.022 | 0.26 | 0.3522 | 2.37 | 0.9536 |

## teste

| modelo | pr_auc | roc_auc | brier | prevalencia | precisao_ALTO | recall_ALTO | recall_eventos | antecedencia_h | taxa_alarme_falso |
|---|---|---|---|---|---|---|---|---|---|
| baseline_regras | 0.0017 | 0.5892 |  | 0.000613 | 0.0019 | 0.0027 | 0.0041 | 2.0 | 0.9756 |
| logistica | 0.0111 | 0.9219 | 0.00061 | 0.000613 | 0.0128 | 0.2731 | 0.4099 | 2.29 | 0.9658 |
| lightgbm | 0.0104 | 0.9324 | 0.000612 | 0.000613 | 0.0154 | 0.2276 | 0.3333 | 2.43 | 0.9636 |

## teste_zona_norte

| modelo | pr_auc | roc_auc | brier | prevalencia | precisao_ALTO | recall_ALTO | recall_eventos | antecedencia_h | taxa_alarme_falso |
|---|---|---|---|---|---|---|---|---|---|
| baseline_regras | 0.001 | 0.5838 |  | 0.000409 | 0.0 | 0.0 | 0.0 |  | 1.0 |
| logistica | 0.0122 | 0.9298 | 0.000406 | 0.000409 | 0.0161 | 0.1954 | 0.3158 | 2.19 | 0.968 |
| lightgbm | 0.0162 | 0.9454 | 0.000405 | 0.000409 | 0.0249 | 0.2107 | 0.3083 | 2.59 | 0.9532 |
