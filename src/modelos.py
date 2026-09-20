"""Fábricas dos modelos comparados no TCC (seção 7.2).

1. ``baseline_regras``  -> ``src/baseline.py`` (sem treino)
2. ``logistica``        -> regressão logística com padronização (e log1p na chuva)
3. ``lightgbm``         -> modelo principal esperado (gradient boosting em árvores)
4. ``random_forest``    -> alternativa ao LightGBM
5. LSTM (extra)         -> ``notebooks/05_lstm_extra.ipynb``

Desbalanceamento: nenhum modelo usa ``class_weight``. O treino recebe ``peso_amostra`` (que só
desfaz a subamostragem das horas secas) e a escala de probabilidade é corrigida depois, na
calibração. Assim a comparação entre modelos fica justa e a porcentagem exibida é honesta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import SEMENTE
from src.features import COLUNAS_CATEGORICAS, COLUNAS_CHUVA, COLUNAS_CHUVA_PREVISTA


def categoria_para_texto(X):
    """Converte colunas categóricas para texto, com 'DESCONHECIDA' no lugar de vazio.

    Função no nível do módulo para que o pipeline possa ser salvo com joblib.
    """
    return pd.DataFrame(X).astype("object").fillna("DESCONHECIDA").astype(str)


def _separar(features: list[str]) -> tuple[list[str], list[str], list[str]]:
    chuva = [f for f in features if f in COLUNAS_CHUVA + COLUNAS_CHUVA_PREVISTA and f != "horas_desde_ultima_chuva"]
    categoricas = [f for f in features if f in COLUNAS_CATEGORICAS]
    outras = [f for f in features if f not in chuva and f not in categoricas]
    return chuva, outras, categoricas


def _preprocessador(features: list[str], escalar: bool) -> ColumnTransformer:
    chuva, outras, categoricas = _separar(features)
    passos_chuva = [("imputar", SimpleImputer(strategy="median")),
                    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))]
    passos_outras = [("imputar", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:
        passos_chuva.append(("escalar", StandardScaler()))
        passos_outras.append(("escalar", StandardScaler()))
    return ColumnTransformer([
        ("chuva", Pipeline(passos_chuva), chuva),
        ("numericas", Pipeline(passos_outras), outras),
        ("categoricas", Pipeline([
            ("texto", FunctionTransformer(categoria_para_texto, feature_names_out="one-to-one")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]), categoricas),
    ])


def criar_regressao_logistica(features: list[str]) -> Pipeline:
    """Regressão logística: log1p na chuva, padronização, one-hot na zona, NaN -> mediana."""
    return Pipeline([
        ("preparo", _preprocessador(features, escalar=True)),
        ("modelo", LogisticRegression(C=1.0, max_iter=3000)),
    ])


def criar_random_forest(features: list[str]) -> Pipeline:
    """Random Forest de comparação (o principal é o LightGBM).

    ``max_samples`` e o número de árvores são menores que o usual de propósito: com ~4,5 milhões
    de linhas de treino, a floresta completa não cabe na memória de uma máquina comum e leva
    horas. Cada árvore vê 30 % das linhas sorteadas, que é suficiente para o papel dela aqui —
    servir de referência na tabela de comparação do TCC.
    """
    return Pipeline([
        ("preparo", _preprocessador(features, escalar=False)),
        ("modelo", RandomForestClassifier(
            n_estimators=150, min_samples_leaf=50, max_features="sqrt", max_samples=0.3,
            n_jobs=-1, random_state=SEMENTE)),
    ])


PARAMETROS_LIGHTGBM = {
    "objective": "binary",
    "metric": "average_precision",  # PR-AUC na validação (early stopping)
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEMENTE,
    "verbose": -1,
}


def criar_lightgbm(**ajustes):
    """LightGBM com os parâmetros padrão acima (qualquer um pode ser sobrescrito)."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**{**PARAMETROS_LIGHTGBM, **ajustes})


def treinar_lightgbm(modelo, X_treino, y_treino, pesos_treino, X_val, y_val, paciencia: int = 150):
    """Treina com parada antecipada pela PR-AUC da validação."""
    import inspect

    import lightgbm as lgb

    avaliacao = {"eval_names": ["validacao"]}
    if "eval_X" in inspect.signature(modelo.fit).parameters:  # LightGBM >= 4.7
        avaliacao.update(eval_X=(X_val,), eval_y=(y_val,))
    else:
        avaliacao.update(eval_set=[(X_val, y_val)])
    modelo.fit(
        X_treino, y_treino, sample_weight=pesos_treino, **avaliacao,
        callbacks=[lgb.early_stopping(paciencia, verbose=False), lgb.log_evaluation(0)],
    )
    return modelo


def pontuacao(modelo, X) -> np.ndarray:
    """Probabilidade bruta (antes da calibração) da classe positiva."""
    return modelo.predict_proba(X)[:, 1]


def importancia_lightgbm(modelo, features: list[str]) -> pd.DataFrame:
    """Importância por ganho (quanto cada feature reduziu a perda), normalizada."""
    booster = modelo.booster_
    ganho = booster.feature_importance(importance_type="gain")
    divisoes = booster.feature_importance(importance_type="split")
    df = pd.DataFrame({"feature": features, "ganho": ganho, "divisoes": divisoes})
    df["ganho_pct"] = (100 * df["ganho"] / df["ganho"].sum()).round(2)
    return df.sort_values("ganho", ascending=False).reset_index(drop=True)
