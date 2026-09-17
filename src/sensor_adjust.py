"""Ajuste do nível de risco pela leitura do sensor — **regra, não treinada** (seção 7.5).

Por que regra? Não existe histórico de ``nivelAgua``/``porcentagemLixo`` para treinar: os
sensores são novos (e, no TCC, simulados). O modelo aprende só com chuva + características do
ponto; o sensor entra depois, por uma regra transparente e testada.

Regras (valores em ``src/config.py`` → ``AJUSTE_SENSOR``):

1. ``nivelAgua >= 100 %``  → **medição** "transbordando agora": nível CRITICO e
   ``medicao_transbordando = True``. Isso NÃO é previsão; o app deve mostrar como "Medido".
2. ``nivelAgua >= 80 %``   → sobe um nível.
3. ``porcentagemLixo >= 60 %`` **e** chuva recente/prevista acima de ``chuva_minima_mm``
   → sobe um nível (lixo sem chuva não alaga).
4. Com ``acumular=True``, as regras 2 e 3 somam (podem subir dois níveis).
5. Nunca desce abaixo do nível do modelo; teto CRITICO.
6. Sem leitura (sensor INACTIVE/MAINTENANCE ou campos ausentes): nenhum ajuste e
   ``sem_leitura = True`` (o app mostra "sem leitura do sensor").

A probabilidade do modelo **não é alterada** — só o nível. Ver docs/recomendacoes.md (R5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import AJUSTE_SENSOR
from src.niveis import NivelRisco, maior_nivel, subir_nivel


@dataclass
class ResultadoAjuste:
    nivel_final: NivelRisco
    nivel_modelo: NivelRisco
    medicao_transbordando: bool = False
    ajuste_aplicado: bool = False
    sem_leitura: bool = False
    motivos: list[str] = field(default_factory=list)


def ajustar_por_sensor(
    nivel_modelo: NivelRisco,
    nivel_agua: float | None,
    porcentagem_lixo: float | None,
    chuva_recente_mm: float | None = None,
    chuva_prevista_mm: float | None = None,
    regras: dict = AJUSTE_SENSOR,
) -> ResultadoAjuste:
    """Aplica as regras do sensor sobre o nível do modelo (ver docstring do módulo)."""
    resultado = ResultadoAjuste(nivel_final=nivel_modelo, nivel_modelo=nivel_modelo)
    if nivel_agua is None and porcentagem_lixo is None:
        resultado.sem_leitura = True
        return resultado

    if nivel_agua is not None and nivel_agua >= regras["nivel_agua_transbordando"]:
        resultado.nivel_final = NivelRisco.CRITICO
        resultado.medicao_transbordando = True
        resultado.ajuste_aplicado = nivel_modelo != NivelRisco.CRITICO
        resultado.motivos.append(
            f"nivelAgua {nivel_agua:.0f}% >= {regras['nivel_agua_transbordando']:.0f}%: transbordando agora (medido)"
        )
        return resultado

    passos = 0
    if nivel_agua is not None and nivel_agua >= regras["nivel_agua_alerta"]:
        passos += 1
        resultado.motivos.append(f"nivelAgua {nivel_agua:.0f}% >= {regras['nivel_agua_alerta']:.0f}%")

    chuva = max(chuva_recente_mm or 0.0, chuva_prevista_mm or 0.0)
    if (porcentagem_lixo is not None and porcentagem_lixo >= regras["lixo_alerta"]
            and chuva > regras["chuva_minima_mm"]):
        passos += 1
        resultado.motivos.append(
            f"porcentagemLixo {porcentagem_lixo:.0f}% >= {regras['lixo_alerta']:.0f}% com chuva {chuva:.1f} mm"
        )

    if not regras.get("acumular", True):
        passos = min(passos, 1)
    if passos:
        ajustado = maior_nivel(nivel_modelo, subir_nivel(nivel_modelo, passos))
        resultado.nivel_final = ajustado
        resultado.ajuste_aplicado = ajustado != nivel_modelo
    return resultado
