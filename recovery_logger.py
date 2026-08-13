"""
recovery_logger.py

Modulo simples para logar, a cada frame do autopilot, os sensores de pista
(s_current, lateral_position, heading_error, curvature_*, speed_error) junto
com o estado de saida de pista (tyres_out, surface_grip).

Objetivo: confirmar (ou descartar) a hipotese de que a busca do ponto mais
proximo (usando janela em torno de s_previous) trava num ponto errado da
trajetoria quando o carro ja desviou bastante, alimentando o modelo com
curvature/heading_error errados e piorando o desvio em vez de corrigi-lo.

COMO USAR:
1. Copie este arquivo para a pasta do projeto.
2. No autopilot.py, importe e instancie o logger uma vez, antes do loop
   principal:

       from recovery_logger import RecoveryLogger
       logger = RecoveryLogger("recovery_log.csv")

3. Dentro do loop principal, depois de calcular os sensores de pista e
   antes (ou depois) de aplicar as acoes, chame:

       logger.log(
           s_current=s_current,
           lateral_position=lateral_position,
           heading_error=heading_error,
           curvature_0m=curvature_0m,
           curvature_10m=curvature_10m,
           speed_error=speed_error,
           tyres_out=tyres_out,
           surface_grip=surface_grip,
           steer_pred=steer_pred,
           throttle_pred=throttle_pred,
           brake_pred=brake_pred,
       )

   Ajuste os nomes dos argumentos para bater com as variaveis reais do seu
   autopilot.py (todos sao opcionais, so loga o que voce passar).

4. Rode uma sessao onde o carro saia da pista pelo menos uma vez.

5. Feche o programa (ou chame logger.close()) e rode analyze_recovery_log.py
   apontando para o CSV gerado.
"""

import csv
import time
from pathlib import Path


class RecoveryLogger:
    def __init__(self, filepath: str = "recovery_log.csv"):
        self.filepath = Path(filepath)
        self._file = open(self.filepath, "w", newline="")
        self._writer = None  # criado no primeiro log(), para descobrir as colunas dinamicamente
        self._t0 = time.time()

    def log(self, **fields):
        fields["t"] = time.time() - self._t0

        if self._writer is None:
            fieldnames = list(fields.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            self._writer.writeheader()

        self._writer.writerow(fields)

    def close(self):
        self._file.flush()
        self._file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
