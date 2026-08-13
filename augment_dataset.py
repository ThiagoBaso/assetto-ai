"""
augment_dataset.py

Gera amostras sinteticas de "recuperacao" a partir do dataset original,
simulando o carro deslocado da linha ideal (lateral_position e heading_error
perturbados) com a acao de steering corrigida para trazer o carro de volta
ao centro da pista.

Isso ataca o problema de distribution shift do Behavioral Cloning: o dataset
original so tem exemplos de pilotagem quase perfeita, entao o modelo nunca
aprendeu "o que fazer quando ja estou fora da linha". Aqui geramos esses
exemplos artificialmente, sem precisar coletar dados novos no simulador.

Ajuste os nomes de coluna no bloco CONFIG se forem diferentes dos seus.
"""

import numpy as np
import pandas as pd

# ============================== CONFIG ==============================

INPUT_CSV = "data/processed/dataset_with_track.csv"
OUTPUT_CSV = "data/processed/dataset_with_track_augmented.csv"

COL_LATERAL_POS = "lateral_position"   # metros, com sinal (- esquerda / + direita, ajuste conforme sua convencao)
COL_HEADING_ERR = "heading_error"      # radianos
COL_STEERING = "steering_input"        # coluna de acao real no dataset (-1 a 1)
COL_SPEED = "speed"                    # ajuste para o nome real de velocidade

# Quantas amostras perturbadas gerar por amostra original.
# Comece baixo (1-2) para nao explodir o tamanho do dataset e o tempo de treino.
N_AUGMENTED_PER_ROW = 2

# Faixas de perturbacao. Perturbar tanto pra esquerda quanto pra direita.
LATERAL_OFFSET_RANGE_M = (0.3, 1.2)     # desvio lateral simulado em metros
HEADING_OFFSET_RANGE_RAD = (0.05, 0.25) # desvio angular simulado (~3 a 14 graus)

# Ganho da correcao: quanto mais fora da linha, mais forte a correcao de steering.
# Sao dois termos somados: um proporcional ao erro lateral, outro ao erro de heading.
# Esses ganhos sao um ponto de partida - ajuste observando o resultado no autopilot.
K_LATERAL = 0.6   # steering adicional por metro de desvio lateral
K_HEADING = 0.8   # steering adicional por radiano de erro de heading

# Limite fisico de steering
STEER_MIN, STEER_MAX = -1.0, 1.0

SEED = 42

# ======================================================================


def perturb_row(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Gera uma versao perturbada de uma linha do dataset."""
    new_row = row.copy()

    # Sorteia o lado do desvio (esquerda/direita) e a magnitude dentro da faixa.
    side = rng.choice([-1.0, 1.0])
    lateral_mag = rng.uniform(*LATERAL_OFFSET_RANGE_M)
    heading_mag = rng.uniform(*HEADING_OFFSET_RANGE_RAD)

    lateral_offset = side * lateral_mag
    heading_offset = side * heading_mag

    original_lateral = row[COL_LATERAL_POS]
    original_heading = row[COL_HEADING_ERR]

    new_lateral = original_lateral + lateral_offset
    new_heading = original_heading + heading_offset

    new_row[COL_LATERAL_POS] = new_lateral
    new_row[COL_HEADING_ERR] = new_heading

    # Acao de correcao: soma a acao original (o que o piloto faria idealmente
    # naquele trecho) com um termo de correcao proporcional ao quao fora o
    # carro esta. O sinal negativo de "side" e proposital: se o carro foi
    # deslocado para a direita (side > 0), a correcao deve puxar para a
    # esquerda (steering negativo), e vice-versa. Ajuste o sinal se sua
    # convencao de steering for invertida.
    correction = -side * (K_LATERAL * lateral_mag + K_HEADING * heading_mag)

    new_steer = row[COL_STEERING] + correction
    new_row[COL_STEERING] = float(np.clip(new_steer, STEER_MIN, STEER_MAX))

    return new_row


def main():
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(INPUT_CSV)
    print(f"Dataset original: {len(df)} linhas")

    for col in (COL_LATERAL_POS, COL_HEADING_ERR, COL_STEERING):
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' nao encontrada no CSV. "
                f"Ajuste o bloco CONFIG com os nomes reais das colunas. "
                f"Colunas disponiveis: {list(df.columns)}"
            )

    augmented_rows = []
    for _, row in df.iterrows():
        for _ in range(N_AUGMENTED_PER_ROW):
            augmented_rows.append(perturb_row(row, rng))

    augmented_df = pd.DataFrame(augmented_rows)

    # Marca a origem de cada linha, para poder filtrar/pesar depois no treino
    # (ex: dar peso menor as sinteticas se notar que estao prejudicando o
    # comportamento em reta).
    df["is_synthetic"] = False
    augmented_df["is_synthetic"] = True

    final_df = pd.concat([df, augmented_df], ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Amostras sinteticas geradas: {len(augmented_df)}")
    print(f"Dataset final: {len(final_df)} linhas -> salvo em {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
