"""
augment_dataset.py

Gera amostras sintéticas de "recuperação" a partir do dataset original,
simulando o carro deslocado da linha ideal (lateral_position e heading_error
perturbados) com a ação de steering corrigida para trazer o carro de volta
ao centro da pista.

Este script AUMENTA APENAS AS VOLTAS DE TREINO, mantendo validação e teste
intactos. As amostras sintéticas são inseridas logo após a original, para
preservar a ordem temporal dentro de cada volta.

Divisão automática:
- 70% das voltas para treino
- 15% para validação
- 15% para teste
"""

import numpy as np
import pandas as pd

# ============================== CONFIG ==============================

INPUT_CSV = "data/processed/dataset_with_track.csv"
OUTPUT_CSV = "data/processed/dataset_with_track_augmented.csv"

COL_LATERAL_POS = "lateral_position"
COL_HEADING_ERR = "heading_error"
COL_STEERING = "steering_input"
COL_SPEED = "speed"

N_AUGMENTED_PER_ROW = 1            # quantas amostras sintéticas por linha original

LATERAL_OFFSET_RANGE_M = (0.3, 1.2)
HEADING_OFFSET_RANGE_RAD = (0.05, 0.25)

K_LATERAL = 0.3
K_HEADING = 0.4

STEER_MIN, STEER_MAX = -1.0, 1.0
SEED = 42

# ======================================================================


def perturb_row(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Gera uma versão perturbada de uma linha do dataset."""
    new_row = row.copy()

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

    # Correção de steering: puxa para o lado oposto ao desvio
    correction = -side * (K_LATERAL * lateral_mag + K_HEADING * heading_mag)
    new_steer = row[COL_STEERING] + correction
    new_row[COL_STEERING] = float(np.clip(new_steer, STEER_MIN, STEER_MAX))

    return new_row


def main():
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(INPUT_CSV)
    print(f"Dataset original: {len(df)} linhas")

    # Verificar colunas necessárias
    for col in (COL_LATERAL_POS, COL_HEADING_ERR, COL_STEERING):
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não encontrada no CSV. "
                f"Ajuste o bloco CONFIG. Colunas disponíveis: {list(df.columns)}"
            )

    # Detectar voltas únicas
    unique_laps = sorted(df['lap'].unique())
    print(f"Voltas encontradas: {unique_laps}")

    # Divisão proporcional
    n_laps = len(unique_laps)
    n_train = int(np.floor(0.70 * n_laps))
    n_val = int(np.floor(0.15 * n_laps))
    n_test = n_laps - n_train - n_val

    train_laps = unique_laps[:n_train]
    val_laps = unique_laps[n_train:n_train+n_val]
    test_laps = unique_laps[n_train+n_val:]

    print(f"Laps de treino: {train_laps}")
    print(f"Laps de validação: {val_laps}")
    print(f"Laps de teste: {test_laps}")

    # Separar os DataFrames
    df_train = df[df['lap'].isin(train_laps)].copy()
    df_val_test = df[df['lap'].isin(val_laps + test_laps)].copy()

    # Adicionar coluna is_synthetic
    df_train['is_synthetic'] = False
    df_val_test['is_synthetic'] = False

    # Lista para armazenar DataFrames (mais seguro que listas de Series)
    train_parts = []

    # Processar cada volta de treino
    for lap in train_laps:
        df_lap = df_train[df_train['lap'] == lap]
        print(f"Processando volta {lap} ({len(df_lap)} linhas originais)...")

        # Lista de linhas sintéticas para esta volta
        synth_rows = []
        for _, row in df_lap.iterrows():
            for _ in range(N_AUGMENTED_PER_ROW):
                synth = perturb_row(row, rng)
                synth['is_synthetic'] = True
                synth_rows.append(synth)

        if synth_rows:
            df_synth = pd.DataFrame(synth_rows)
            # Intercala original e sintética na ordem correta
            # Para cada linha original, seguem N_AUGMENTED_PER_ROW sintéticas
            combined = []
            for i, (_, orig) in enumerate(df_lap.iterrows()):
                combined.append(orig)
                for j in range(N_AUGMENTED_PER_ROW):
                    combined.append(synth_rows[i * N_AUGMENTED_PER_ROW + j])
            # Converte para DataFrame
            df_combined = pd.DataFrame(combined)
            train_parts.append(df_combined)
        else:
            train_parts.append(df_lap)

    # Concatenar todas as voltas de treino (originais + sintéticas)
    df_train_augmented = pd.concat(train_parts, ignore_index=True)

    # Concatenar com validação e teste intactos
    final_df = pd.concat([df_train_augmented, df_val_test], ignore_index=True)

    # Garantir que a coluna is_synthetic existe em todo o DataFrame final
    final_df['is_synthetic'] = final_df['is_synthetic'].astype(bool)

    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nAmostras sintéticas geradas: {len(df_train_augmented) - len(df_train)}")
    print(f"Dataset final: {len(final_df)} linhas -> salvo em {OUTPUT_CSV}")


if __name__ == "__main__":
    main()