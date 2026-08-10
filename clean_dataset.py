from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DATA_DIR = Path("data/sessions")
OUTPUT_DIR = Path("data/processed")

# Se None, usa automaticamente o CSV mais recente.
CSV_FILE = None


# ============================================================
# LOCALIZAR DATASET
# ============================================================

def find_dataset():
    if CSV_FILE is not None:
        if not CSV_FILE.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {CSV_FILE}"
            )
        return CSV_FILE

    files = sorted(
        DATA_DIR.glob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {DATA_DIR}"
        )

    return files[0]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ASSETTO CORSA - DATASET CLEANING")
    print("=" * 70)

    csv_path = find_dataset()

    print(f"\nDataset RAW:")
    print(f"  {csv_path}")

    df = pd.read_csv(csv_path)

    original_size = len(df)

    print(f"\nAmostras originais: {original_size:,}")

    # ========================================================
    # 1. NaN
    # ========================================================

    nan_mask = df.isna().any(axis=1)
    nan_removed = nan_mask.sum()

    df = df.loc[~nan_mask].copy()

    print(f"\n[1] NaN removidos: {nan_removed:,}")

    # ========================================================
    # 2. Inf
    # ========================================================

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    inf_mask = np.isinf(
        df[numeric_columns]
    ).any(axis=1)

    inf_removed = inf_mask.sum()

    df = df.loc[~inf_mask].copy()

    print(f"[2] Inf removidos: {inf_removed:,}")

    # ========================================================
    # 3. RPM negativo
    # ========================================================

    rpm_removed = 0

    if "rpm" in df.columns:

        rpm_mask = df["rpm"] < 0

        rpm_removed = rpm_mask.sum()

        df = df.loc[~rpm_mask].copy()

    print(f"[3] RPM negativo removidos: {rpm_removed:,}")

    # ========================================================
    # 4. VOLTA 7
    # ========================================================

    lap_removed = 0

    if "lap" in df.columns:

        lap_mask = df["lap"] == 13

        lap_removed = lap_mask.sum()

        df = df.loc[~lap_mask].copy()

    print(f"[4] Volta 13 removida: {lap_removed:,}")

    # ========================================================
    # RESET INDEX
    # ========================================================

    df = df.reset_index(drop=True)

    # ========================================================
    # ESTATÍSTICAS
    # ========================================================

    final_size = len(df)

    total_removed = original_size - final_size

    percentage_removed = (
        total_removed / original_size * 100
    )

    percentage_remaining = (
        final_size / original_size * 100
    )

    # ========================================================
    # SALVAR
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / "dataset_clean.csv"

    df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # RELATÓRIO
    # ========================================================

    print("\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)

    print(f"\nAmostras originais:  {original_size:,}")
    print(f"Amostras removidas:  {total_removed:,}")
    print(f"Amostras restantes:  {final_size:,}")

    print(
        f"\nRemovido: {percentage_removed:.2f}%"
    )

    print(
        f"Preservado: {percentage_remaining:.2f}%"
    )

    print(f"\nDataset limpo:")
    print(f"  {output_path}")

    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDAÇÃO")
    print("=" * 70)

    print(
        f"\nNaN restantes: "
        f"{df.isna().sum().sum():,}"
    )

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    print(
        f"Inf restantes: "
        f"{np.isinf(df[numeric_columns]).sum().sum():,}"
    )

    if "rpm" in df.columns:

        negative_rpm = (
            df["rpm"] < 0
        ).sum()

        print(
            f"RPM negativo restante: "
            f"{negative_rpm:,}"
        )

    if "lap" in df.columns:

        lap7_remaining = (
            df["lap"] == 13
        ).sum()

        print(
            f"Lap 13 restante: "
            f"{lap7_remaining:,}"
        )

    print("\nDataset limpo com sucesso.")


if __name__ == "__main__":
    main()