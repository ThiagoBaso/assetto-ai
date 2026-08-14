from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DATA_DIR = Path("data/sessions")            # pasta com CSVs brutos
OUTPUT_DIR = Path("data/processed")         # pasta de saída

# Se None, usa automaticamente o CSV mais recente.
CSV_FILE = Path("data/processed/dataset_with_track_augmented.csv")
#CSV_FILE = None

# Ativar / desativar filtros
ENABLE_TYRES_OUT_FILTER = True              # remove amostras com tyres_out > 0 (fora da pista)
ENABLE_CURVATURE_FILTERS = True            # remove amostras com ações incoerentes com a curvatura (requer features de pista)
ENABLE_LAP_ANOMALY_FILTER = True           # remove voltas com variância anômala de steering/brake
ENABLE_SMOOTHING = True                    # suaviza targets (steering, throttle, brake) com média móvel

# Limiares dos filtros de curvatura (ajuste conforme pista/carro)
STEER_SATURATION = 0.95                    # |steering| acima disso é considerado saturado
CURV_STRAIGHT_STEER = 0.002                # curvatura a 5m abaixo disso é considerada reta
BRAKE_STRONG = 0.5                         # brake acima disso é considerado forte
CURV_STRAIGHT_BRAKE = 0.001                # curvatura a 10m abaixo disso é considerada reta

# Configuração do filtro de voltas anômalas
ANOMALY_PERCENTILE = 90                    # remove as voltas cujo score está acima deste percentil (0-100)
# Score de anomalia = std_steering + std_brake

# Suavização das ações
SMOOTH_WINDOW = 5                          # tamanho da janela da média móvel


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
    # 4. FILTRO DE PNEUS FORA DA PISTA
    # ========================================================

    tyres_out_removed = 0

    if ENABLE_TYRES_OUT_FILTER and "tyres_out" in df.columns:

        tyres_mask = df["tyres_out"] > 0

        tyres_out_removed = tyres_mask.sum()

        df = df.loc[~tyres_mask].copy()

    print(f"[4] Amostras com pneus fora da pista removidas: {tyres_out_removed:,}")

    # ========================================================
    # 5. FILTROS DE CURVATURA (opcional)
    # ========================================================

    curvature_removed = 0

    if ENABLE_CURVATURE_FILTERS:
        # Verifica se as colunas necessárias existem
        required_cols = ["steering_input", "brake_input", "curvature_5m", "curvature_10m"]
        if all(col in df.columns for col in required_cols):
            # a) Steering saturado em reta
            mask_steer_reta = (
                (df["steering_input"].abs() > STEER_SATURATION) &
                (df["curvature_5m"].abs() < CURV_STRAIGHT_STEER)
            )
            # b) Freada forte em reta
            mask_brake_reta = (
                (df["brake_input"] > BRAKE_STRONG) &
                (df["curvature_10m"].abs() < CURV_STRAIGHT_BRAKE)
            )
            combined_mask = mask_steer_reta | mask_brake_reta
            curvature_removed = combined_mask.sum()
            df = df.loc[~combined_mask].copy()
            print(f"[5] Amostras com ações incoerentes com a curvatura removidas: {curvature_removed:,}")
        else:
            print("[5] Filtros de curvatura não aplicados (colunas ausentes).")

    # ========================================================
    # 6. FILTRO DE VOLTAS ANÔMALAS (opcional)
    # ========================================================

    laps_anomaly_removed = 0

    if ENABLE_LAP_ANOMALY_FILTER and "lap" in df.columns and "steering_input" in df.columns:
        lap_stats = df.groupby("lap").agg(
            std_steering=("steering_input", "std"),
            std_brake=("brake_input", "std"),
            count=("steering_input", "count")
        )
        # Score de anomalia
        lap_stats["anomaly_score"] = (
            lap_stats["std_steering"] + lap_stats["std_brake"]
        )
        # Determina o limiar como percentil
        threshold = np.percentile(
            lap_stats["anomaly_score"],
            ANOMALY_PERCENTILE
        )
        # Seleciona voltas acima do limiar
        bad_laps = lap_stats[
            lap_stats["anomaly_score"] > threshold
        ].index

        laps_anomaly_removed = len(bad_laps)

        if laps_anomaly_removed > 0:
            df = df[~df["lap"].isin(bad_laps)].copy()
            print(f"[6] Voltas anômalas removidas: {laps_anomaly_removed:,} "
                  f"({(laps_anomaly_removed / len(lap_stats)) * 100:.1f}% das voltas)")
        else:
            print("[6] Nenhuma volta anômala removida.")
    else:
        print("[6] Filtro de voltas anômalas não aplicado (colunas ausentes ou desativado).")

    # ========================================================
    # 7. SUAVIZAÇÃO DAS AÇÕES (opcional)
    # ========================================================

    if ENABLE_SMOOTHING and all(col in df.columns for col in ["lap", "s_current", "steering_input", "throttle_input", "brake_input"]):
        print("[7] Aplicando suavização das ações...")
        # Ordena por volta e posição na pista para a média móvel
        df = df.sort_values(["lap", "s_current"]).reset_index(drop=True)
        for col in ["steering_input", "throttle_input", "brake_input"]:
            df[col] = df.groupby("lap")[col].transform(
                lambda x: x.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
            )
        print("    Suavização concluída.")
    else:
        print("[7] Suavização não aplicada (colunas ausentes ou desativada).")

    # ========================================================
    # RESET INDEX
    # ========================================================

    df = df.reset_index(drop=True)

    # ========================================================
    # ESTATÍSTICAS
    # ========================================================

    final_size = len(df)
    total_removed = original_size - final_size
    percentage_removed = total_removed / original_size * 100
    percentage_remaining = final_size / original_size * 100

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

    if "tyres_out" in df.columns:

        tyres_out_remaining = (
            df["tyres_out"] > 0
        ).sum()

        print(
            f"Amostras com tyres_out > 0 restantes: "
            f"{tyres_out_remaining:,}"
        )

    print("\nDataset limpo com sucesso.")


if __name__ == "__main__":
    main()