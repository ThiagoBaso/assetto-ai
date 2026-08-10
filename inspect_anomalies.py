from pathlib import Path
import pandas as pd
import numpy as np


DATA_DIR = Path("data/sessions")
CSV_FILE = None

# Quantas linhas mostrar antes/depois de cada anomalia
CONTEXT = 3


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def find_dataset():
    if CSV_FILE is not None:
        return CSV_FILE

    files = sorted(
        DATA_DIR.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not files:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {DATA_DIR}"
        )

    return files[0]


def show_anomalies(df, mask, description, columns):
    indexes = df.index[mask]

    print(f"\n{description}")
    print(f"Total encontrado: {len(indexes):,}")

    if len(indexes) == 0:
        return

    print("\nPrimeiras ocorrências:")

    # Limita a quantidade exibida para não gerar um output gigante
    for idx in indexes[:20]:

        start = max(0, idx - CONTEXT)
        end = min(len(df), idx + CONTEXT + 1)

        print(f"\n--- índice {idx} ---")

        available_columns = [
            column
            for column in columns
            if column in df.columns
        ]

        print(
            df.loc[start:end - 1, available_columns]
            .to_string(index=True)
        )


def main():

    section("ASSETTO CORSA - ANOMALY INSPECTION")

    csv_path = find_dataset()

    print(f"\nDataset:")
    print(csv_path)

    df = pd.read_csv(csv_path)

    print(f"Amostras: {len(df):,}")

    # ========================================================
    # SLIP
    # ========================================================

    section("1. SLIP")

    slip_columns = [
        "slip_fl",
        "slip_fr",
        "slip_rl",
        "slip_rr",
    ]

    for column in slip_columns:

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Valores acima de 10 são extremamente suspeitos
        mask = values.abs() > 10

        show_anomalies(
            df,
            mask,
            f"{column} > |10|",
            [
                "timestamp",
                "speed",
                "rpm",
                "gear",
                "steering_input",
                "throttle_input",
                "brake_input",
                column,
            ],
        )

    # ========================================================
    # RPM
    # ========================================================

    section("2. RPM")

    if "rpm" in df.columns:

        rpm = pd.to_numeric(
            df["rpm"],
            errors="coerce"
        )

        mask_negative = rpm < 0

        show_anomalies(
            df,
            mask_negative,
            "RPM negativo",
            [
                "timestamp",
                "speed",
                "rpm",
                "gear",
                "steering_input",
                "throttle_input",
                "brake_input",
            ],
        )

        mask_extreme = rpm > 10000

        show_anomalies(
            df,
            mask_extreme,
            "RPM > 10.000",
            [
                "timestamp",
                "speed",
                "rpm",
                "gear",
                "steering_input",
                "throttle_input",
                "brake_input",
            ],
        )

    # ========================================================
    # ACELERAÇÃO
    # ========================================================

    section("3. ACCELERATION")

    acceleration_columns = [
        "acc_g_x",
        "acc_g_y",
        "acc_g_z",
    ]

    for column in acceleration_columns:

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Valores acima de 5G são suspeitos
        mask = values.abs() > 5

        show_anomalies(
            df,
            mask,
            f"{column} > |5G|",
            [
                "timestamp",
                "speed",
                "rpm",
                "gear",
                "steering_input",
                "throttle_input",
                "brake_input",
                column,
            ],
        )

    # ========================================================
    # STEERING
    # ========================================================

    section("4. STEERING EXTREME")

    if "steering_input" in df.columns:

        steering = pd.to_numeric(
            df["steering_input"],
            errors="coerce"
        )

        mask = steering.abs() > 0.9

        show_anomalies(
            df,
            mask,
            "|steering_input| > 0.9",
            [
                "timestamp",
                "speed",
                "rpm",
                "gear",
                "steer_angle",
                "steering_input",
                "throttle_input",
                "brake_input",
                "lap",
                "sector",
                "lap_progress",
            ],
        )

    # ========================================================
    # TYRES OUT
    # ========================================================

    section("5. TYRES OUT")

    if "tyres_out" in df.columns:

        tyres_out = pd.to_numeric(
            df["tyres_out"],
            errors="coerce"
        )

        mask = tyres_out > 0

        show_anomalies(
            df,
            mask,
            "Pneus fora da pista",
            [
                "timestamp",
                "speed",
                "rpm",
                "gear",
                "steering_input",
                "throttle_input",
                "brake_input",
                "tyres_out",
                "lap",
                "sector",
                "lap_progress",
            ],
        )

    # ========================================================
    # AMOSTRAS DUPLICADAS
    # ========================================================

    section("6. TIMESTAMP")

    if "timestamp" in df.columns:

        timestamp = pd.to_numeric(
            df["timestamp"],
            errors="coerce"
        )

        delta = timestamp.diff()

        print(
            f"\nIntervalo mínimo: "
            f"{delta.min() * 1000:.3f} ms"
        )

        print(
            f"Intervalo máximo: "
            f"{delta.max() * 1000:.3f} ms"
        )

        print(
            f"Intervalo médio: "
            f"{delta.mean() * 1000:.3f} ms"
        )

        negative_delta = delta < 0

        print(
            f"\nTimestamps regressivos: "
            f"{negative_delta.sum():,}"
        )

        zero_delta = delta == 0

        print(
            f"Timestamps duplicados: "
            f"{zero_delta.sum():,}"
        )

    # ========================================================
    # VOLTA 7
    # ========================================================

    section("7. LAP 7")

    if "lap" in df.columns:

        lap7 = df["lap"] == 7

        print(
            f"Amostras da volta 7: "
            f"{lap7.sum():,}"
        )

        if lap7.any():

            print(
                "\nIntervalo temporal da volta 7:"
            )

            print(
                f"  início: "
                f"{df.loc[lap7, 'timestamp'].min():.3f}s"
            )

            print(
                f"  fim: "
                f"{df.loc[lap7, 'timestamp'].max():.3f}s"
            )

    # ========================================================
    # FINAL
    # ========================================================

    section("INSPEÇÃO CONCLUÍDA")

    print(
        "\nNão alteramos o dataset original."
    )

    print(
        "Analise as ocorrências acima antes de remover qualquer dado."
    )


if __name__ == "__main__":
    main()