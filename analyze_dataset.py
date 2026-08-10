from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DATA_DIR = Path("data/sessions")

# Se quiser analisar um arquivo específico, coloque o nome aqui.
# Exemplo:
# CSV_FILE = DATA_DIR / "session_123456.csv"
CSV_FILE = None


# ============================================================
# UTILITÁRIOS
# ============================================================

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_stat(name, value):
    print(f"{name:<35} {value}")


# ============================================================
# LOCALIZAR DATASET
# ============================================================

def find_dataset():
    if CSV_FILE is not None:
        if not CSV_FILE.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {CSV_FILE}")
        return CSV_FILE

    files = sorted(
        DATA_DIR.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not files:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em: {DATA_DIR}"
        )

    return files[0]


# ============================================================
# ANÁLISE
# ============================================================

def main():

    print("=" * 70)
    print("ASSETTO CORSA - DATASET ANALYSIS")
    print("=" * 70)

    csv_path = find_dataset()

    print(f"\nDataset analisado:")
    print(csv_path)

    # --------------------------------------------------------
    # CARREGAR
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    section("1. DATASET")

    print_stat("Número de amostras", f"{len(df):,}")
    print_stat("Número de features/colunas", len(df.columns))

    print("\nColunas:")
    for column in df.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # TIMESTAMP / DURAÇÃO / FREQUÊNCIA
    # --------------------------------------------------------

    section("2. TEMPO E FREQUÊNCIA")

    if "timestamp" in df.columns:

        timestamp = pd.to_numeric(
            df["timestamp"],
            errors="coerce"
        )

        valid_timestamp = timestamp.dropna()

        if len(valid_timestamp) > 1:

            duration = (
                valid_timestamp.iloc[-1]
                - valid_timestamp.iloc[0]
            )

            delta = valid_timestamp.diff().dropna()

            mean_dt = delta.mean()
            median_dt = delta.median()

            frequency = 1 / mean_dt if mean_dt > 0 else np.nan

            print_stat("Timestamp inicial", f"{valid_timestamp.iloc[0]:.4f}s")
            print_stat("Timestamp final", f"{valid_timestamp.iloc[-1]:.4f}s")
            print_stat("Duração", f"{duration:.2f}s")
            print_stat("Duração", f"{duration / 60:.2f} minutos")

            print_stat(
                "Intervalo médio",
                f"{mean_dt * 1000:.3f} ms"
            )

            print_stat(
                "Intervalo mediano",
                f"{median_dt * 1000:.3f} ms"
            )

            print_stat(
                "Frequência estimada",
                f"{frequency:.2f} Hz"
            )

            print_stat(
                "Menor intervalo",
                f"{delta.min() * 1000:.3f} ms"
            )

            print_stat(
                "Maior intervalo",
                f"{delta.max() * 1000:.3f} ms"
            )

    else:
        print("Coluna timestamp não encontrada.")

    # --------------------------------------------------------
    # VALORES AUSENTES
    # --------------------------------------------------------

    section("3. VALORES AUSENTES")

    missing = df.isna().sum()

    total_missing = missing.sum()

    print_stat("Total de valores ausentes", f"{total_missing:,}")

    if total_missing > 0:

        print("\nColunas com valores ausentes:")

        for column, count in missing[missing > 0].items():

            percentage = count / len(df) * 100

            print(
                f"  {column:<25}"
                f"{count:>8,}"
                f" ({percentage:.3f}%)"
            )

    else:
        print("Nenhum valor ausente encontrado.")

    # --------------------------------------------------------
    # INF
    # --------------------------------------------------------

    section("4. VALORES INFINITOS")

    numeric_df = df.select_dtypes(include=[np.number])

    inf_count = np.isinf(numeric_df).sum()

    total_inf = inf_count.sum()

    print_stat("Total de valores Inf", f"{total_inf:,}")

    if total_inf > 0:

        print("\nColunas afetadas:")

        for column, count in inf_count[inf_count > 0].items():
            print(f"  {column:<25}{count:,}")

    else:
        print("Nenhum valor infinito encontrado.")

    # --------------------------------------------------------
    # ESTATÍSTICAS NUMÉRICAS
    # --------------------------------------------------------

    section("5. ESTATÍSTICAS DAS FEATURES")

    stats = numeric_df.describe().T

    for column in stats.index:

        print(
            f"\n{column}"
        )

        print(
            f"  min    = {stats.loc[column, 'min']:.6f}"
        )

        print(
            f"  max    = {stats.loc[column, 'max']:.6f}"
        )

        print(
            f"  mean   = {stats.loc[column, 'mean']:.6f}"
        )

        print(
            f"  std    = {stats.loc[column, 'std']:.6f}"
        )

        print(
            f"  median = {stats.loc[column, '50%']:.6f}"
        )

    # --------------------------------------------------------
    # AÇÕES
    # --------------------------------------------------------

    action_columns = [
        "steering_input",
        "throttle_input",
        "brake_input"
    ]

    section("6. DISTRIBUIÇÃO DAS AÇÕES")

    for column in action_columns:

        if column not in df.columns:
            print(f"\n{column}: coluna não encontrada.")
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        print(f"\n{column}")

        print(f"  min      = {values.min():.6f}")
        print(f"  max      = {values.max():.6f}")
        print(f"  mean     = {values.mean():.6f}")
        print(f"  std      = {values.std():.6f}")
        print(f"  median   = {values.median():.6f}")

        # Percentis
        print(f"  p01      = {values.quantile(0.01):.6f}")
        print(f"  p05      = {values.quantile(0.05):.6f}")
        print(f"  p25      = {values.quantile(0.25):.6f}")
        print(f"  p75      = {values.quantile(0.75):.6f}")
        print(f"  p95      = {values.quantile(0.95):.6f}")
        print(f"  p99      = {values.quantile(0.99):.6f}")

    # --------------------------------------------------------
    # STEERING
    # --------------------------------------------------------

    section("7. STEERING")

    if "steering_input" in df.columns:

        steering = df["steering_input"]

        left = (steering < -0.05).sum()
        center = (
            (steering >= -0.05)
            & (steering <= 0.05)
        ).sum()
        right = (steering > 0.05).sum()

        total = len(steering)

        print_stat(
            "Esquerda",
            f"{left:,} ({left / total * 100:.2f}%)"
        )

        print_stat(
            "Centro",
            f"{center:,} ({center / total * 100:.2f}%)"
        )

        print_stat(
            "Direita",
            f"{right:,} ({right / total * 100:.2f}%)"
        )

        extreme = (steering.abs() > 0.9).sum()

        print_stat(
            "Steering extremo > 0.9",
            f"{extreme:,} ({extreme / total * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # THROTTLE / BRAKE
    # --------------------------------------------------------

    section("8. THROTTLE / BRAKE")

    if "throttle_input" in df.columns:

        throttle = df["throttle_input"]

        zero = (throttle < 0.05).sum()
        partial = (
            (throttle >= 0.05)
            & (throttle < 0.95)
        ).sum()
        full = (throttle >= 0.95).sum()

        total = len(throttle)

        print("Throttle:")

        print_stat(
            "  Solto",
            f"{zero:,} ({zero / total * 100:.2f}%)"
        )

        print_stat(
            "  Parcial",
            f"{partial:,} ({partial / total * 100:.2f}%)"
        )

        print_stat(
            "  Total",
            f"{full:,} ({full / total * 100:.2f}%)"
        )

    if "brake_input" in df.columns:

        brake = df["brake_input"]

        zero = (brake < 0.05).sum()
        braking = (brake >= 0.05).sum()
        strong = (brake >= 0.7).sum()

        total = len(brake)

        print("\nBrake:")

        print_stat(
            "  Sem freio",
            f"{zero:,} ({zero / total * 100:.2f}%)"
        )

        print_stat(
            "  Freando",
            f"{braking:,} ({braking / total * 100:.2f}%)"
        )

        print_stat(
            "  Frenagem forte >= 0.7",
            f"{strong:,} ({strong / total * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # VELOCIDADE
    # --------------------------------------------------------

    section("9. VELOCIDADE")

    if "speed" in df.columns:

        speed = df["speed"]

        stopped = (speed < 1).sum()
        slow = (speed < 10).sum()
        moving = (speed >= 10).sum()

        total = len(speed)

        print_stat(
            "Velocidade média",
            f"{speed.mean():.2f} km/h"
        )

        print_stat(
            "Velocidade máxima",
            f"{speed.max():.2f} km/h"
        )

        print_stat(
            "Carro parado < 1 km/h",
            f"{stopped:,} ({stopped / total * 100:.2f}%)"
        )

        print_stat(
            "Velocidade < 10 km/h",
            f"{slow:,} ({slow / total * 100:.2f}%)"
        )

        print_stat(
            "Velocidade >= 10 km/h",
            f"{moving:,} ({moving / total * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # TYRES OUT
    # --------------------------------------------------------

    section("10. TYRES OUT")

    if "tyres_out" in df.columns:

        tyres_out = df["tyres_out"]

        count = (tyres_out > 0).sum()

        print_stat(
            "Amostras com pneus fora",
            f"{count:,} ({count / len(df) * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # LAP / SECTOR
    # --------------------------------------------------------

    section("11. VOLTAS E SETORES")

    if "lap" in df.columns:

        laps = df["lap"].nunique()

        print_stat(
            "Número de valores de lap",
            laps
        )

        print("\nDistribuição por volta:")

        lap_counts = df["lap"].value_counts().sort_index()

        for lap, count in lap_counts.items():

            percentage = count / len(df) * 100

            print(
                f"  Lap {lap:<5}"
                f"{count:>8,} samples"
                f" ({percentage:.2f}%)"
            )

    if "sector" in df.columns:

        print("\nDistribuição por setor:")

        sector_counts = df["sector"].value_counts().sort_index()

        for sector, count in sector_counts.items():

            print(
                f"  Sector {sector:<5}"
                f"{count:>8,} samples"
            )

    # --------------------------------------------------------
    # VALORES SUSPEITOS
    # --------------------------------------------------------

    section("12. VALORES POTENCIALMENTE SUSPEITOS")

    suspicious = []

    checks = {
        "speed > 400": (
            "speed" in df.columns,
            df["speed"] > 400 if "speed" in df.columns else None
        ),

        "rpm > 20000": (
            "rpm" in df.columns,
            df["rpm"] > 20000 if "rpm" in df.columns else None
        ),

        "steering fora [-1, 1]": (
            "steering_input" in df.columns,
            (
                (df["steering_input"] < -1)
                | (df["steering_input"] > 1)
            )
            if "steering_input" in df.columns
            else None
        ),

        "throttle fora [0, 1]": (
            "throttle_input" in df.columns,
            (
                (df["throttle_input"] < 0)
                | (df["throttle_input"] > 1)
            )
            if "throttle_input" in df.columns
            else None
        ),

        "brake fora [0, 1]": (
            "brake_input" in df.columns,
            (
                (df["brake_input"] < 0)
                | (df["brake_input"] > 1)
            )
            if "brake_input" in df.columns
            else None
        ),
    }

    for name, (exists, mask) in checks.items():

        if not exists:
            continue

        count = mask.sum()

        if count > 0:

            suspicious.append((name, count))

            print(
                f"  ⚠ {name}: {count:,} amostras"
            )

    if not suspicious:
        print("Nenhum valor obviamente inválido encontrado.")

    # --------------------------------------------------------
    # FEATURES RECOMENDADAS
    # --------------------------------------------------------

    section("13. FEATURES CANDIDATAS PARA O BASELINE")

    recommended = [
        "speed",
        "rpm",
        "gear",
        "steer_angle",
        "acc_g_x",
        "acc_g_y",
        "local_vel_x",
        "local_vel_y",
        "local_vel_z",
        "angular_vel_x",
        "angular_vel_y",
        "angular_vel_z",
        "slip_fl",
        "slip_fr",
        "slip_rl",
        "slip_rr",
        "lap_progress",
        "surface_grip",
        "tyres_out",
    ]

    print(
        "\nEstas são candidatas iniciais."
        "\nA seleção definitiva será feita após"
        "\na análise do dataset."
    )

    for feature in recommended:

        if feature in df.columns:
            print(f"  ✓ {feature}")
        else:
            print(f"  - {feature} (não encontrada)")

    # --------------------------------------------------------
    # RESUMO
    # --------------------------------------------------------

    section("14. RESUMO")

    print(f"Dataset: {csv_path.name}")
    print(f"Amostras: {len(df):,}")

    if "timestamp" in df.columns:

        timestamp = pd.to_numeric(
            df["timestamp"],
            errors="coerce"
        ).dropna()

        if len(timestamp) > 1:

            duration = timestamp.iloc[-1] - timestamp.iloc[0]

            delta = timestamp.diff().dropna()

            frequency = 1 / delta.mean()

            print(f"Duração: {duration:.2f}s")
            print(f"Frequência: {frequency:.2f} Hz")

    print(f"Valores ausentes: {total_missing:,}")
    print(f"Valores Inf: {total_inf:,}")

    print("\nAnálise concluída.")

    print(
        "\nPRÓXIMO PASSO:"
        "\nEnvie o resultado deste relatório."
        "\nNão precisamos enviar o CSV inteiro."
    )


if __name__ == "__main__":
    main()