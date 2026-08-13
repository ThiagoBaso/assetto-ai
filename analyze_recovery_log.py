"""
analyze_recovery_log.py

Le o CSV gerado pelo recovery_logger.py e plota os sensores de pista em
torno dos momentos em que tyres_out > 0 (carro fora da pista).

O que procurar nos graficos:
- Se s_current "trava" (fica plano ou salta abruptamente) exatamente quando
  tyres_out sobe, e um forte indicio de que a busca do ponto mais proximo
  na trajetoria esta se perdendo quando o carro desvia muito.
- Se curvature_0m/curvature_10m mudam de forma brusca e incoerente com o
  que deveria ser a pista naquele trecho, reforca a mesma hipotese.
- Se heading_error ou lateral_position disparam para valores absurdos
  (nao apenas grandes, mas descontinuos) no mesmo instante, tambem aponta
  para erro no sensor, nao so no controle.

Ajuste INPUT_CSV para o caminho do log gerado.
"""

import matplotlib.pyplot as plt
import pandas as pd

INPUT_CSV = "recovery_log.csv"

# Quantos frames mostrar antes/depois de cada evento de saida de pista
WINDOW_BEFORE = 60
WINDOW_AFTER = 60


def find_excursion_starts(df: pd.DataFrame) -> list:
    """Encontra os indices onde tyres_out passa de 0 para >0."""
    if "tyres_out" not in df.columns:
        print("Coluna 'tyres_out' nao encontrada no log — pulei essa etapa.")
        return []

    out = df["tyres_out"] > 0
    starts = df.index[out & ~out.shift(1, fill_value=False)]
    return list(starts)


def plot_event(df: pd.DataFrame, center_idx: int, event_num: int):
    lo = max(0, center_idx - WINDOW_BEFORE)
    hi = min(len(df), center_idx + WINDOW_AFTER)
    window = df.iloc[lo:hi]

    cols_to_plot = [
        c for c in [
            "s_current", "lateral_position", "heading_error",
            "curvature_0m", "curvature_10m", "speed_error",
        ]
        if c in df.columns
    ]

    if not cols_to_plot:
        print("Nenhuma das colunas esperadas foi encontrada no log.")
        return

    fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 2.2 * len(cols_to_plot)), sharex=True)
    if len(cols_to_plot) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols_to_plot):
        ax.plot(window.index, window[col])
        ax.axvline(center_idx, color="red", linestyle="--", linewidth=1, label="tyres_out ligou")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    axes[-1].set_xlabel("indice (frame)")
    fig.suptitle(f"Saida de pista #{event_num} (frame {center_idx})")
    fig.tight_layout()


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Log carregado: {len(df)} frames")

    starts = find_excursion_starts(df)
    print(f"Eventos de saida de pista encontrados: {len(starts)}")

    if not starts:
        print("Nenhum evento encontrado. Confirme se tyres_out esta sendo logado "
              "e se houve pelo menos uma saida de pista na sessao gravada.")
        return

    # Plota os 5 primeiros eventos para nao sobrecarregar a tela.
    for i, idx in enumerate(starts[:5], start=1):
        plot_event(df, idx, i)

    plt.show()


if __name__ == "__main__":
    main()
