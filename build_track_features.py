import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from pathlib import Path
import joblib

# =============================================
# PARÂMETROS AJUSTÁVEIS
# =============================================
N_BINS = 1000
SMOOTH_WINDOW = 51
POLYORDER = 3
FUTURE_DISTANCES = [5, 10, 20, 30, 50]

# Parâmetros para cálculo do offset de heading
MIN_SPEED_FOR_OFFSET = 30.0   # km/h
MAX_CURV_FOR_OFFSET = 0.001   # 1/m


def build_centerline(df, n_bins=1000, smooth_window=51, polyorder=3):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    df['progress_bin'] = pd.cut(
        df['lap_progress'],
        bins=bins,
        labels=False,
        include_lowest=True
    )

    grouped = df.groupby('progress_bin').agg({
        'pos_x': 'mean',
        'pos_z': 'mean'
    }).reset_index().sort_values('progress_bin')

    x_raw = grouped['pos_x'].values
    z_raw = grouped['pos_z'].values

    x_smooth = savgol_filter(x_raw, smooth_window, polyorder, mode='wrap')
    z_smooth = savgol_filter(z_raw, smooth_window, polyorder, mode='wrap')

    dx = np.diff(x_smooth, prepend=x_smooth[0])
    dz = np.diff(z_smooth, prepend=z_smooth[0])
    ds = np.sqrt(dx**2 + dz**2)
    s = np.cumsum(ds)
    total_length = s[-1]

    heading_traj = np.arctan2(np.gradient(z_smooth), np.gradient(x_smooth))
    heading_smooth = savgol_filter(heading_traj, smooth_window, polyorder, mode='wrap')
    curvature = np.gradient(heading_smooth) / np.gradient(s)

    progress_to_s = interp1d(bin_centers, s, kind='linear', fill_value='extrapolate')
    s_to_progress = interp1d(s, bin_centers, kind='linear', fill_value='extrapolate')
    s_to_x = interp1d(s, x_smooth, kind='cubic', fill_value='extrapolate')
    s_to_z = interp1d(s, z_smooth, kind='cubic', fill_value='extrapolate')
    s_to_heading = interp1d(s, heading_smooth, kind='cubic', fill_value='extrapolate')
    s_to_curv = interp1d(s, curvature, kind='cubic', fill_value='extrapolate')

    return {
        'progress_bins': bin_centers,
        'x': x_smooth,
        'z': z_smooth,
        's': s,
        'heading': heading_smooth,
        'curvature': curvature,
        'total_length': total_length,
        'interpolators': (
            progress_to_s,
            s_to_progress,
            s_to_x,
            s_to_z,
            s_to_heading,
            s_to_curv
        )
    }


def add_track_features(df, traj):
    """
    Adiciona sensores de pista ao DataFrame.
    Retorna (df_atualizado, offset_heading).
    """
    progress_to_s, s_to_progress, s_to_x, s_to_z, s_to_heading, s_to_curv = traj['interpolators']

    s_now = progress_to_s(df['lap_progress'].values)
    heading_car = df['heading'].values
    heading_traj_now = s_to_heading(s_now)

    # ===== Cálculo automático do offset de heading =====
    curvature_now = s_to_curv(s_now)
    speed = df['speed'].values
    mask = (
        (np.abs(curvature_now) < MAX_CURV_FOR_OFFSET) &
        (speed > MIN_SPEED_FOR_OFFSET)
    )

    if mask.sum() > 100:
        raw_diff = heading_traj_now[mask] - heading_car[mask]
        offset = np.angle(np.exp(1j * raw_diff).mean())
        print(f"Offset estimado: {offset:.3f} rad ({np.degrees(offset):.1f}°)")
    else:
        offset = 0.0
        print("Amostras insuficientes para calcular offset. Usando 0.")

    # Aplica a correção
    heading_car_corrected = heading_car + offset
    heading_error = np.angle(np.exp(1j * (heading_car_corrected - heading_traj_now)))

    # ===== Posição lateral =====
    x_traj = s_to_x(s_now)
    z_traj = s_to_z(s_now)
    dx_traj = np.cos(heading_traj_now)
    dz_traj = np.sin(heading_traj_now)

    lateral_pos = np.sqrt((df['pos_x'] - x_traj)**2 + (df['pos_z'] - z_traj)**2)
    lateral_sign = np.sign(
        (df['pos_x'] - x_traj) * (-dz_traj) +
        (df['pos_z'] - z_traj) * dx_traj
    )
    lateral_pos = lateral_pos * lateral_sign

    # ===== Curvaturas futuras =====
    for d in FUTURE_DISTANCES:
        s_future = np.mod(s_now + d, traj['total_length'])
        df[f'curvature_{d}m'] = s_to_curv(s_future)
        df['curvature_0m'] = s_to_curv(s_now)
        df['s_current'] = s_now 

    df['curve_direction'] = np.sign(df['curvature_5m'].values)
    df['heading_error'] = heading_error
    df['lateral_position'] = lateral_pos

    # ===== Speed error =====
    g = 9.81
    max_lat_acc = 0.8 * g
    future_curvature = np.abs(df['curvature_10m'].values)
    radius = np.where(future_curvature > 0.001, 1.0 / future_curvature, 9999.0)
    v_max_ms = np.sqrt(max_lat_acc * radius)
    v_max_kmh = v_max_ms * 3.6
    df['speed_error'] = df['speed'].values - v_max_kmh

    return df, offset


# =============================================
# MAIN
# =============================================
if __name__ == '__main__':
    print("Carregando dataset limpo...")
    df = pd.read_csv('data/processed/dataset_clean.csv')
    print(f"Linhas carregadas: {len(df)}")

    if 'heading' not in df.columns:
        raise ValueError(
            "Coluna 'heading' não encontrada. "
            "Adicione 'heading' ao telemetry.py e regrave o dataset."
        )

    print("Construindo trajetória suavizada...")
    traj = build_centerline(df)
    print(f"Comprimento total: {traj['total_length']:.1f} m")

    print("Adicionando sensores de pista...")
    df, offset = add_track_features(df, traj)

    # Salvar trajetória com o offset
    traj_data = {
        's': traj['s'],
        'x': traj['x'],
        'z': traj['z'],
        'heading': traj['heading'],
        'curvature': traj['curvature'],
        'total_length': traj['total_length'],
        'progress_bins': traj['progress_bins'],
        'heading_offset': offset   # <-- agora está definido
    }
    joblib.dump(traj_data, 'trajectory.pkl')
    print("Trajetória salva em: trajectory.pkl")

    output_path = 'data/processed/dataset_with_track.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset com sensores salvo em: {output_path}")

    # Visualização opcional
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(traj['x'], traj['z'], 'r-', linewidth=2, label='trajetória suave')
    plt.axis('equal')
    plt.legend()
    plt.title('Trajetória extraída (plano XZ)')
    plt.xlabel('pos_x')
    plt.ylabel('pos_z')
    plt.show()

    plt.figure()
    plt.plot(traj['progress_bins'], traj['curvature'])
    plt.xlabel('Progresso na volta')
    plt.ylabel('Curvatura (1/m)')
    plt.title('Curvatura ao longo da pista')
    plt.grid(True)
    plt.show()