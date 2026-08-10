import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from pathlib import Path
import joblib

# =============================================
# PARÂMETROS AJUSTÁVEIS
# =============================================
N_BINS = 1000               # resolução da trajetória (quanto maior, mais suave)
SMOOTH_WINDOW = 51          # deve ser ímpar, ≤ N_BINS
POLYORDER = 3               # ordem do polinômio do filtro Savitzky-Golay
FUTURE_DISTANCES = [5, 10, 20, 30, 50]   # metros à frente

def build_centerline(df, n_bins=1000, smooth_window=51, polyorder=3):
    bins = np.linspace(0, 1, n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    df['progress_bin'] = pd.cut(df['lap_progress'], bins=bins, labels=False, include_lowest=True)
    
    # Agrupar por pos_x e pos_z (não pos_y)
    grouped = df.groupby('progress_bin').agg({'pos_x':'mean', 'pos_z':'mean'}).reset_index()
    grouped = grouped.sort_values('progress_bin')
    
    x_raw = grouped['pos_x'].values
    z_raw = grouped['pos_z'].values          # <-- alterado de pos_y para pos_z
    
    x_smooth = savgol_filter(x_raw, smooth_window, polyorder, mode='wrap')
    z_smooth = savgol_filter(z_raw, smooth_window, polyorder, mode='wrap')  # <-- z_smooth
    
    # Distâncias
    dx = np.diff(x_smooth, prepend=x_smooth[0])
    dz = np.diff(z_smooth, prepend=z_smooth[0])
    ds = np.sqrt(dx**2 + dz**2)
    s = np.cumsum(ds)
    total_length = s[-1]
    
    # Heading no plano XZ
    heading_traj = np.arctan2(np.gradient(z_smooth), np.gradient(x_smooth))
    heading_smooth = savgol_filter(heading_traj, smooth_window, polyorder, mode='wrap')
    curvature = np.gradient(heading_smooth) / np.gradient(s)
    
    # Interpoladores
    progress_to_s = interp1d(bin_centers, s, kind='linear', fill_value='extrapolate')
    s_to_progress = interp1d(s, bin_centers, kind='linear', fill_value='extrapolate')
    s_to_x = interp1d(s, x_smooth, kind='cubic', fill_value='extrapolate')
    s_to_z = interp1d(s, z_smooth, kind='cubic', fill_value='extrapolate')  # <-- NOVO
    s_to_heading = interp1d(s, heading_smooth, kind='cubic', fill_value='extrapolate')
    s_to_curv = interp1d(s, curvature, kind='cubic', fill_value='extrapolate')
    
    return {
        'progress_bins': bin_centers,
        'x': x_smooth,
        'z': z_smooth,                         # <-- renomeado e incluído
        's': s,
        'heading': heading_smooth,
        'curvature': curvature,
        'total_length': total_length,
        'interpolators': (progress_to_s, s_to_progress, s_to_x, s_to_z, s_to_heading, s_to_curv)
    }


def add_track_features(df, traj):
    progress_to_s, s_to_progress, s_to_x, s_to_z, s_to_heading, s_to_curv = traj['interpolators']
    
    s_now = progress_to_s(df['lap_progress'].values)
    
    heading_car = df['heading'].values
    heading_traj_now = s_to_heading(s_now)
    heading_error = np.angle(np.exp(1j * (heading_car - heading_traj_now)))
    
    # Posição na trajetória
    x_traj = s_to_x(s_now)
    z_traj = s_to_z(s_now)
    
    # Lateral position com sinal
    dx_traj = np.cos(heading_traj_now)
    dz_traj = np.sin(heading_traj_now)
    lateral_pos = np.sqrt((df['pos_x'] - x_traj)**2 + (df['pos_z'] - z_traj)**2)
    lateral_sign = np.sign((df['pos_x'] - x_traj)*(-dz_traj) + (df['pos_z'] - z_traj)*dx_traj)
    lateral_pos = lateral_pos * lateral_sign
    
    # Curvaturas futuras
    for d in [5,10,20,30,50]:
        s_future = np.mod(s_now + d, traj['total_length'])
        df[f'curvature_{d}m'] = s_to_curv(s_future)
    
    df['curve_direction'] = np.sign(df['curvature_5m'].values)
    df['heading_error'] = heading_error
    df['lateral_position'] = lateral_pos

    # Calcular velocidade máxima segura para curvatura a 10m (ou a menor distância)
    g = 9.81
    max_lat_acc = 0.8 * g   # m/s² (ajuste conforme carro/pista)
    future_curvature = np.abs(df['curvature_10m'].values)  # usar curvatura a 10m
    # Evitar divisão por zero (reta = raio infinito)
    radius = np.where(future_curvature > 0.001, 1.0 / future_curvature, 9999.0)
    v_max_ms = np.sqrt(max_lat_acc * radius)
    v_max_kmh = v_max_ms * 3.6
    speed_error = df['speed'].values - v_max_kmh      # positivo = rápido demais
    df['speed_error'] = speed_error

    return df


# =============================================
# MAIN: executa quando rodar este script
# =============================================
if __name__ == '__main__':
    print("Carregando dataset limpo...")
    df = pd.read_csv('data/processed/dataset_clean.csv')
    print(f"Linhas carregadas: {len(df)}")
    
    # Verificar se a coluna 'heading' existe
    if 'heading' not in df.columns:
        raise ValueError(
            "Coluna 'heading' não encontrada. "
            "Adicione 'heading' ao telemetry.py e regrave o dataset."
        )
    
    # Construir trajetória
    print("Construindo trajetória suavizada...")
    traj = build_centerline(df)
    print(f"Comprimento total: {traj['total_length']:.1f} m")
    
    # Adicionar features
    print("Adicionando sensores de pista...")
    df = add_track_features(df, traj)

    traj_data = {
    's': traj['s'],
    'x': traj['x'],
    'z': traj['z'],          # <-- agora z em vez de y
    'heading': traj['heading'],
    'curvature': traj['curvature'],
    'total_length': traj['total_length'],
    'progress_bins': traj['progress_bins']
}
    joblib.dump(traj_data, 'trajectory.pkl')
    print("Trajetória salva em: trajectory.pkl")
    
    # Salvar novo dataset
    output_path = 'data/processed/dataset_with_track.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset com sensores salvo em: {output_path}")

    # Visualização rápida
    import matplotlib.pyplot as plt

    # Trajetória
    plt.figure(figsize=(8,6))
    # Se houver scatter dos pontos brutos, use:
    # plt.scatter(df['pos_x'], df['pos_z'], s=0.5, alpha=0.2, label='dados brutos')
    plt.plot(traj['x'], traj['z'], 'r-', linewidth=2, label='trajetória suave')
    plt.axis('equal')
    plt.legend()
    plt.title('Trajetória extraída (plano XZ)')
    plt.xlabel('pos_x')
    plt.ylabel('pos_z')
    plt.show()

    # Curvatura (mantém igual)
    plt.figure()
    plt.plot(traj['progress_bins'], traj['curvature'])
    plt.xlabel('Progresso na volta')
    plt.ylabel('Curvatura (1/m)')
    plt.title('Curvatura ao longo da pista')
    plt.grid(True)
    plt.show()