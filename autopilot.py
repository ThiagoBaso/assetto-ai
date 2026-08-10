import time
import numpy as np
import torch
import joblib
from scipy.interpolate import interp1d
import pyvjoy

from telemetry import Telemetry

# =============================================
# CONFIGURAÇÕES AJUSTÁVEIS
# =============================================
STEER_ALPHA = 0.1
STEER_SCALE = 0.3
HEADING_ALPHA = 0.2
TAKEOFF_SPEED = 10.0

FEATURE_COLS = [
    'speed', 'rpm', 'gear',
    'acc_g_x', 'acc_g_y', 'acc_g_z',
    'local_vel_x', 'local_vel_y', 'local_vel_z',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
    'slip_fl', 'slip_fr', 'slip_rl', 'slip_rr',
    'tyre_temp_fl', 'tyre_temp_fr', 'tyre_temp_rl', 'tyre_temp_rr',
    'lap_progress', 'sector', 'surface_grip', 'tyres_out',
    'heading_error', 'lateral_position',
    'curvature_5m', 'curvature_10m', 'curvature_20m', 'curvature_30m', 'curvature_50m',
    'curve_direction',
    'speed_error'
]

print("Features:", len(FEATURE_COLS))

# =============================================
# CARREGAR MODELO, SCALER E TRAJETÓRIA
# =============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DrivingModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_layers=[256, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(torch.nn.Linear(prev_dim, h))
            layers.append(torch.nn.ReLU())
            prev_dim = h
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

print("Carregando modelo...")
input_dim = len(FEATURE_COLS)
model = DrivingModel(input_dim).to(device)
model.load_state_dict(torch.load('driving_model.pth', map_location=device))
model.eval()

scaler = joblib.load('scaler.save')
traj_data = joblib.load('trajectory.pkl')

# Reconstruir interpoladores (plano XZ)
s = traj_data['s']
total_length = traj_data['total_length']
progress_bins = traj_data['progress_bins']
s_to_x = interp1d(s, traj_data['x'], kind='cubic', fill_value='extrapolate')
s_to_z = interp1d(s, traj_data['z'], kind='cubic', fill_value='extrapolate')
s_to_heading = interp1d(s, traj_data['heading'], kind='cubic', fill_value='extrapolate')
s_to_curv = interp1d(s, traj_data['curvature'], kind='cubic', fill_value='extrapolate')
progress_to_s = interp1d(progress_bins, s, kind='linear', fill_value='extrapolate')

# =============================================
# JOYSTICK VIRTUAL (vJoy)
# =============================================
vj = pyvjoy.VJoyDevice(1)

def send_controls(steer, throttle, brake):
    steer_val = int((steer + 1.0) * 0x4000)
    throttle_val = int(throttle * 0x8000)
    brake_val = int(brake * 0x8000)
    vj.set_axis(pyvjoy.HID_USAGE_X, steer_val)
    vj.set_axis(pyvjoy.HID_USAGE_Y, throttle_val)
    vj.set_axis(pyvjoy.HID_USAGE_Z, brake_val)

# =============================================
# CÁLCULO DE SENSORES DE PISTA
# =============================================
def compute_track_features(lap_progress, heading_car, pos_x, pos_z):
    s_now = progress_to_s(np.clip(lap_progress, 0.0, 1.0))
    heading_traj_now = s_to_heading(s_now)
    heading_error = np.angle(np.exp(1j * (heading_car - heading_traj_now)))

    future_dists = [5, 10, 20, 30, 50]
    curvatures = []
    for d in future_dists:
        s_future = np.mod(s_now + d, total_length)
        curvatures.append(s_to_curv(s_future))
    curve_direction = np.sign(curvatures[0])

    x_traj = s_to_x(s_now)
    z_traj = s_to_z(s_now)
    dx_traj = np.cos(heading_traj_now)
    dz_traj = np.sin(heading_traj_now)
    lateral_pos = np.sqrt((pos_x - x_traj)**2 + (pos_z - z_traj)**2)
    lateral_sign = np.sign((pos_x - x_traj)*(-dz_traj) + (pos_z - z_traj)*dx_traj)
    lateral_pos *= lateral_sign

    return heading_error, lateral_pos, curvatures[0], curvatures[1], curvatures[2], curvatures[3], curvatures[4], curve_direction

# =============================================
# FREIO DE SEGURANÇA (safety brake)
# =============================================
def safety_brake(speed_kmh, curvature):
    """Retorna valor de freio (0..1) se velocidade for alta para a curvatura."""
    if abs(curvature) < 0.001:
        return 0.0
    g = 9.81
    max_lat_acc = 0.8 * g
    radius = 1.0 / abs(curvature)
    v_max_ms = np.sqrt(max_lat_acc * radius)
    v_max_kmh = v_max_ms * 3.6
    if speed_kmh > v_max_kmh * 0.9:
        excess = (speed_kmh - v_max_kmh * 0.9) / (v_max_kmh * 0.3 + 1e-6)
        return np.clip(excess, 0.0, 1.0)
    return 0.0

# =============================================
# LOOP PRINCIPAL
# =============================================
telemetry = Telemetry()
print("Autopilot iniciado. Pressione Ctrl+C para parar.")

steer_smooth = 0.0
heading_error_smooth = 0.0

try:
    while True:
        state = telemetry.read()

        # TAKEOFF AUTOMÁTICO
        if state['speed'] < TAKEOFF_SPEED:
            send_controls(0.0, 1.0, 0.0)
            time.sleep(0.02)
            continue

        heading_car = state['heading']
        pos_x = state['pos_x']
        pos_z = state['pos_z']
        lap_progress = state['lap_progress']

        heading_error, lateral_pos, c5, c10, c20, c30, c50, cdir = compute_track_features(
            lap_progress, heading_car, pos_x, pos_z
        )

        # Suavizar heading_error
        heading_error_smooth = HEADING_ALPHA * heading_error + (1 - HEADING_ALPHA) * heading_error_smooth

        # Calcular speed_error (excesso de velocidade para a curvatura a 10m)
        g = 9.81
        max_lat_acc = 0.8 * g
        future_curv = abs(c10)
        if future_curv > 0.001:
            radius = 1.0 / future_curv
            v_max_kmh = np.sqrt(max_lat_acc * radius) * 3.6
        else:
            v_max_kmh = 999.0
        speed_error = state['speed'] - v_max_kmh

        # Montar vetor de features (33 elementos)
        features = np.array([[
            state['speed'],
            state['rpm'],
            state['gear'],
            state['acc_g_x'], state['acc_g_y'], state['acc_g_z'],
            state['local_vel_x'], state['local_vel_y'], state['local_vel_z'],
            state['angular_vel_x'], state['angular_vel_y'], state['angular_vel_z'],
            state['slip_fl'], state['slip_fr'], state['slip_rl'], state['slip_rr'],
            state['tyre_temp_fl'], state['tyre_temp_fr'], state['tyre_temp_rl'], state['tyre_temp_rr'],
            state['lap_progress'],
            state['sector'],
            state['surface_grip'],
            state['tyres_out'],
            heading_error_smooth,
            lateral_pos,
            c5, c10, c20, c30, c50,
            cdir,
            speed_error
        ]], dtype=np.float32)

        features_scaled = scaler.transform(features)
        features_t = torch.tensor(features_scaled).to(device)

        with torch.no_grad():
            predictions = model(features_t).cpu().numpy()[0]
        steer_raw, throttle_raw, brake_raw = predictions

        # Freio de segurança baseado na curvatura atual (c10)
        brake_safety = safety_brake(state['speed'], c10)
        brake = max(brake_raw, brake_safety)
        brake = np.clip(brake, 0.0, 1.0)

        # Se estiver freando, corta acelerador
        if brake > 0.3:
            throttle = 0.0
        else:
            throttle = np.clip(throttle_raw, 0.0, 1.0)

        # Suavizar e escalar volante
        steer_smooth = STEER_ALPHA * steer_raw + (1 - STEER_ALPHA) * steer_smooth
        steer = steer_smooth * STEER_SCALE
        steer = np.clip(steer, -1.0, 1.0)

        # Debug
        print(f"\rSteer_raw={steer_raw:+.3f} final={steer:+.3f} Thr={throttle:.3f} Brake={brake:.3f} | "
              f"head_err={heading_error:+.3f} curv5={c5:+.4f} speed_err={speed_error:.1f} speed={state['speed']:.1f}   ", end='')

        send_controls(steer, throttle, brake)
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nAutopilot encerrado.")