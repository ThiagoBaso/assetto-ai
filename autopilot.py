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

STEER_ALPHA = 0.05          # suavização do volante (maior = mais responsivo)
STEER_SCALE = 0.5           # fator de escala (a rede já limita a saída)
HEADING_ALPHA = 0.25        # suavização do heading_error
TAKEOFF_SPEED = 10.0        # km/h abaixo do qual força aceleração máxima

FEATURE_COLS = [
    'speed', 'rpm', 'gear',
    'acc_g_x', 'acc_g_y', 'acc_g_z',
    'local_vel_x', 'local_vel_y', 'local_vel_z',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
    'slip_fl', 'slip_fr', 'slip_rl', 'slip_rr',
    'tyre_temp_fl', 'tyre_temp_fr', 'tyre_temp_rl', 'tyre_temp_rr',
    'lap_progress', 'sector', 'surface_grip', 'tyres_out',
    'heading_error', 'lateral_position',
    'curvature_0m',            # curvatura atual
    'curvature_5m', 'curvature_10m', 'curvature_20m', 'curvature_30m', 'curvature_50m',
    'curve_direction',
    'speed_error',
    's_current'                # posição longitudinal contínua
]

print("Features:", len(FEATURE_COLS))

# =============================================
# CARREGAR MODELO, SCALER E TRAJETÓRIA
# =============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DrivingModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers=[256, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(torch.nn.Linear(prev_dim, h))
            layers.append(torch.nn.ReLU())
            prev_dim = h

        self.shared = torch.nn.Sequential(*layers)

        # Saídas com ativações corretas
        self.steering_head = torch.nn.Sequential(
            torch.nn.Linear(prev_dim, 1),
            torch.nn.Tanh()
        )
        self.throttle_head = torch.nn.Sequential(
            torch.nn.Linear(prev_dim, 1),
            torch.nn.Sigmoid()
        )
        self.brake_head = torch.nn.Sequential(
            torch.nn.Linear(prev_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        shared = self.shared(x)
        steer = self.steering_head(shared)
        throttle = self.throttle_head(shared)
        brake = self.brake_head(shared)
        return torch.cat([steer, throttle, brake], dim=1)


print("Carregando modelo...")

input_dim = len(FEATURE_COLS)
model = DrivingModel(input_dim).to(device)
model.load_state_dict(torch.load('driving_model.pth', map_location=device))
model.eval()

scaler = joblib.load('scaler.save')
traj_data = joblib.load('trajectory.pkl')
heading_offset = traj_data.get('heading_offset', 0.0)

# =============================================
# RECONSTRUIR INTERPOLADORES
# =============================================

s = traj_data['s']
total_length = traj_data['total_length']
progress_bins = traj_data['progress_bins']

s_to_x = interp1d(s, traj_data['x'], kind='cubic', fill_value='extrapolate')
s_to_z = interp1d(s, traj_data['z'], kind='cubic', fill_value='extrapolate')
s_to_heading = interp1d(s, traj_data['heading'], kind='cubic', fill_value='extrapolate')
s_to_curv = interp1d(s, traj_data['curvature'], kind='cubic', fill_value='extrapolate')

# =============================================
# BUSCA DO PONTO MAIS PRÓXIMO DA TRAJETÓRIA
# =============================================

trajectory_x = np.asarray(traj_data['x'])
trajectory_z = np.asarray(traj_data['z'])
trajectory_s = np.asarray(traj_data['s'])

s_previous = 0.0  # será atualizado a cada iteração

def find_nearest_s(pos_x, pos_z, s_prev=None):
    """Busca o ponto da trajetória mais próximo, com janela se tiver estimativa anterior."""
    if s_prev is not None:
        # Janela de ±50m ao redor da estimativa anterior
        idx_window = np.where(
            (trajectory_s >= s_prev - 50) &
            (trajectory_s <= s_prev + 50)
        )[0]
        if len(idx_window) > 0:
            dx = trajectory_x[idx_window] - pos_x
            dz = trajectory_z[idx_window] - pos_z
            dist_sq = dx*dx + dz*dz
            nearest_idx = idx_window[np.argmin(dist_sq)]
            return trajectory_s[nearest_idx], np.sqrt(dist_sq.min())
    # Fallback: busca global
    dx = trajectory_x - pos_x
    dz = trajectory_z - pos_z
    dist_sq = dx*dx + dz*dz
    nearest_idx = np.argmin(dist_sq)
    return trajectory_s[nearest_idx], np.sqrt(dist_sq[nearest_idx])

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

def compute_track_features(heading_car, pos_x, pos_z, heading_offset, speed=None):
    """
    Retorna heading_error, lateral_position, curvaturas futuras, curve_direction,
    track_distance, curvature_0m, s_current.
    """
    # Encontrar ponto mais próximo
    s_now, track_distance = find_nearest_s(pos_x, pos_z, s_previous)
    heading_traj_now = s_to_heading(s_now)

    # Aplicar offset fixo carregado
    heading_car_corrected = heading_car + heading_offset
    heading_error = np.angle(np.exp(1j * (heading_car_corrected - heading_traj_now)))

    # Curvaturas futuras
    future_dists = [5, 10, 20, 30, 50]
    curvatures = []
    for d in future_dists:
        s_future = np.mod(s_now + d, total_length)
        curvatures.append(float(s_to_curv(s_future)))
    curve_direction = np.sign(curvatures[0])

    # Posição lateral
    x_traj = float(s_to_x(s_now))
    z_traj = float(s_to_z(s_now))
    dx_traj = np.cos(heading_traj_now)
    dz_traj = np.sin(heading_traj_now)

    lateral_pos = np.sqrt((pos_x - x_traj)**2 + (pos_z - z_traj)**2)
    lateral_sign = np.sign((pos_x - x_traj)*(-dz_traj) + (pos_z - z_traj)*dx_traj)
    lateral_pos *= lateral_sign

    # Curvatura atual (0m)
    curvature_0m = float(s_to_curv(s_now))

    return (
        heading_error,
        lateral_pos,
        curvatures[0],   # c5
        curvatures[1],   # c10
        curvatures[2],   # c20
        curvatures[3],   # c30
        curvatures[4],   # c50
        curve_direction,
        track_distance,
        curvature_0m,
        s_now            # s_current
    )


# =============================================
# FREIO DE SEGURANÇA
# =============================================

def safety_brake(speed_kmh, curvature):
    if abs(curvature) < 0.001:
        return 0.0
    g = 9.81
    max_lat_acc = 1.0 * g
    radius = 1.0 / abs(curvature)
    v_max_ms = np.sqrt(max_lat_acc * radius)
    v_max_kmh = v_max_ms * 3.6
    if speed_kmh > v_max_kmh * 0.95:
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
s_previous = 0.0
from recovery_logger import RecoveryLogger
logger = RecoveryLogger("recovery_log.csv")

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

        # SENSORES DA TRAJETÓRIA
        (
            heading_error,
            lateral_pos,
            c5, c10, c20, c30, c50,
            cdir,
            track_distance,
            curvature_0m,
            s_current
        ) = compute_track_features(
            heading_car,
            pos_x,
            pos_z,
            heading_offset,
            state['speed']   # pode ser usado para calibração futura (não necessário agora)
        )

        # Atualizar s_previous para busca refinada na próxima iteração
        s_previous = s_current

        # SUAVIZAÇÃO DO HEADING
        heading_error_smooth = (
            HEADING_ALPHA * heading_error +
            (1 - HEADING_ALPHA) * heading_error_smooth
        )

        # SPEED ERROR
        g = 9.81
        max_lat_acc = 0.8 * g
        future_curv = abs(c10)
        if future_curv > 0.001:
            radius = 1.0 / future_curv
            v_max_kmh = np.sqrt(max_lat_acc * radius) * 3.6
            speed_error = state['speed'] - v_max_kmh
        else:
            v_max_kmh = 999.0
            speed_error = state['speed'] - v_max_kmh

        # MONTAR VETOR DE FEATURES
        features = np.array([[
            state['speed'],
            state['rpm'],
            state['gear'],
            state['acc_g_x'],
            state['acc_g_y'],
            state['acc_g_z'],
            state['local_vel_x'],
            state['local_vel_y'],
            state['local_vel_z'],
            state['angular_vel_x'],
            state['angular_vel_y'],
            state['angular_vel_z'],
            state['slip_fl'],
            state['slip_fr'],
            state['slip_rl'],
            state['slip_rr'],
            state['tyre_temp_fl'],
            state['tyre_temp_fr'],
            state['tyre_temp_rl'],
            state['tyre_temp_rr'],
            state['lap_progress'],
            state['sector'],
            state['surface_grip'],
            state['tyres_out'],
            heading_error_smooth,
            lateral_pos,
            curvature_0m,
            c5,
            c10,
            c20,
            c30,
            c50,
            cdir,
            speed_error,
            s_current
        ]], dtype=np.float32)

        # NORMALIZAÇÃO
        features_scaled = scaler.transform(features)
        features_t = torch.tensor(features_scaled, dtype=torch.float32).to(device)

        # PREDIÇÃO
        with torch.no_grad():
            predictions = model(features_t).cpu().numpy()[0]

        steer_raw, throttle_raw, brake_raw = predictions

        # FREIO DE SEGURANÇA
        brake_safety = safety_brake(state['speed'], c10)
        brake = max(brake_raw, brake_safety)
        brake = np.clip(brake, 0.0, 1.0)

        # ACELERADOR
        if brake > 0.3:
            throttle = 0.0
        else:
            throttle = np.clip(throttle_raw, 0.0, 1.0)

        # VOLANTE
        steer_smooth = STEER_ALPHA * steer_raw + (1 - STEER_ALPHA) * steer_smooth
        steer = steer_smooth * STEER_SCALE
        steer = np.clip(steer, -1.0, 1.0)

        # DEBUG
        print(
            f"\r"
            f"Steer_raw={steer_raw:+.3f} "
            f"final={steer:+.3f} "
            f"Thr={throttle:.3f} "
            f"Brake={brake:.3f} | "
            f"head_err={heading_error:+.3f} "
            f"curv0={curvature_0m:+.4f} "
            f"curv10={c10:+.4f} "
            f"speed_err={speed_error:+.1f} "
            f"vmax={v_max_kmh if v_max_kmh != 999.0 else 'straight'} "
            f"speed={state['speed']:.1f} "
            f"progress={lap_progress:.4f} "
            f"track_dist={track_distance:.2f}m   ",
            end=''
        )

        # ENVIAR CONTROLES
        send_controls(steer, throttle, brake)
        time.sleep(0.02)

        logger.log(
            s_current=s_current,
            lateral_position=lateral_pos,
            heading_error=heading_error_smooth,
            curvature_0m=curvature_0m,
            curvature_10m=c10,
            speed_error=speed_error,
            tyres_out=state['tyres_out'],
            surface_grip=state['surface_grip'],
            steer_pred=steer_raw,
            throttle_pred=throttle_raw,
            brake_pred=brake_raw,
        )

except KeyboardInterrupt:
    send_controls(0.0, 0.0, 0.0)
    print("\nAutopilot encerrado.")