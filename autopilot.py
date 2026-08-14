import time
import numpy as np
import torch
import joblib
from scipy.interpolate import interp1d
import pyvjoy
from collections import deque

from telemetry import Telemetry

# =============================================
# CONFIGURAÇÕES AJUSTÁVEIS
# =============================================

STEER_ALPHA = 0.07           # suavização do volante (maior = mais responsivo)
STEER_SCALE = 0.25           # fator de escala (reduzir se instável)
HEADING_ALPHA = 0.25        # suavização do heading_error
TAKEOFF_SPEED = 10.0        # km/h abaixo do qual força aceleração máxima
SEQ_LEN = 20                 # mesmo valor usado no treinamento

OFF_TRACK_ENTER = 99.0   # só considera lateral_position se estiver muito longe
OFF_TRACK_EXIT = 3.0
MAX_STEER_DELTA = 0.03    # limite de variação por frame (0.08 = suave)

FEATURE_COLS = [
    'speed', 'rpm', 'gear',
    'acc_g_x', 'acc_g_y', 'acc_g_z',
    'local_vel_x', 'local_vel_y', 'local_vel_z',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
    'slip_fl', 'slip_fr', 'slip_rl', 'slip_rr',
    'tyre_temp_fl', 'tyre_temp_fr', 'tyre_temp_rl', 'tyre_temp_rr',
    'lap_progress', 'sector', 'surface_grip', 'tyres_out',
    'heading_error', 'lateral_position',
    'curvature_0m', 'curvature_5m', 'curvature_10m',
    'curvature_20m', 'curvature_30m', 'curvature_50m',
    'curve_direction', 'speed_error', 's_current'
]

# =============================================
# CARREGAR MODELO, SCALER E TRAJETÓRIA
# =============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DrivingGRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.steering_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Tanh()
        )
        self.throttle_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )
        self.brake_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        steer = self.steering_head(last)
        throttle = self.throttle_head(last)
        brake = self.brake_head(last)
        return torch.cat([steer, throttle, brake], dim=1)

print("Carregando modelo...")
model = DrivingGRU(len(FEATURE_COLS)).to(device)
model.load_state_dict(torch.load('driving_model_gru.pth', map_location=device))
model.eval()

scaler = joblib.load('scaler_gru.save')
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

trajectory_x = np.asarray(traj_data['x'])
trajectory_z = np.asarray(traj_data['z'])
trajectory_s = np.asarray(traj_data['s'])

s_previous = 0.0

def find_nearest_s(pos_x, pos_z, s_prev=None, max_window_dist=15.0):
    if s_prev is not None:
        idx_window = np.where((trajectory_s >= s_prev - 50) & (trajectory_s <= s_prev + 50))[0]
        if len(idx_window) > 0:
            dx = trajectory_x[idx_window] - pos_x
            dz = trajectory_z[idx_window] - pos_z
            dist_sq = dx*dx + dz*dz
            nearest_idx = idx_window[np.argmin(dist_sq)]
            dist = np.sqrt(dist_sq.min())
            if dist <= max_window_dist:
                return trajectory_s[nearest_idx], dist
            # janela não contém um ponto plausível -> cai pro fallback global abaixo

    # Busca global (primeiro frame, ou quando a janela local não é confiável)
    dx = trajectory_x - pos_x
    dz = trajectory_z - pos_z
    dist_sq = dx*dx + dz*dz
    nearest_idx = np.argmin(dist_sq)
    return trajectory_s[nearest_idx], np.sqrt(dist_sq[nearest_idx])

# =============================================
# JOYSTICK VIRTUAL
# =============================================
vj = pyvjoy.VJoyDevice(1)

def send_controls(steer, throttle, brake):
    if abs(steer) < 0.03:   # pequeno demais?
        steer = 0.0
    steer_val = int((steer + 1.0) * 0x4000)
    throttle_val = int(throttle * 0x8000)
    brake_val = int(brake * 0x8000)
    vj.set_axis(pyvjoy.HID_USAGE_X, steer_val)
    vj.set_axis(pyvjoy.HID_USAGE_Y, throttle_val)
    vj.set_axis(pyvjoy.HID_USAGE_Z, brake_val)

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
# RECUPERAÇÃO MANUAL
# =============================================
def recovery_control(speed, lateral_pos, heading_error):
    print('recovery_control')
    brake = 0.0
    throttle = 0.0
    if speed > 40:
        brake = min(0.5, (speed - 40) * 0.02)
    else:
        throttle = 0.3
    # Ganho reduzido + termo de heading
    steer = np.clip(
        -np.sign(lateral_pos) * min(1.0, abs(lateral_pos) * 0.3) - 0.2 * heading_error,
        -1.0, 1.0
    )
    if abs(lateral_pos) < 0.1:
        steer = np.clip(-heading_error * 0.5, -1.0, 1.0)
    return steer, throttle, brake

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
        print('safety_brake')
        return np.clip(excess, 0.0, 1.0)
    return 0.0

# =============================================
# LOOP PRINCIPAL
# =============================================

telemetry = Telemetry()
print("Autopilot iniciado. Pressione Ctrl+C para parar.")

# Variáveis de suavização separadas para IA e recuperação
steer_smooth_ai = 0.0
steer_smooth_recovery = 0.0

# Variáveis de last steering para rate limiter (separadas também)
steer_prev_ai = 0.0
steer_prev_recovery = 0.0

heading_error_smooth = 0.0
recovering = False

# Inicializa s_previous com a posição atual
state0 = telemetry.read()
s_previous, _ = find_nearest_s(state0['pos_x'], state0['pos_z'], s_prev=None)

# Buffer para contexto temporal
feature_buffer = deque(maxlen=SEQ_LEN)

try:
    while True:
        state = telemetry.read()
        steer_raw = throttle_raw = brake_raw = 0.0

        if state['speed'] < TAKEOFF_SPEED:
            send_controls(0.0, 1.0, 0.0)
            time.sleep(0.02)
            feature_buffer.clear()
            # Reseta suavizações quando parado
            steer_smooth_ai = 0.0
            steer_smooth_recovery = 0.0
            steer_prev_ai = 0.0
            steer_prev_recovery = 0.0
            continue

        heading_car = state['heading']
        pos_x = state['pos_x']
        pos_z = state['pos_z']
        lap_progress = state['lap_progress']

        # Sensores
        (
            heading_error,
            lateral_pos,
            c5, c10, c20, c30, c50,
            cdir,
            track_distance,
            curvature_0m,
            s_current
        ) = compute_track_features(heading_car, pos_x, pos_z, heading_offset, state['speed'])

        s_previous = s_current

        heading_error_smooth = (
            HEADING_ALPHA * heading_error +
            (1 - HEADING_ALPHA) * heading_error_smooth
        )

        # Verificar off-track com prioridade para pneus/aderência
        # off_track = (
        #     state['tyres_out'] > 0 or
        #     state['surface_grip'] < 0.85 or
        #     abs(lateral_pos) > OFF_TRACK_ENTER  # só usa lateral como último recurso
        # )
        # if not off_track and recovering:
        #     # Se está recuperando e os sinais fortes sumiram, sai apenas quando lateral < EXIT
        #     off_track = abs(lateral_pos) > OFF_TRACK_EXIT
        # recovering = off_track

        # if off_track:
        #     print(f"OFF_TRACK | lat={lateral_pos:+.3f} tyres_out={state['tyres_out']} grip={state['surface_grip']:.2f}")
        #     steer, throttle, brake = recovery_control(
        #         state['speed'], lateral_pos, heading_error_smooth
        #     )

        #     # Suavização específica da recuperação
        #     steer_smooth_recovery = STEER_ALPHA * steer + (1 - STEER_ALPHA) * steer_smooth_recovery
        #     steer = np.clip(steer_smooth_recovery, -1.0, 1.0)

        #     # Rate limiter da recuperação
        #     delta = np.clip(steer - steer_prev_recovery, -MAX_STEER_DELTA, MAX_STEER_DELTA)
        #     steer = np.clip(steer_prev_recovery + delta, -1.0, 1.0)
        #     steer_prev_recovery = steer

        #     send_controls(steer, throttle, brake)
        #     time.sleep(0.02)
        #     continue

        # Calcular speed_error
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

        # Montar feature atual
        feature_vector = np.array([[
            state['speed'], state['rpm'], state['gear'],
            state['acc_g_x'], state['acc_g_y'], state['acc_g_z'],
            state['local_vel_x'], state['local_vel_y'], state['local_vel_z'],
            state['angular_vel_x'], state['angular_vel_y'], state['angular_vel_z'],
            state['slip_fl'], state['slip_fr'], state['slip_rl'], state['slip_rr'],
            state['tyre_temp_fl'], state['tyre_temp_fr'], state['tyre_temp_rl'], state['tyre_temp_rr'],
            state['lap_progress'], state['sector'], state['surface_grip'], state['tyres_out'],
            heading_error_smooth, lateral_pos,
            curvature_0m, c5, c10, c20, c30, c50, cdir,
            speed_error, s_current
        ]], dtype=np.float32)

        feature_buffer.append(feature_vector[0])

        # Só prediz se tivermos pelo menos SEQ_LEN frames
        if len(feature_buffer) == SEQ_LEN:
            seq = np.array(feature_buffer, dtype=np.float32)
            seq_scaled = scaler.transform(seq)
            seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                predictions = model(seq_tensor).cpu().numpy()[0]
            steer_raw, throttle_raw, brake_raw = predictions

            # Freio de segurança
            brake_safety = safety_brake(state['speed'], c10)
            brake = max(brake_raw, brake_safety)
            brake = np.clip(brake, 0.0, 1.0)

            if brake > 0.3:
                throttle = 0.0
            else:
                throttle = np.clip(throttle_raw, 0.0, 1.0)

            # Suavização específica da IA
            steer_smooth_ai = STEER_ALPHA * steer_raw + (1 - STEER_ALPHA) * steer_smooth_ai
            steer = steer_smooth_ai * STEER_SCALE
            steer = np.clip(steer, -1.0, 1.0)

            # Rate limiter da IA
            delta = np.clip(steer - steer_prev_ai, -MAX_STEER_DELTA, MAX_STEER_DELTA)
            steer = np.clip(steer_prev_ai + delta, -1.0, 1.0)
            steer_prev_ai = steer
        else:
            # Ainda não tem sequência suficiente, manter controle neutro
            throttle = 0.2
            brake = 0.0
            steer = 0.0
            steer_raw = throttle_raw = brake_raw = 0.0

        # Debug (removidos prints de scaler desnecessários)
        print(
            f"AI | steer={steer:+.3f} thr={throttle:+.3f} brake={brake:+.3f} | "
            f"speed={state['speed']:.1f} head={heading_error:+.3f} "
            f"headOff={heading_offset:+.3f} lat={lateral_pos:+.3f} "
            f"c5={c5:+.4f} c10={c10:+.4f}",
            flush=True
        )

        send_controls(steer, throttle, brake)
        time.sleep(0.02)

except KeyboardInterrupt:
    send_controls(0.0, 0.0, 0.0)
    print("\nAutopilot encerrado.")