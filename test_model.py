import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from collections import deque

# ============================================================
# CONFIG
# ============================================================

CSV_FILE = "data/processed/dataset_clean.csv"   # ajuste se necessário
MODEL_FILE = "driving_model_gru.pth"            # modelo GRU salvo
SCALER_FILE = "scaler_gru.save"                 # scaler correspondente

SEQ_LEN = 20                                    # deve ser o mesmo usado no treinamento
BATCH_SIZE = 64

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

TARGET_COLS = [
    'steering_input',
    'throttle_input',
    'brake_input'
]

# ============================================================
# MODELO GRU (mesma arquitetura do treinamento)
# ============================================================

class DrivingGRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_dim, hidden_size, num_layers,
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


# ============================================================
# CARREGAR
# ============================================================

print("Carregando dataset...")
df = pd.read_csv(CSV_FILE)
print(f"Linhas: {len(df)}")
print(f"Features: {len(FEATURE_COLS)}")

missing = [col for col in FEATURE_COLS + TARGET_COLS if col not in df.columns]
if missing:
    print("ERRO: colunas ausentes:")
    for col in missing:
        print(" -", col)
    raise SystemExit

# Extrai arrays
X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
y = df[TARGET_COLS].to_numpy(dtype=np.float32)
lap = df['lap'].to_numpy() if 'lap' in df.columns else None

# ============================================================
# GERAR SEQUÊNCIAS (respeitando voltas)
# ============================================================

def create_sequences_by_lap(features, targets, laps, seq_len):
    X_seq, y_seq = [], []
    if laps is None:
        # se não houver coluna lap, trata como uma única sequência contínua
        for i in range(len(features) - seq_len + 1):
            X_seq.append(features[i:i+seq_len])
            y_seq.append(targets[i+seq_len-1])
    else:
        unique_laps = np.unique(laps)
        for lap_val in unique_laps:
            idx = np.where(laps == lap_val)[0]
            for i in range(len(idx) - seq_len + 1):
                seq_idx = idx[i:i+seq_len]
                X_seq.append(features[seq_idx])
                y_seq.append(targets[seq_idx[-1]])
    return np.array(X_seq), np.array(y_seq)

print("Gerando sequências...")
X_seq, y_seq = create_sequences_by_lap(X, y, lap, SEQ_LEN)
print(f"Sequências geradas: {len(X_seq)}")

# ============================================================
# SCALER
# ============================================================
print("Carregando scaler...")
scaler = joblib.load(SCALER_FILE)

# O scaler espera entrada 2D (n_features); transformamos as sequências planificadas
X_flat = X_seq.reshape(-1, len(FEATURE_COLS))
X_scaled_flat = scaler.transform(X_flat)
X_scaled = X_scaled_flat.reshape(X_seq.shape)

# ============================================================
# MODELO
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Carregando modelo...")
model = DrivingGRU(len(FEATURE_COLS)).to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()

# ============================================================
# PREDIÇÃO (em lotes para evitar memória excessiva)
# ============================================================
print("Realizando predições...")
predictions = []
with torch.no_grad():
    for i in range(0, len(X_scaled), BATCH_SIZE):
        batch = torch.tensor(X_scaled[i:i+BATCH_SIZE], dtype=torch.float32).to(device)
        pred = model(batch).cpu().numpy()
        predictions.append(pred)
predictions = np.vstack(predictions)

# ============================================================
# MÉTRICAS
# ============================================================
steer_real = y_seq[:, 0]
throttle_real = y_seq[:, 1]
brake_real = y_seq[:, 2]

steer_pred = predictions[:, 0]
throttle_pred = predictions[:, 1]
brake_pred = predictions[:, 2]

def mae(real, pred):
    return np.mean(np.abs(real - pred))

print("\n==============================")
print("RESULTADO DO MODELO")
print("==============================")
print(f"Steering MAE : {mae(steer_real, steer_pred):.4f}")
print(f"Throttle MAE : {mae(throttle_real, throttle_pred):.4f}")
print(f"Brake MAE    : {mae(brake_real, brake_pred):.4f}")

# ============================================================
# ESTATÍSTICAS
# ============================================================
print("\n==============================")
print("DISTRIBUIÇÃO")
print("==============================")
print("\nSTEERING REAL")
print(f"min={steer_real.min():.3f} max={steer_real.max():.3f} mean={steer_real.mean():.3f} std={steer_real.std():.3f}")
print("\nSTEERING PREDITO")
print(f"min={steer_pred.min():.3f} max={steer_pred.max():.3f} mean={steer_pred.mean():.3f} std={steer_pred.std():.3f}")
print("\nTHROTTLE REAL")
print(f"min={throttle_real.min():.3f} max={throttle_real.max():.3f} mean={throttle_real.mean():.3f} std={throttle_real.std():.3f}")
print("\nTHROTTLE PREDITO")
print(f"min={throttle_pred.min():.3f} max={throttle_pred.max():.3f} mean={throttle_pred.mean():.3f} std={throttle_pred.std():.3f}")

# ============================================================
# GRÁFICOS
# ============================================================
N = min(3000, len(y_seq))
plt.figure(figsize=(14,5))
plt.plot(steer_real[:N], label="Steering real")
plt.plot(steer_pred[:N], label="Steering IA")
plt.title("Steering — Real vs IA")
plt.xlabel("Amostra")
plt.ylabel("Steering")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("comparison/steering_comparison.png", dpi=150)
plt.show()

plt.figure(figsize=(14,5))
plt.plot(throttle_real[:N], label="Throttle real")
plt.plot(throttle_pred[:N], label="Throttle IA")
plt.title("Throttle — Real vs IA")
plt.xlabel("Amostra")
plt.ylabel("Throttle")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("comparison/throttle_comparison.png", dpi=150)
plt.show()

plt.figure(figsize=(14,5))
plt.plot(brake_real[:N], label="Brake real")
plt.plot(brake_pred[:N], label="Brake IA")
plt.title("Brake — Real vs IA")
plt.xlabel("Amostra")
plt.ylabel("Brake")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("comparison/brake_comparison.png", dpi=150)
plt.show()

print("\nTest completed.")