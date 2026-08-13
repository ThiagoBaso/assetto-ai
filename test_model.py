import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

CSV_FILE = "data/processed/dataset_with_track.csv"
MODEL_FILE = "driving_model.pth"
SCALER_FILE = "scaler.save"

FEATURE_COLS = [
    'speed', 'rpm', 'gear',
    'acc_g_x', 'acc_g_y', 'acc_g_z',
    'local_vel_x', 'local_vel_y', 'local_vel_z',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
    'slip_fl', 'slip_fr', 'slip_rl', 'slip_rr',
    'tyre_temp_fl', 'tyre_temp_fr', 'tyre_temp_rl', 'tyre_temp_rr',
    'lap_progress', 'sector', 'surface_grip', 'tyres_out',
    'heading_error', 'lateral_position',
    'curvature_0m',            # <-- ADICIONADA
    'curvature_5m', 'curvature_10m', 'curvature_20m', 'curvature_30m', 'curvature_50m',
    'curve_direction',
    'speed_error',
    's_current'                # <-- ADICIONADA
]

TARGET_COLS = [
    'steering_input',
    'throttle_input',
    'brake_input'
]


# ============================================================
# MODELO
# ============================================================

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


# ============================================================
# CARREGAR
# ============================================================

print("Carregando dataset...")

df = pd.read_csv(CSV_FILE)

print("Linhas:", len(df))
print("Features encontradas:", len(FEATURE_COLS))

missing = [
    col for col in FEATURE_COLS + TARGET_COLS
    if col not in df.columns
]

if missing:
    print("\nERRO: colunas ausentes:")
    for col in missing:
        print(" -", col)
    raise SystemExit


X = df[FEATURE_COLS].values.astype(np.float32)
Y = df[TARGET_COLS].values.astype(np.float32)


print("Carregando scaler...")
scaler = joblib.load(SCALER_FILE)

X_scaled = scaler.transform(X)


print("Carregando modelo...")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = DrivingModel(len(FEATURE_COLS)).to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.eval()


# ============================================================
# PREDIÇÃO
# ============================================================

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    predictions = model(
        X_tensor
    ).cpu().numpy()


# ============================================================
# MÉTRICAS
# ============================================================

steer_real = Y[:, 0]
throttle_real = Y[:, 1]
brake_real = Y[:, 2]

steer_pred = predictions[:, 0]
throttle_pred = predictions[:, 1]
brake_pred = predictions[:, 2]


def mae(real, pred):

    return np.mean(
        np.abs(real - pred)
    )


print("\n==============================")
print("RESULTADO DO MODELO")
print("==============================")

print(
    f"Steering MAE : {mae(steer_real, steer_pred):.4f}"
)

print(
    f"Throttle MAE : {mae(throttle_real, throttle_pred):.4f}"
)

print(
    f"Brake MAE    : {mae(brake_real, brake_pred):.4f}"
)


# ============================================================
# ESTATÍSTICAS
# ============================================================

print("\n==============================")
print("DISTRIBUIÇÃO")
print("==============================")

print("\nSTEERING REAL")
print(
    f"min={steer_real.min():.3f} "
    f"max={steer_real.max():.3f} "
    f"mean={steer_real.mean():.3f} "
    f"std={steer_real.std():.3f}"
)

print("\nSTEERING PREDITO")
print(
    f"min={steer_pred.min():.3f} "
    f"max={steer_pred.max():.3f} "
    f"mean={steer_pred.mean():.3f} "
    f"std={steer_pred.std():.3f}"
)

print("\nTHROTTLE REAL")
print(
    f"min={throttle_real.min():.3f} "
    f"max={throttle_real.max():.3f} "
    f"mean={throttle_real.mean():.3f} "
    f"std={throttle_real.std():.3f}"
)

print("\nTHROTTLE PREDITO")
print(
    f"min={throttle_pred.min():.3f} "
    f"max={throttle_pred.max():.3f} "
    f"mean={throttle_pred.mean():.3f} "
    f"std={throttle_pred.std():.3f}"
)


# ============================================================
# GRÁFICO STEERING
# ============================================================

N = min(3000, len(df))

plt.figure(figsize=(14, 5))

plt.plot(
    steer_real[:N],
    label="Steering real"
)

plt.plot(
    steer_pred[:N],
    label="Steering IA"
)

plt.title("Steering — Real vs IA")
plt.xlabel("Amostra")
plt.ylabel("Steering")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "comparison/steering_comparison.png",
    dpi=150
)

plt.show()


# ============================================================
# GRÁFICO THROTTLE
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    throttle_real[:N],
    label="Throttle real"
)

plt.plot(
    throttle_pred[:N],
    label="Throttle IA"
)

plt.title("Throttle — Real vs IA")
plt.xlabel("Amostra")
plt.ylabel("Throttle")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "comparison/throttle_comparison.png",
    dpi=150
)

plt.show()


# ============================================================
# GRÁFICO BRAKE
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    brake_real[:N],
    label="Brake real"
)

plt.plot(
    brake_pred[:N],
    label="Brake IA"
)

plt.title("Brake — Real vs IA")
plt.xlabel("Amostra")
plt.ylabel("Brake")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "comparison/brake_comparison.png",
    dpi=150
)

plt.show()

print("\nTest completed.")