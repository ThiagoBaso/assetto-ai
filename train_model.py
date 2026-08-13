import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import matplotlib.pyplot as plt

# =============================================
# CONFIGURAÇÕES
# =============================================
INPUT_CSV = 'data/processed/dataset_with_track.csv'
MODEL_FILE = 'driving_model.pth'
SCALER_FILE = 'scaler.save'

FEATURE_COLS = [
    'speed', 'rpm', 'gear',
    'acc_g_x', 'acc_g_y', 'acc_g_z',
    'local_vel_x', 'local_vel_y', 'local_vel_z',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
    'slip_fl', 'slip_fr', 'slip_rl', 'slip_rr',
    'tyre_temp_fl', 'tyre_temp_fr', 'tyre_temp_rl', 'tyre_temp_rr',
    'lap_progress', 'sector', 'surface_grip', 'tyres_out',
    'heading_error', 'lateral_position',
    'curvature_0m',            # curvatura no ponto atual (nova)
    'curvature_5m', 'curvature_10m', 'curvature_20m', 'curvature_30m', 'curvature_50m',
    'curve_direction',
    'speed_error',
    's_current'                # posição longitudinal contínua (nova)
]

TARGET_COLS = ['steering_input', 'throttle_input', 'brake_input']

TEST_SIZE = 0.1
VAL_SIZE = 0.1
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_LAYERS = [256, 256, 128]

# =============================================
# DISPOSITIVO
# =============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# =============================================
# CARREGAR E PREPARAR DADOS
# =============================================
print("\nCarregando dataset...")
df = pd.read_csv(INPUT_CSV)

missing_features = [c for c in FEATURE_COLS if c not in df.columns]
missing_targets = [c for c in TARGET_COLS if c not in df.columns]
if missing_features:
    raise ValueError(f"Colunas de feature ausentes: {missing_features}")
if missing_targets:
    raise ValueError(f"Colunas de target ausentes: {missing_targets}")

X = df[FEATURE_COLS].values
y = df[TARGET_COLS].values

# =============================================
# SPLIT POR VOLTA (para evitar overfitting temporal)
# =============================================
train_laps = [0, 1, 2, 3, 4, 5, 6, 7, 8]
val_laps   = [9, 10]
test_laps  = [11, 12]

train_idx = df.index[df['lap'].isin(train_laps)].to_numpy()
val_idx   = df.index[df['lap'].isin(val_laps)].to_numpy()
test_idx  = df.index[df['lap'].isin(test_laps)].to_numpy()

print(f"Treino: {len(train_idx)} | Validação: {len(val_idx)} | Teste: {len(test_idx)}")

# =============================================
# SCALER SOMENTE NO TREINO
# =============================================
scaler = StandardScaler()
scaler.fit(X[train_idx])

X_train = scaler.transform(X[train_idx])
X_val   = scaler.transform(X[val_idx])
X_test  = scaler.transform(X[test_idx])

y_train = y[train_idx]
y_val   = y[val_idx]
y_test  = y[test_idx]

# =============================================
# TENSORES
# =============================================
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# =============================================
# MODELO
# =============================================
class DrivingModel(nn.Module):
    def __init__(self, input_dim, hidden_layers=[256, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        self.shared = nn.Sequential(*layers)

        # Cabeças de saída com ativações corretas
        self.steering_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Tanh()
        )
        self.throttle_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
        self.brake_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        shared = self.shared(x)
        steer = self.steering_head(shared)
        throttle = self.throttle_head(shared)
        brake = self.brake_head(shared)
        return torch.cat([steer, throttle, brake], dim=1)

input_dim = len(FEATURE_COLS)
model = DrivingModel(input_dim, hidden_layers=HIDDEN_LAYERS).to(device)

# =============================================
# FUNÇÃO DE PERDA PONDERADA
# =============================================
class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, pred, target):
        diff = (pred - target) ** 2
        weighted_diff = diff * self.weights.to(pred.device)
        return weighted_diff.mean()

# Pesos: maior para steering (precisão na direção)
criterion = WeightedMSELoss(weights=[2.0, 0.5, 1.0])

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =============================================
# TREINAMENTO
# =============================================
print("\nIniciando treinamento...")
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

# =============================================
# AVALIAÇÃO FINAL
# =============================================
mse_unweighted = nn.MSELoss()
model.eval()
with torch.no_grad():
    test_pred = model(X_test_t)
    test_loss = mse_unweighted(test_pred, y_test_t).item()
print(f"\nTest Loss (MSE simples): {test_loss:.4f}")

mse_per_action = np.mean((test_pred.cpu().numpy() - y_test)**2, axis=0)
for i, col in enumerate(TARGET_COLS):
    print(f"  MSE {col}: {mse_per_action[i]:.4f}")

# =============================================
# SALVAR
# =============================================
torch.save(model.state_dict(), MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
print(f"\nModelo salvo em: {MODEL_FILE}")
print(f"Scaler salvo em: {SCALER_FILE}")

# =============================================
# GRÁFICOS
# =============================================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Treino')
plt.plot(val_losses, label='Validação')
plt.xlabel('Época')
plt.ylabel('Loss (ponderada)')
plt.title('Curva de aprendizado')
plt.legend()
plt.grid(True)
plt.show()