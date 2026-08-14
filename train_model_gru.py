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
INPUT_CSV = 'data/processed/dataset_clean.csv'
MODEL_FILE = 'driving_model_gru.pth'
SCALER_FILE = 'scaler_gru.save'

SEQ_LEN = 10
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.0003
HIDDEN_SIZE = 64
NUM_LAYERS = 1

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

TARGET_COLS = ['steering_input', 'throttle_input', 'brake_input']

# =============================================
# DISPOSITIVO
# =============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# =============================================
# CARREGAR DATASET
# =============================================
print("Carregando dataset...")
df = pd.read_csv(INPUT_CSV)
print(f"Linhas: {len(df)}")

missing = [c for c in FEATURE_COLS + TARGET_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Colunas ausentes: {missing}")

# Separar por volta
train_laps = list(range(0, 16))   # voltas 0–15
val_laps   = [16, 17, 18]
test_laps  = [19, 20, 21]

train_idx = df.index[df['lap'].isin(train_laps)].to_numpy()
val_idx   = df.index[df['lap'].isin(val_laps)].to_numpy()
test_idx  = df.index[df['lap'].isin(test_laps)].to_numpy()

# =============================================
# CRIAR SEQUÊNCIAS (VERSÃO VETORIZADA)
# =============================================
def create_sequences_optimized(data, indices, seq_len):
    """
    Cria sequências de forma otimizada usando NumPy.
    Acessa os arrays diretamente, sem pandas, e verifica a continuidade das voltas.
    """
    # Extrai arrays subjacentes uma única vez
    feature_array = data[FEATURE_COLS].to_numpy(dtype=np.float32)
    target_array  = data[TARGET_COLS].to_numpy(dtype=np.float32)
    lap_array     = data['lap'].to_numpy() if 'lap' in data.columns else None

    X_list, y_list = [], []
    indices = np.asarray(indices)

    for i in range(len(indices) - seq_len + 1):
        seq_idx = indices[i:i+seq_len]

        # Verifica se todos os índices são da mesma volta
        if lap_array is not None:
            if not np.all(lap_array[seq_idx] == lap_array[seq_idx[0]]):
                continue

        # Extrai a sequência e o target (último frame)
        X_list.append(feature_array[seq_idx])
        y_list.append(target_array[seq_idx[-1]])

    return np.array(X_list), np.array(y_list)

print("Gerando sequências de treino...")
X_train_raw, y_train = create_sequences_optimized(df, train_idx, SEQ_LEN)
print("Gerando sequências de validação...")
X_val_raw, y_val = create_sequences_optimized(df, val_idx, SEQ_LEN)
print("Gerando sequências de teste...")
X_test_raw, y_test = create_sequences_optimized(df, test_idx, SEQ_LEN)

# =============================================
# NORMALIZAÇÃO
# =============================================
scaler = StandardScaler()
scaler.fit(X_train_raw.reshape(-1, len(FEATURE_COLS)))

X_train = scaler.transform(X_train_raw.reshape(-1, len(FEATURE_COLS))).reshape(X_train_raw.shape)
X_val   = scaler.transform(X_val_raw.reshape(-1, len(FEATURE_COLS))).reshape(X_val_raw.shape)
X_test  = scaler.transform(X_test_raw.reshape(-1, len(FEATURE_COLS))).reshape(X_test_raw.shape)

print(f"Treino: {len(X_train)} sequências")
print(f"Validação: {len(X_val)} sequências")
print(f"Teste: {len(X_test)} sequências")

# =============================================
# TENSORES
# =============================================
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_t   = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t   = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test_t  = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# =============================================
# MODELO GRU
# =============================================
class DrivingGRU(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.steering_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )
        self.throttle_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.brake_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        steer = self.steering_head(last)
        throttle = self.throttle_head(last)
        brake = self.brake_head(last)
        return torch.cat([steer, throttle, brake], dim=1)

input_dim = len(FEATURE_COLS)
model = DrivingGRU(input_dim, HIDDEN_SIZE, NUM_LAYERS).to(device)

# =============================================
# FUNÇÃO DE PERDA PONDERADA
# =============================================
class WeightedMSELoss(nn.Module):
    def __init__(self, weights=[2.0, 0.5, 1.0]):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, pred, target):
        diff = (pred - target) ** 2
        return (diff * self.weights.to(pred.device)).mean()

criterion = WeightedMSELoss(weights=[2.0, 0.5, 1.0])
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# =============================================
# TREINAMENTO
# =============================================
print("\nIniciando treinamento...")
train_losses, val_losses = [], []

best_val_loss = float('inf')
patience = 3
patience_counter = 0

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

    # Validação
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
    val_losses.append(val_loss)

    # ---------- EARLY STOPPING ----------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Salva o melhor modelo até agora
        torch.save(model.state_dict(), 'best_model_gru.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping na época {epoch+1}")
            break

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
# -----------------------------------------

# Após o loop, carrega o melhor modelo salvo
model.load_state_dict(torch.load('best_model_gru.pth', map_location=device))

# =============================================
# AVALIAÇÃO
# =============================================
mse_unweighted = nn.MSELoss()
model.eval()
with torch.no_grad():
    test_pred = model(X_test_t)
    test_loss = mse_unweighted(test_pred, y_test_t).item()
print(f"\nTest Loss (MSE simples): {test_loss:.4f}")

# =============================================
# SALVAR (agora salva o melhor modelo como final)
# =============================================
torch.save(model.state_dict(), MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
print(f"Modelo salvo em: {MODEL_FILE}")
print(f"Scaler salvo em: {SCALER_FILE}")

plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Treino')
plt.plot(val_losses, label='Validação')
plt.xlabel('Época')
plt.ylabel('Loss ponderada')
plt.legend()
plt.grid(True)
plt.show()