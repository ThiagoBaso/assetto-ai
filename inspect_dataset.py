import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/dataset_with_track.csv')

# Plota uma volta por vez
for lap in sorted(df['lap'].unique()):
    df_lap = df[df['lap'] == lap]
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(df_lap['s_current'], df_lap['steering_input'], label='Steering')
    plt.ylabel('Steering')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(df_lap['s_current'], df_lap['brake_input'], label='Brake', color='red')
    plt.ylabel('Brake')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(df_lap['s_current'], df_lap['speed'], label='Speed', color='green')
    plt.xlabel('Distância (m)')
    plt.ylabel('Speed (km/h)')
    plt.grid(True)
    
    plt.suptitle(f'Volta {lap}')
    plt.tight_layout()
    plt.show()