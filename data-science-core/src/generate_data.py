import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def simulate_driver_stream(records=1000, seed=42):
    """
    Simula um fluxo contínuo de dados de telemetria e sensores de cabine.
    Cria uma transição gradual de um estado Alerta (Safe) para Fadiga (Risk).
    """
    np.random.seed(seed)
    
    # Criando uma sequência de timestamp minuto a minuto
    start_time = datetime.now()
    timestamps = [start_time + timedelta(minutes=i) for i in range(records)]
    
    data = []
    
    for i in range(records):
        # Fator de progressão: quanto mais tempo dirige, maior o cansaço acumulado (0.0 a 1.0)
        progression = i / records 
        
        # 1. PERCLOS: olhos fechados aumentam com o cansaço + ruído estocástico
        perclos = 0.04 + (progression * 0.18) + np.random.normal(0, 0.02)
        perclos = max(0.0, min(1.0, perclos)) # Truncar limites entre 0 e 1
        
        # 2. Bocejos por minuto: aumentam na segunda metade da viagem
        yawn_base = 0 if progression < 0.4 else (progression * 3)
        yawns = int(np.random.poisson(yawn_base))
        
        # 3. Micro-correções do volante: motorista alerta faz correções suaves. 
        # Motorista cansado cochila, sai da faixa e dá uma guinada brusca (jerk).
        steering_base = 12 - (progression * 8) if progression < 0.7 else 2 + (progression * 15)
        steering_jerks = max(0, int(np.random.normal(steering_base, 2)))
        
        # 4. Tempo total de estrada (em horas)
        hours_driven = round((i * 1) / 60, 2) # 1 registro por minuto
        
        # Regra de Especialista para Rotular o Target (Fadiga Detectada: 0 ou 1)
        # Se PERCLOS > 12% ou Bocejos acumulados com horas altas
        if perclos > 0.13 or (yawns >= 2 and perclos > 0.10) or (steering_jerks > 14 and progression > 0.8):
            fatigue_target = 1
        else:
            fatigue_target = 0
            
        data.append({
            "timestamp": timestamps[i].strftime("%Y-%m-%d %H:%M:%S"),
            "perclos": round(perclos, 4),
            "yawn_count": yawns,
            "steering_jerks": steering_jerks,
            "hours_driven": hours_driven,
            "driver_fatigue": fatigue_target # Nosso Target para o Machine Learning
        })
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Gerando dataset sintético de telemetria veicular...")
    df_simulated = simulate_driver_stream(records=2000)
    
    # Criando diretório de output caso não exista
    output_dir = os.path.join("data-science-core", "notebooks")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "telemetry_fatigue_dataset.csv")
    df_simulated.to_csv(output_path, index=False)
    
    print(f"Dataset gerado com sucesso em: {output_path}")
    print(df_simulated.head(10))
    print(f"\nDistribuição das classes (0=Alerta, 1=Fadiga):\n{df_simulated['driver_fatigue'].value_counts(normalize=True)}")