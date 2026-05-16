import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Configuração da página do Streamlit
st.set_page_config(
    page_title="Argus - Central de Monitoramento",
    page_icon="👁️",
    layout="wide"
)

st.title("Argus - Sistema Preditivo de Fadiga Humana")
st.subheader("Monitoramento de Telemetria e Biometria de Cabine em Tempo Real")

# URL da nossa API FastAPI que já está validada e rodando
API_URL = "http://127.0.0.1:8000/predict"

# 2. Inicializar o histórico de dados na sessão do Streamlit para alimentar os gráficos
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Timestamp", "PERCLOS", "Bocejos", "Micro_Correcoes", "Horas_Estrada", "Probabilidade_Risco"
    ])

# 3. Layout de Colunas Superiores (Controles do Simulador)
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("### Controles do Simulador")
    run_simulation = st.toggle("Ativar Fluxo de Sensores do Veículo", value=True)
    speed = st.slider("Intervalo de Atualização (segundos)", 0.5, 3.0, 1.0)

# Inicializar o contador de passos para simular a linha do tempo do motorista
if "step" not in st.session_state:
    st.session_state.step = 0

# Container dinâmico que será limpo e atualizado a cada ciclo do loop
placeholder = st.empty()

# 4. Loop de Execução em Tempo Real
while run_simulation:
    st.session_state.step += 1
    step = st.session_state.step
    
    # Simulação Matemática: O motorista começa alerta e vai cansando progressivamente
    base_noise = np.random.normal(0, 0.01)
    if step < 15:
        # Estado Inicial: Motorista Alerta e Descansado
        perclos = max(0.01, 0.04 + base_noise)
        yawns = int(np.random.poisson(0.1))
        jerks = int(np.random.normal(10, 1.5))
        hours = round(step * 0.1, 2)
    else:
        # Estado Avançado: Fadiga e Perda de Reflexo Progressiva
        perclos = min(1.0, 0.05 + (step * 0.008) + np.random.normal(0, 0.02))
        yawns = int(np.random.poisson(0.15 * (step - 12)))
        jerks = int(np.random.normal(11 + (step * 0.3), 2.5))
        hours = round(1.5 + (step * 0.1), 2)

    # 5. Estruturar o payload JSON para disparar contra a API FastAPI
    payload = {
        "perclos": float(round(perclos, 4)),
        "yawn_count": int(yawns),
        "steering_jerks": int(jerks),
        "hours_driven": float(hours)
    }

    # Consumir o endpoint /predict da API de forma assíncrona
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            risk_prob = result["risk_probability_percentage"]
            fatigue_detected = result["fatigue_detected"]
            status_msg = result["status_message"]
        else:
            risk_prob, fatigue_detected, status_msg = 0.0, False, "Erro de resposta da API"
    except Exception:
        risk_prob, fatigue_detected, status_msg = 0.0, False, "Servidor API offline"

    # Salvar a leitura atual no histórico da sessão (mantendo os últimos 30 registros)
    new_row = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "PERCLOS": perclos,
        "Bocejos": yawns,
        "Micro_Correcoes": jerks,
        "Horas_Estrada": hours,
        "Probabilidade_Risco": risk_prob
    }])
    st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True).tail(30)
    df_hist = st.session_state.history

    # 6. Renderização Dinâmica dos Componentes Visuais
    with placeholder.container():
        # Grid de Cards com as Métricas Atuais dos Sensores
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PERCLOS (Olhos Fechados)", f"{perclos*100:.1f}%")
        m2.metric("Bocejos / Minuto", yawns)
        m3.metric("Micro-correções (Volante)", jerks)
        m4.metric("Tempo de Direção", f"{hours}h")

        st.markdown("---")

        # Alertas Visuais Baseados na Decisão do Modelo Random Forest
        if fatigue_detected:
            st.error(f"ALERTA CRÍTICO DA IA: {status_msg} ({risk_prob}% de certeza)")
        elif risk_prob > 40:
            st.warning(f"Atenção Operacional: Sinais iniciais de sonolência detectados ({risk_prob}%)")
        else:
            st.success(f"Monitoramento Seguro: {status_msg}")

        # Exibição dos Gráficos de Tendência Temporal
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### Evolução do Índice de Risco (%)")
            st.line_chart(df_hist.set_index("Timestamp")["Probabilidade_Risco"], color="#ff4b4b" if fatigue_detected else "#29b5e8")
        with g2:
            st.markdown("#### Comportamento Histórico do Sensor PERCLOS")
            st.line_chart(df_hist.set_index("Timestamp")["PERCLOS"], color="#ffaa00")

    time.sleep(speed)