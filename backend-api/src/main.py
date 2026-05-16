import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Definir a estrutura dos dados de entrada usando Pydantic (Validação Forte)
class TelemetryInput(BaseModel):
    perclos: float = Field(..., description="Porcentagem de tempo com olhos fechados", ge=0.0, le=1.0)
    yawn_count: int = Field(..., description="Quantidade de bocejos no último minuto", ge=0)
    steering_jerks: int = Field(..., description="Micro-correções bruscas no volante", ge=0)
    hours_driven: float = Field(..., description="Tempo total de direção em horas", ge=0.0)

    class Config:
        json_schema_extra = {
            "example": {
                "perclos": 0.16,
                "yawn_count": 2,
                "steering_jerks": 18,
                "hours_driven": 4.5
            }
        }

# 2. Inicializar o app FastAPI
app = FastAPI(
    title="API de Predição de Fadiga - Argus",
    description="API assíncrona de alta performance para detecção de fadiga em motoristas utilizando Machine Learning.",
    version="1.0.0"
)

# 3. Carregar o cérebro da IA na inicialização da API
MODEL_PATH = os.path.join("data-science-core", "models", "fatigue_rf_model.pkl")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Cérebro da IA (Random Forest) carregado com sucesso na memória da API!")
else:
    raise RuntimeError(f"Erro Crítico: O modelo não foi encontrado em {MODEL_PATH}. Treine o modelo primeiro!")

# 4. Endpoint de checagem de saúde (Healthcheck)
@app.get("/", tags=["Health"])
def health_check():
    return {"status": "healthy", "model_loaded": True, "api_version": "1.0.0"}

# 5. Endpoint de Predição (Onde a mágica acontece)
@app.post("/predict", tags=["Machine Learning"])
def predict_fatigue(data: TelemetryInput):
    try:
        # Converter os dados de entrada para o formato que o Scikit-Learn espera: um array 2D [[...]]
        input_data = np.array([[
            data.perclos,
            data.yawn_count,
            data.steering_jerks,
            data.hours_driven
        ]])
        
        # Fazer a predição (0 ou 1)
        prediction = int(model.predict(input_data)[0])
        
        # Calcular a probabilidade de risco para dar um insight mais rico no Dashboard
        probabilities = model.predict_proba(input_data)[0]
        risk_probability = round(float(probabilities[1]) * 100, 2)
        
        # Mapear o resultado para um status amigável
        status = "Fadiga Detectada! Risco Crítico." if prediction == 1 else "Motorista Alerta e Seguro."
        
        return {
            "fatigue_detected": bool(prediction),
            "risk_probability_percentage": risk_probability,
            "status_message": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento da predição: {str(e)}")