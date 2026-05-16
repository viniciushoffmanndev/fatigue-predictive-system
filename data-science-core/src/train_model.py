import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_fatigue_model():
    # 1. Caminhos dos arquivos
    dataset_path = os.path.join("data-science-core", "notebooks", "telemetry_fatigue_dataset.csv")
    model_output_path = os.path.join("data-science-core", "models", "fatigue_rf_model.pkl")
    
    if not os.path.exists(dataset_path):
        print(f"Erro: O dataset não foi encontrado em {dataset_path}. Rode o generate_data.py primeiro!")
        return

    # 2. Carregar o dataset
    print("Carregando dados de telemetria...")
    df = pd.read_csv(dataset_path)

    # 3. Separar Features (X) e Target (y)
    # Não usamos o 'timestamp' porque o modelo precisa aprender o padrão dos sensores, independente da hora.
    features = ["perclos", "yawn_count", "steering_jerks", "hours_driven"]
    X = df[features]
    y = df["driver_fatigue"]

    # 4. Dividir em Treino (80%) e Teste (20%) para validação
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dados estruturados: {X_train.shape[0]} amostras para treino e {X_test.shape[0]} para teste.")

    # 5. Inicializar e Treinar o modelo Random Forest (Algoritmo de Ensemble)
    print("Treinando o modelo Random Forest Classifier (Aguarde)...")
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    # 6. Avaliar o desempenho da nossa IA
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nRELATÓRIO DE PERFORMANCE DO MODELO ---")
    print(f"Acurácia Geral: {accuracy * 100:.2f}%")
    print("\nMétricas Detalhadas por Classe (0=Alerta, 1=Fadiga):")
    print(classification_report(y_test, y_pred))

    # 7. Salvar o modelo treinado (O cérebro em formato binário .pkl)
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Cérebro da IA exportado e salvo com sucesso em: {model_output_path}\n")

if __name__ == "__main__":
    train_fatigue_model()