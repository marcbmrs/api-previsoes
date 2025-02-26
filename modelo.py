import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import random

# Carregar o banco de dados CSV
df = pd.read_csv("previsoes_astrologicas.csv", encoding="utf-8")

# Criar uma coluna combinada (Signo + Área + Emocional) para entrada do modelo
df["Entrada"] = df["Signo"] + " " + df["Área"] + " " + df["Emocional"]

# Agrupar as previsões por combinação de Entrada
previsoes_por_entrada = df.groupby("Entrada")["Previsão"].apply(list).to_dict()

# Separar as entradas e previsões
entradas = list(previsoes_por_entrada.keys())  # Combinações únicas de entrada
previsoes = list(previsoes_por_entrada.values())  # Listas de previsões para cada combinação

# 🔹 Converter listas de previsões para strings separadas por "|"
previsoes = [" | ".join(previsao) for previsao in previsoes]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(entradas, previsoes, test_size=0.2, random_state=42)

# Criar o pipeline de processamento (TF-IDF + Random Forest)
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2, max_df=0.8),
    RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
)

# Treinar o modelo
model.fit(X_train, y_train)

# Salvar o modelo treinado
joblib.dump(model, "modelo_astrologico_com_multiplas_previsoes.pkl")

print("✅ Modelo treinado e salvo com sucesso!")
