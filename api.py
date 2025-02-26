from fastapi import FastAPI
import joblib
import random
from fastapi.middleware.cors import CORSMiddleware

# Adicionar suporte a CORS (caso esteja bloqueando no Flutter)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Carregar o modelo com suporte a UTF-8
model = joblib.load("modelo_astrologico_com_multiplas_previsoes.pkl")

# Função para previsão
def prever_aleatorio(signo, area, emocional):
    entrada = f"{signo} {area} {emocional}"

    previsoes_str = model.predict([entrada])[0]

    # 🔹 Garantir que a string de saída esteja no formato correto
    previsoes_possiveis = previsoes_str.split(" | ")
    previsao_aleatoria = random.choice(previsoes_possiveis)

    return previsao_aleatoria

# Rota de previsão
@app.get("/prever/")
def fazer_previsao(signo: str, area: str, emocional: str):
    previsao = prever_aleatorio(signo, area, emocional)
    return {"signo": signo, "area": area, "emocional": emocional, "previsao": previsao}

# Rota raiz
@app.get("/")
def root():
    return {"mensagem": "API de Previsões Astrológicas funcionando!"}
