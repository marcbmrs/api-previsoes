import joblib
import random

# Carregar o modelo treinado
model = joblib.load("modelo_astrologico_com_multiplas_previsoes.pkl")

# Função de previsão que escolhe aleatoriamente uma previsão entre as várias possíveis
def prever_aleatorio(signo, area, emocional):
    entrada = f"{signo} {area} {emocional}"
    
    # Fazer a previsão
    previsoes_str = model.predict([entrada])[0]  # Retorna uma string única

    # 🔹 Transformar a string de volta em uma lista de previsões
    previsoes_possiveis = previsoes_str.split(" | ")
    
    # Escolher uma previsão aleatória
    previsao_aleatoria = random.choice(previsoes_possiveis)
    
    return previsao_aleatoria

# Testando a função
signo_usuario = "Câncer"
area_usuario = "Amor"
emocional_usuario = "Triste"

previsao = prever_aleatorio(signo_usuario, area_usuario, emocional_usuario)
print(f"🔮 Previsão para {signo_usuario} ({area_usuario} - {emocional_usuario}): {previsao}")
