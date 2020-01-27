from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

#Coleta dados
csv = pd.read_csv('dados.csv', sep=',')
csv = csv.drop(columns=['usuario'])
dados = csv.values

#Cria o modelo
modelo = KMeans(5, random_state=0)
resultado = modelo.fit_predict(dados)

#Organiza os grupos
grupos = [
    [], #dados no grupo 0
    [], #dados no grupo 1
    [], #dados no grupo 2
    [], #dados no grupo 3
    [], #dados no grupo 4
]
for i in range(len(resultado)):
    #Adiciona os dados do usuário no grupo que ele pertence
    grupos[resultado[i]].append(dados[i])

#Avalia resultados de cada grupo
for grupo in grupos:
    print('Usuários no grupo: ', len(grupo))
    print(np.sum(grupo, axis=0))
