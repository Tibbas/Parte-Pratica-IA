import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = pd.read_csv(r"C:\Users\USUARIO\Downloads\Parte_pratica-Rafael_Resende-20241217T121431Z-001\Parte_pratica-Rafael_Resende\Electric_Vehicle_Population_Data.csv")


data_grouped = data.groupby('City').count().reset_index()
data_grouped.rename(columns={'VIN (1-10)': 'Electric_Vehicle_Count'}, inplace=True)


X = data_grouped[['Electric_Vehicle_Count']]

# Aplicando cotovelo
inertia = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo para Determinar o Número de Clusters')
plt.grid()
plt.show()

#olha eu não entendi muito bem esse metodo do cotuvelo mas acredito que o num. de clusters é 9
#oque eu entendi é no ponto de menor inercia que é aonde a curva estar menor é num. de clusters
