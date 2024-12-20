import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#acho que voce vai ter que mudar o caminho pra textar seu codigo, tenho problema com isso toda hora.
data = pd.read_csv(r"C:\Users\USUARIO\Downloads\Parte_pratica-Rafael_Resende-20241217T121431Z-001\Parte_pratica-Rafael_Resende\Electric_Vehicle_Population_Data.csv")



data_filtered = data[['City', 'VIN (1-10)']].dropna()

#pegando os paremetros para fazer o clusterização.
#A ideia aq é mostrar o num. de carros eletricos p/ cidade
data_grouped = data_filtered.groupby('City').count().reset_index()
data_grouped.rename(columns={'VIN (1-10)': 'Electric_Vehicle_Count'}, inplace=True)

# Normalizando os dados 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_grouped['Count_Normalized'] = scaler.fit_transform(data_grouped[['Electric_Vehicle_Count']])


kmeans = KMeans(n_clusters=4, random_state=42)
data_grouped['Cluster'] = kmeans.fit_predict(data_grouped[['Count_Normalized']])


plt.figure(figsize=(10, 6))
for cluster in data_grouped['Cluster'].unique():
    cluster_data = data_grouped[data_grouped['Cluster'] == cluster]
    plt.scatter(cluster_data['City'], cluster_data['Electric_Vehicle_Count'], label=f'Cluster {cluster}')
plt.xticks(rotation=90)
plt.xlabel("Cidade")
plt.ylabel("Número de Veículos Elétricos")
plt.title("Clusterização de Cidades com Base no Número de Veículos Elétricos")
plt.legend()
plt.tight_layout()
plt.show()

# Salvar os resultados em um arquivo CSV
data_grouped.to_csv("Electric_Vehicle_Clusters.csv", index=False)

# Exibir a tabela com os clusters
print(data_grouped)
