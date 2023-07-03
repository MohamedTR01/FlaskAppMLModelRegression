#Importez les bibliothèques nécessaires :
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Chargez les données :

# Supposez que vous avez un fichier CSV contenant vos données
data = pd.read_csv('Data_CarsX.csv')
# Divisez les données en variables indépendantes (X) et dépendante (y)
X = data[['Années', 'Kilométrage', 'Le nombre de chevaux']]  # Remplacez les noms de colonnes par les vôtres
y = data['Prix']

#    Divisez les données en ensembles d'entraînement et de test :

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créez une instance du modèle de régression linéaire multiple :

model = LinearRegression()

# Entraînez le modèle sur l'ensemble d'entraînement :
model.fit(X_train, y_train)

# Faites des prédictions sur l'ensemble de test :
y_pred = model.predict(X_test)

# Évaluez les performances du modèle en calculant l'erreur quadratique moyenne (RMSE) :

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

'''...........................................................................................................'''

# Visualisez les premières lignes du DataFrame pour vérifier que les données ont été chargées correctement :
print(data.head())

# Affichez les statistiques descriptives des données :
print(data.describe())

# Diagramme de dispersion (scatter plot) pour chaque paire de variables :
pd.plotting.scatter_matrix(data, figsize=(10, 10))
plt.show()

# Diagramme de corrélation sous forme de heatmap :
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation='vertical')
plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
plt.show()

# Matrice de corrélation avec heatmap:
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Pairplot pour afficher les diagrammes de dispersion entre toutes les paires de variables :
sns.pairplot(data)
plt.show()

sns.scatterplot(x='Années', y='Kilométrage', hue='Prix', style='Le nombre de chevaux' , data=data)
plt.xlabel('Années')
plt.ylabel('Kilométrage')
plt.show()