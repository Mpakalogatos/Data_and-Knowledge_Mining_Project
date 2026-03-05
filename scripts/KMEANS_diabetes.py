import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("D:\Data_Mining_Projects\diabetes_reduced.csv")

#View basic data
print(df.info())
print(df.describe())

#Separation of characteristics (X) and goals (y)
X = df.drop(columns=['Outcome'])  #characteristics
y = df['Outcome']  #goals

#Apply normalization for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Dividing the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#Implementation of Kmeans algorithm
kmeans = KMeans(n_clusters=2, random_state=42)  #2 clusters will be used, 1) For non-diabetics 2) For diabetics
kmeans.fit(X_train)

#Predicting clusters for the test data
y_pred = kmeans.predict(X_test)

#Map clusters to binary result (0 and 1)
cluster_map = {}
for cluster in np.unique(kmeans.labels_):
    labels_in_cluster = y_train[kmeans.labels_ == cluster]
    majority_label = labels_in_cluster.mode()[0]
    cluster_map[cluster] = majority_label

#Convert cluster predictions to target predictions
y_pred_mapped = [cluster_map[cluster] for cluster in y_pred]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_mapped))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_mapped))

accuracy = accuracy_score(y_test, y_pred_mapped)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
