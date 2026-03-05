import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree


data = pd.read_csv("D:\Data_Mining_Project\diabetes.csv")

#Preview of data
print("Dataset Preview:\n", data.head())

#Separation of characteristics (X) and goals (y)
X = data.drop(columns=['Outcome']) #Χαρακτηριστικα (εκτος απο την στηλη του "στοχου")
y = data['Outcome'] #Μεταβλητη στοχου

#Dividing the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Initializing and training the Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

#Predictions in the test set
y_pred = clf.predict(X_test)

#Model performance evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

#Decision tree representation
plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], fontsize=10)
plt.title("Decision Tree for Diabetes Prediction")
plt.show()