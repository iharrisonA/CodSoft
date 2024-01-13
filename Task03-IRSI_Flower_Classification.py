import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset from the specified file path
iris_data = pd.read_csv('E:\\codsoft\\Data Science\\DataSets\\IRIS.csv')

# Encoding the species names to numerical values
le = LabelEncoder()
iris_data['species'] = le.fit_transform(iris_data['species'])

# Splitting the dataset into features (X) and target variable (y)
X = iris_data.drop('species', axis=1)
y = iris_data['species']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = dt_classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy: ", accuracy)
print("Classification Report: ")
print(classification_rep)