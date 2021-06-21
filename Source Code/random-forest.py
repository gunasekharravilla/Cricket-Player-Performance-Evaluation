import pandas as pd
dataset = pd.read_csv("cleaned-modified.csv", names=['Runs','Mins','BF','4s','6s','SR','Pos','Inns','ha','result'],dtype=float).values
X = dataset[:,0:9]
y = dataset[:,9]
y = y.astype('int')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))