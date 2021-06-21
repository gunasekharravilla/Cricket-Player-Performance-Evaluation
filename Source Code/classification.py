import pandas as pd
dataset = pd.read_csv("cleaned-modified.csv", names=['Runs','Mins','BF','4s','6s','SR','Pos','Inns','ha','result']).values
X = dataset[:,0:9]
y = dataset[:,9]
#X = X.astype('int')
#y=y.astype('int')
#print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
import xgboost
svclassifier = xgboost.XGBClassifier()
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_pred,y_test)*100)
