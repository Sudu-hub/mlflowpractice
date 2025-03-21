import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = iris.data 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 15
n_estimators = 100

# to set experiment to track experiment
mlflow.set_experiment('iris-dt')

#run mlflow
with mlflow.start_run():
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt=dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #we use metrics
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    #mlflow.log_param('n_estimator',n_estimators)
    print('accuracy:',accuracy)