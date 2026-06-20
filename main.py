from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

br = load_breast_cancer()
a = br.data
b = br.target

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(a_train, b_train)
print(round(accuracy_score(b_test, model.predict(a_test))*100), "%")

joblib.dump(model, "load_breast_cancer.pkl")

model2 = joblib.load("load_breast_cancer.pkl")

new = a_test[0]
print(model2.predict([new]))
print(b_test[0])