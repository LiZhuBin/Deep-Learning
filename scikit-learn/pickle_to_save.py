from sklearn.svm import SVC
from sklearn.datasets import load_iris
import pickle

clf = SVC()
iris = load_iris()
X = iris.data
y = iris.target
clf.fit(X, y)

with open("./save/pick_to_save_model","wb") as f:
    pickle.dump(clf,f)

with open("./save/pick_to_save_model","rb") as f:
    clf2 = pickle.load(f)
    print(clf2.predict(X[0:1]))