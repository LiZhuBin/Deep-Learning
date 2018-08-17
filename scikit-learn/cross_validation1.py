from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)
#得到的得分是将总数据分成5份进行取得分，最后用mean得平均分
scroes = cross_val_score(knn,X,y,cv=5, scoring='accuracy')
print(scroes.mean())
