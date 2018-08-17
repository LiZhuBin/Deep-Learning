from sklearn.learning_curve import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as  plt
import numpy as np


digits = load_digits()
X = digits.data
y = digits.target

train_size, train_loss, test_loss = learning_curve(
    SVC(),
    X,y,
    cv=10,
    scoring='neg_mean_squared_error',
    train_sizes=[0.1,0.25, 0.5, 0.75, 1]
)
train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)


plt.plot(train_size, train_loss_mean,'o-',color="r",label="Training")
plt.plot(train_size, test_loss_mean,'o-',color="g",label="Cross-validation")
plt.xlabel("Train examples")
plt.ylabel("loss")
plt.legend(loc="best")
plt.show()