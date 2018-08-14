from __future__ import print_function
from keras.datasets import fashion_mnist
from keras.layers import Conv2D,Flatten,Dropout,MaxPool2D,Dense
from keras.optimizers import Adadelta
from keras.activations import relu
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.utils import np_utils
from keras.models import  Sequential


#批处理个数
batch_size = 128
#数据处理轮次
epochs = 3
#卷积核大小
kernel_size = (3,3)
#池化层大小
pool_size = (2,2)
#图片大小
img_rows, img_cols = 28, 28
#卷积高度
nb_filters = 32
nb_classes = 10

(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()
if K.image_data_format() is "channel_first":
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_test = X_test.astype('float32')
X_train = X_train.astype('float32')
X_test /=255
X_train /=255

y_train = np_utils.to_categorical(y_train,nb_classes)
y_test = np_utils.to_categorical(y_test,nb_classes)

model = Sequential()
model.add(Conv2D(nb_filters,kernel_size,padding='same', input_shape=input_shape, activation='relu'))
model.add( Conv2D(nb_filters,kernel_size,activation='relu'))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes,activation='softmax'))

model.compile(optimizer='adadelta',loss=categorical_crossentropy,metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

