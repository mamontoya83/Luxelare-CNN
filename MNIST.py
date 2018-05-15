from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
#import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)
from keras.utils import np_utils

# Se carga el data set MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#plt.imshow(X_train[0])
#plt.show()

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]).astype('float32')

print X_train.shape

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2, verbose=1)

posicion = 5

#prediccionesCategorias = model.predict(np.array([X_test[posicion]]))

prediccionesCategorias = model.predict(X_test)

predicciones = []
for prediccionCategorias in prediccionesCategorias:
     index_max = max(xrange(len(prediccionCategorias)), key=prediccionCategorias.__getitem__)

     predicciones.append(index_max)

#print "predicciones : ",np.array(predicciones).reshape(len(predicciones), 1)

print np.argmax(y_test[posicion])

real = []
for y in y_test:
    real.append(np.argmax(y))

#print "real : ",np.array(real).reshape(len(real), 1)

correctos = 0
for idx, value in enumerate(predicciones):
    if value == real[idx]:
        correctos = correctos +1

print correctos / float(len(predicciones)) * 100
