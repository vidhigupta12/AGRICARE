from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

df = pd.read_csv('fertilizer.csv')
dataset = df.values
X = dataset[:, 0:9]
y = dataset[:, 9]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model2 = Sequential()

# Hidden Layer 1
model2.add(Dense(2000, activation='relu',
           input_dim=9, kernel_regularizer=l2(0.01)))
model2.add(Dropout(0.3, noise_shape=None, seed=None))

# Hidden Layer 1
model2.add(Dense(1000, activation='relu',
           input_dim=18, kernel_regularizer=l2(0.01)))
model2.add(Dropout(0.3, noise_shape=None, seed=None))

# Hidden Layer 2
model2.add(Dense(500, activation='relu', kernel_regularizer=l2(0.01)))
model2.add(Dropout(0.3, noise_shape=None, seed=None))

model2.add(Dense(4, activation='softmax'))

model2.compile(loss='binary_crossentropy',
               optimizer='adam', metrics=['accuracy'])


earlystop = EarlyStopping(monitor='val_loss',  # value being monitored for improvement
                          min_delta=0,  # Abs value and is the min change required before we stop
                          patience=2,  # Number of epochs we wait before stopping
                          verbose=1,
                          restore_best_weights=True)  # keeps the best weigths once stopped

batch_size = 32
epochs = 33

history = model2.fit(X_train,
                     y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(X_test, y_test))

score = model2.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model2.summary()

# model = Sequential()
# model.add(Dense(18, input_dim=9, activation='relu'))
# # model.add(Dense(100, activation='relu'))
# # model.add(Dense(100, activation='relu'))
# model.add(Dense(1))
# # model.add(Dense(1), activation = 'sigmoid')

# model.compile(loss='binary_crossentropy',
#               optimizer='adam', metrics=['accuracy'])


# model.fit(X, Y, epochs=10, batch_size=10)
# _, accuracy = model.evaluate(X, Y)
# print('Accuracy: %.2f' % (accuracy*100))
# print(model.summary())
