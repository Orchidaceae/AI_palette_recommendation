from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from matplotlib import pyplot
from csv_reader import import_csvdata #used to read data from csv files as numpy arrays
import numpy as np
import keras.layers as Layers
import keras.optimizers as Optimizers
from keras import Sequential
from keras.optimizers import SGD
import math

#import the csv file
data = import_csvdata("maryam")

# shuffle the data values
np.random.shuffle(data)

# label data
y = data[:,0]# collect label array
x = np.delete(data, 0, axis=1)

# categorical encoding
y = to_categorical(y) # hot 1 encoding

# Split data into training, testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# create the model
model = Sequential()
# inizilise weights to random uniform numbers
weights = 'random_uniform'
# add layers to the model
model.add(Dense(54, activation='relu', input_shape=(9,), kernel_initializer=weights))

model.add(Dense(54, activation='relu', kernel_initializer=weights))
model.add(Dropout(0.1))
model.add(Dense(54, activation='relu', kernel_initializer=weights))

model.add(Dense(3, activation='softmax', kernel_initializer=weights))

# compile the model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# Fit the model to our training data and validate with test data
#print(training_x)
#print(training_y)
# train model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size=32)
# evaluate the model
_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

# saving options for generated model
answer = input("Do you want to save the model y/n: ")
input_list = ["y","n"]
while (answer in input_list) != True:
    answer = input("Do you want to save the model y/n: ")

if(answer == "y"):
    name = input("Name model: ")
    name = 'Models/' + name + ".h5"
    model.save(name)
