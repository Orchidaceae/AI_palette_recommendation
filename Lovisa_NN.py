from csv_reader import import_csvdata #used to read data from csv files as numpy arrays
import keras.layers as layers
import keras.optimizers as optimizers
from keras import Sequential
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import math
import csv

# add data to csv file
def report(val_loss, val_accu, comment, file):
    myData = [str(val_loss), str(val_accu), comment]

    path = "./Data_mining/" + file + ".csv"
    # open file in append mode
    myFile = open(path, 'a')
    writer = csv.writer(myFile)
    writer.writerow(myData)
    myFile.flush()
    myFile.close()
           
    print("Written \n")

def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # calculate number of epochs from history data
    epochs = range(1, len(loss) + 1)

    # create figure and add subplots
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    # plot training vs validation loss
    ax.plot(epochs, loss, 'bo', label='Training loss')
    ax.plot(epochs, val_loss, 'b', label='Validation loss')
    ax.title.set_text('Training and validation loss')
    ax.set(xlabel='Epochs', ylabel='Loss')
    ax.legend(['Train', 'Test'], loc='upper right')

    # plot training vs validation accuracy
    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.title.set_text('Model accuracy')
    ax2.set(xlabel='Epochs', ylabel='Accuracy')
    ax2.legend(['Train', 'Test'], loc='upper right')

    plt.subplots_adjust(hspace = 0.6) # Add space between subplots
    plt.show()

""" generate a prediction from a data sample using the model"""
def predictWithModel(model, samples):
    return model.predict(samples) #predict() always takes a list of values

""" train model with samples and labels for a number if epochs
    - epochs: defines how many iterations of training that we wish to do.
    - batchSize: defines how many samples we should batch into one calculation of the gradient."""
def trainModel(model, samples, labels, epochs, batch_size, x_val, y_val):
    history = model.fit(samples, labels,
              epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    plot_training_history(history)
    


"""generate sequential model object"""
def createNewSeqModel():
    model = Sequential()

    # inizilise weights to random uniform numbers
    weights = 'random_uniform'

    # input layer = 1, first layer 50
    model.add(layers.Dense(45, input_shape=(9,),
                           activation='relu', kernel_initializer=weights))
    model.add(layers.Dropout(0.5))
    # second layer 50
    model.add(layers.Dense(45, activation='relu', kernel_initializer=weights))
    model.add(layers.Dropout(0.5))
    # output layer 1, the number of classifications
    model.add(layers.Dense(3, activation='softmax',
                           kernel_initializer=weights))

    # model training specification: adam -> "Stochastic Gradiant Decent (SGD) on steriods"
    # loss function: Mean Square Error (MSE), the network does not optimize accuracy 
    # instead it tries to minimaze the loss of accuracy with a loss function
    model.compile(optimizers.adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# set up data
data = import_csvdata("lovisa") # returns tuple of numpy arrays

# shuffle the data values
np.random.shuffle(data)

# label data
labels = data[:,0]# collect label array
size = labels.size

# categorical encoding
labels = to_categorical(labels) # hot 1 encoding

# Split data into training, testing and validation
# Data: |             ~80% training             |  ~10% test   |  ~10% validation   |
training_index = math.floor((size/10)*8) # ~80 percent of data
training_labels = labels[:training_index] # every element up to index

test_index = (math.floor((size - training_index)/2)) + training_index

test_labels = labels[training_index:test_index] # elements between indecies, excluding value at end index
validation_labels = labels[test_index:] # every element after index, including start index

# plt.plot(training_labels)
# plt.show()

# collect samples
samples = np.delete(data, 0, axis=1)

# Split data into training, testing and validation
training_samples = samples[:training_index:] # slice tensor up to row of index
test_samples = samples[training_index:test_index:] # slice tensor between indecies
validation_samples = samples[test_index::] # slice tensor after index

# create new network
Net = createNewSeqModel()

# Fit the model to our training data
trainModel(Net, training_samples, training_labels, 150, 32, test_samples, test_labels)  #(model, samples, labels, epochs, batchsize, x_val, y_val):

# # Test model on evaluation data
# Predictions = predictWithModel(Net, test_samples)
# print ( "Predictions: \n " , Predictions)
# print ( "Labels: \n " , test_labels)

# calculate validation loss and validation accuracy
val_loss, val_acc = Net.evaluate(validation_samples, validation_labels)
print ( "Validation loss: \n " , val_loss)
print ( "Validation accuracy: \n " , val_acc)
