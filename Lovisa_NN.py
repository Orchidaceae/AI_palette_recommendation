from csv_reader import import_csvdata #used to read data from csv files as numpy arrays
import keras.layers as Layers
import keras.optimizers as Optimizers
from keras import Sequential
from keras.utils.np_utils import to_categorical
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

""" generate a prediction from a data sample using the model"""
def predictWithModel(model, samples):
    return model.predict(samples) #predict() always takes a list of values

""" train model with samples and labels for a number if epochs
    - epochs: defines how many iterations of training that we wish to do.
    - batchSize: defines how many samples we should batch into one calculation of the gradient."""
def trainModel(model, samples, labels, epochs, batchSize):
    model.fit(samples, labels,
              epochs=epochs, batch_size=batchSize)


"""generate sequential model object"""
def createNewSeqModel():
    model = Sequential()

    # inizilise weights to random uniform numbers
    weights = 'random_uniform'

    # input layer = 1, first layer 50
    model.add(Layers.Dense(45, input_shape=(9,),
                           activation='relu', kernel_initializer=weights))
    # second layer 50
    model.add(Layers.Dense(45, activation='relu', kernel_initializer=weights))
    # output layer 1, the number of classifications
    model.add(Layers.Dense(3, activation='softmax',
                           kernel_initializer=weights))

    # model training specification: adam -> "Stochastic Gradiant Decent (SGD) on steriods"
    # loss function: Mean Square Error (MSE), the network does not optimize accuracy 
    # instead it tries to minimaze the loss of accuracy with a loss function
    model.compile(Optimizers.adam(), loss='categorical_crossentropy', metrics=['accuracy'])

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

training_index = math.floor((size/10)*8) # ~80 percent of data
training_labels = labels[:training_index] # every element up to index
test_labels = labels[training_index:] # every element after index

# collect samples
samples = np.delete(data, 0, axis=1)
training_samples = samples[:training_index:] # slice tensor up to row of index
test_samples = samples[training_index::] # slice tensor after index

# create new network
Net = createNewSeqModel()

# Fit the model to our training data
trainModel(Net, training_samples, training_labels, 25, 100) #(model, samples, labels, epochs, batchsize)

# Test model on evaluation data
Predictions = predictWithModel(Net, test_samples)
print ( "Predictions: \n " , Predictions)
print ( "Labels: \n " , test_labels)

# calculate validation loss and validation accuracy
val_loss, val_acc = Net.evaluate(test_samples, test_labels)
print ( "Validation loss: \n " , val_loss)
print ( "Validation accuracy: \n " , val_acc)
