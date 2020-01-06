import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from os import listdir
from os.path import isfile, join
import random
import csv
import numpy as np
import math

# global variable
current_palette = []
palette_session_counter = 0
session_rateing_sum = 0
moving_avg_sum = 0
average_session_rating = []
session_vector = []

def hex_to_rgb(hex):
    return [int(hex[i:i+2], 16) / 255.0 for i in (1, 3, 5)]

# takes a palette and returns a list of rgb components as normalized floats
def rgb_palette_from(list):
    c1 = hex_to_rgb(list[0])
    c2 = hex_to_rgb(list[1])
    c3 = hex_to_rgb(list[2])
    return [] + c1 + c2 + c3

# predicts the ratings of a list of palettes and returns a list of integer [1 2 3]
def predict_ratings(palettes):
    global myModel 

    # preprocess palette data to 9d vector of normalized rgb values
    data = []
    for p in palettes:
        tmp = rgb_palette_from(p)
        data.append(tmp)
    data = np.array(data) # transform data into numpy array
    pred = myModel.predict(data) #predict() always takes a list of values

    predictions = []
    # translate predictions to integers
    for n in pred:
        max = n[0]
        r = 1
        if n[1] > max:
            max = n[1]
            r = 2
        if n[2] > max:
            max = n[2]
            r = 3
        predictions.append(r)
    return predictions


# generate a list of 3 6-digit color hexcodes
def new_palette():
    palette = []
    # randomly generate 3 colors
    for _ in range (0,3):
        hexcode = ""
        # generate 2 digits at a time
        for _ in range (0,3):
            r_n = random.randint(0,255)
            if r_n <= 15:
                byte_str = "0f"
            else:
                byte_str = format(r_n, 'x')
            hexcode = hexcode + byte_str
        palette.append("#" + hexcode)
    return palette

def get_recommendation():
    max_iterations = 10
    t = 0
    # try a limited number of times to find a palette with rating 3
    while t < max_iterations:
        # generate 10 random palettes
        palettes = []
        for _ in range (0,10):
            palettes.append(new_palette())
        # predict their ratings
        ratings = predict_ratings(palettes) #get a list of integer ratings [1 2 3]
        print("Recommendation ratings: ", ratings)
        # return a palette with high rating (2 or 3)
        for i in range (0,10):
            if ratings[i] == 3:
                return palettes[i]
        t = t + 1
    print("No 3 rated palette found")
    for i in range (0,10):
        if ratings[i] == 2:
            return palettes[i]
    print("No recommendation found")
    return palettes[0]

# updates the displayed palette, create new palette if there is not already one 
def show_palette(fig, colors):
    ax_list = fig.get_axes()
   
    # Generate subplots for each of the colors
    i = 1
    for c in colors:

        if not ax_list: # if list is empty, generate axes objects
            subplot_pos = 130 + i #calculate position index of subplot
            i = i + 1
            fig.add_subplot(subplot_pos, facecolor=c, aspect='equal') # create an axes object as subplot
            frame1 = plt.gca() #get current matplotlib axes object
            #hide x and y axis in subplot
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)
            #adjust the spacing between subplots to zero
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
        else:
            # update the color of existing axes objects
            for i in range (0,3):
                ax = ax_list[i]
                ax.set_facecolor(colors[i]) 
    # update the plot
    fig.canvas.draw() 

# add data to csv file
def update_csv(rate, palette):
    global csv_file
    myData = [str(rate), palette[0], palette[1], palette[2]]
    path = './data/' + csv_file
    # open file in append mode
    myFile = open(path, 'a')
    writer = csv.writer(myFile)
    writer.writerow(myData)
    myFile.flush()
    myFile.close()
           
    print("Written \n")

# get users as a list
def get_users():
    user_list = []
    path = "./data/users.csv"
    with open(path, "rt") as user_f:
        reader = csv.reader(user_f, delimiter=";")
        for row in reader:
            if not row:
                continue
            else:
                user_list.append(row[0])
    user_f.close()
    return user_list

# get average rating from training data and previous recommendation data
def collect_statistics():
    global csv_file
    recom_path = './data/' + csv_file
    training_data_path = '../Data_mining/' + csv_file
    with open(recom_path, "rt") as f:
        reader = csv.reader(f, delimiter=",")
        sum = 0
        n = 0
        for row in reader:
            if not row:
                continue
            else:
                rate = int(row[0])
                sum = sum + rate
                n = n + 1
    if n != 0:
        recom_average_rating = sum/n
    else:
        print("no previous sessions found")
        recom_average_rating = 0
    
    training_average_rating = 0    
    try:
        with open(training_data_path, "rt") as f:
            reader = csv.reader(f, delimiter=",")
            sum = 0
            n = 0
            for row in reader:
                if not row:
                    continue
                else:
                    rate = int(row[0])
                    sum = sum + rate
                    n = n + 1
        if n != 0:
            training_average_rating = sum/n
    except IOError:
        print("Training data not found")      
    return (recom_average_rating, training_average_rating)

def display_statistics():
    global recom_average_prev
    global train_avarage
    global session_rateing_sum
    global palette_session_counter

    session_average = session_rateing_sum/palette_session_counter
    print("Session: ", palette_session_counter)
    print("Session average rating: ", session_average)
    print("Average rating from previous sessions: ", recom_average_prev)
    print("Average rating from random training: ", train_avarage)
    print("\n")
    print("current palette:")
    print(current_palette)

# train model using the n latest data rows
def train_model(n):
    global myModel
    # collect data from the latest inputs
    global csv_file
    recom_path = './data/' + csv_file
    with open(recom_path, "rt") as f:
        reader = csv.reader(f, delimiter=",")
        data = []
        for row in reader:
                if not row:
                    continue
                else:
                    data.append(row)

        #data = list(reader)
        row_count = len(data)
        cut_index = row_count-n

        array = []
        # every row of data after index, including start index
        for i in range(0,row_count):
            if i >= cut_index:
                array.append(data[i])

    a = []
    # preprocess data
    for v in array:
        # build a 10 element vector with ratings and normalized rgb values of palette
        c1 = hex_to_rgb(v[1])
        c2 = hex_to_rgb(v[2])
        c3 = hex_to_rgb(v[3])
        label = int(v[0]) - 1  # -1 from labels -> range [0 1 2]
        row = [label] + c1 + c2 + c3
        a.append(row) # build a 2d numpy array with one 10 element vector per row
    
    data = np.array(a) 

    # shuffle the data values
    np.random.shuffle(data)

    # label data
    labels = data[:,0]# collect label array
    size = labels.size

    # categorical encoding
    labels = to_categorical(labels, num_classes=3) # hot 1 encoding

    # Split data into training, testing and validation
    # Data: |             ~80% training             |  ~10% test   |  ~10% validation   |
    training_index = math.floor((size/10)*8) # ~80 percent of data
    training_labels = labels[:training_index] # every element up to index

    test_index = (math.floor((size - training_index)/2)) + training_index

    test_labels = labels[training_index:test_index] # elements between indecies, excluding value at end index
    validation_labels = labels[test_index:] # every element after index, including start index

    # collect samples
    samples = np.delete(data, 0, axis=1)

    # Split data into training, testing and validation
    training_samples = samples[:training_index:] # slice tensor up to row of index
    test_samples = samples[training_index:test_index:] # slice tensor between indecies
    validation_samples = samples[test_index::] # slice tensor after index

    print(training_samples)
    print(training_labels)

    # train network

    # Fit the model to our training data and validate with test data
    print(test_labels)
    history = myModel.fit(training_samples, training_labels, epochs=150, batch_size=32, validation_data=(test_samples, test_labels))

    #TODO: plot training history?
    # # show plot with training and testing statistics
    # plot_training_history(history)

    # calculate validation loss and validation accuracy on validation data
    val_loss, val_acc = myModel.evaluate(validation_samples, validation_labels)
    print ( "Validation loss: \n " , val_loss)
    print ( "Validation accuracy: \n " , val_acc)

    #TODO: save changes to model

# collect input from textbox
def submit(input):
    global current_palette
    global text_box
    global palette_session_counter
    global session_rateing_sum
    global moving_avg_sum
    global average_session_rating
    global session_vector

    # check input
    try:
        input = int(input)
    except:
        print("invalid input")
    else:
        if 1 <= input <= 3:
            print("input:", input)
            # save values to csv
            update_csv(input, current_palette)

            # get new recommendation palette
            palette = get_recommendation()

            # update plot with new palette colors
            fig = plt.figure("Palette")
            show_palette(fig, palette)

            # set global var
            session_rateing_sum = session_rateing_sum + input
            moving_avg_sum = moving_avg_sum + input
            palette_session_counter = palette_session_counter + 1
            current_palette = palette

            display_statistics()

            n = 10
            if ((palette_session_counter % n) == 0): # train model every n:th input
                train_model(n)
                average_session_rating.append(moving_avg_sum / 10)
                session_vector.append(palette_session_counter)
                moving_avg_sum = 0

                #TODO: plot training curve with avg session data
                fig = plt.figure("Stats")
                ax = fig.add_subplot(1, 1, 1)
                ax.grid(True)
                ax.set_xlabel('Palette index')
                ax.set_ylabel('Avg. rating')
                ax.set_ylim([1.0, 3.0])
                ax.set_title("Avg. rating over 10 recent palettes")
                ax.plot(session_vector, average_session_rating, '-bo')
                expected_value_random = 1.98 # E(Uniform([1, 2, 3]))
                ax.plot([0, palette_session_counter], [expected_value_random, expected_value_random], '--r')

            # update plot
            plt.draw()
        else:
            print("pick a number between 1 and 3")

# set up user-profiles
user_list = get_users()
print("Current user profiles:", user_list)
user_id = input("Please enter user_id: ")

# choose among exsisting user profiles
while (user_id in user_list) != True:
    user_id = input("Please enter user_id: ")

# set file to write collected data to
csv_file = str(user_id) + ".csv"

tmp = collect_statistics()
recom_average_prev = tmp[0]
train_avarage = tmp[1]

def get_models(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files

model_list = get_models("../Models")

print("Current models: ", model_list)
model_id = input("Please enter model name: ")

# choose among exsisting models
while (model_id in model_list) != True:
    model_id = input("Please enter model name: ")

# set model for recommendations
myModel = load_model('../Models/' + model_id)

# create plot figure windows
fig = plt.figure("Stats")
fig.canvas.draw()
fig = plt.figure("Palette")

# generate first palette and show
current_palette = get_recommendation()
show_palette(fig, current_palette)
print("current palette:")
print(current_palette)

# redraw figure
fig.canvas.draw()

# create new axes object
axbox = plt.axes([0.125, 0.05, 0.777, 0.075])
# add text box to axbox
text_box = TextBox(axbox, "Rate 1 to 3", initial="2")
# call function submit after user press enter i textbox
text_box.on_submit(submit)
# open plot in window
plt.show()
