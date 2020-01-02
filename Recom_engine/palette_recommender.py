import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from keras.models import load_model
import random
import csv
import numpy as np

# global variable
current_palette = []
palette_session_counter = 0
session_rateing_sum = 0

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
        print(ratings)
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
            fig1.add_subplot(subplot_pos, facecolor=c, aspect='equal') # create an axes object as subplot
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
    fig1.canvas.draw() 

# add data to csv file
def update_csv(rate, palette):
    global csv_file
    myData = [str(rate), palette[0], palette[1], palette[2]]
    path = '/Users/lovisa/Desktop/Ht19/AI_ID1214/Project/AI_palette_recommendation/Recom_engine/' + csv_file
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
    with open("users.csv", "rt") as user_f:
        reader = csv.reader(user_f, delimiter=";")
        for row in reader:
            user_list.append(row[0])
    user_f.close()
    return user_list

# get avrage rating from training data and previous recommendation data
def collect_statistics():
    global csv_file
    recom_path = '/Users/lovisa/Desktop/Ht19/AI_ID1214/Project/AI_palette_recommendation/Recom_engine/' + csv_file
    training_data_path = '/Users/lovisa/Desktop/Ht19/AI_ID1214/Project/AI_palette_recommendation/Data_mining/' + csv_file
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
    recom_avrage_rating = sum/n
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
    training_avrage_rating = sum/n
    return (recom_avrage_rating, training_avrage_rating)

def display_statistics():
    global recom_avrage_prev
    global train_avarage
    global session_rateing_sum
    global palette_session_counter

    session_avrage = session_rateing_sum/palette_session_counter
    print("session: ", palette_session_counter)
    print("session avrage rating: ", session_avrage)
    print("Avrage rating from previous sessions: ", recom_avrage_prev)
    print("Avrage rating from random training: ", train_avarage)
    print("\n")
    print("current palette:")
    print(current_palette)

# collect input from textbox
def submit(input):
    global fig1
    global current_palette
    global text_box
    global palette_session_counter
    global session_rateing_sum

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

            # call random function to generate new palette
            palette = new_palette()

            # update plot with new palette colors
            show_palette(fig1, palette)
            # set global var
            session_rateing_sum = session_rateing_sum + input
            palette_session_counter = palette_session_counter + 1
            current_palette = palette

            # display statistics
            display_statistics()

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
recom_avrage_prev = tmp[0]
train_avarage = tmp[1]

#TODO: load user defined model to use for predictions
myModel = load_model('../Models/net72acc.h5')

# create plot figure
fig1 = plt.figure()

# generate first palette and show
current_palette = get_recommendation()
show_palette(fig1, current_palette)
print("current palette:")
print(current_palette)

# redraw figure
fig1.canvas.draw()


# create new axes object
axbox = plt.axes([0.125, 0.05, 0.777, 0.075])
# add text box to axbox
text_box = TextBox(axbox, "Rate 1 to 3", initial="2")
# call function submit after user press enter i textbox
text_box.on_submit(submit)
# open plot in window
plt.show()
