import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import random
import csv

# global variable
current_palette = []

# generate a list of 3 6-digit color hexcodes
def new_palette():
    palette = []
    # randomly generate 3 colors
    for i in range (0,3):
        hexcode = ""
        # generate 2 digits at a time
        for j in range (0,3):
            r_n = random.randint(0,255)
            if r_n <= 15:
                byte_str = "0f"
            else:
                byte_str = format(r_n, 'x')
            hexcode = hexcode + byte_str
        palette.append("#" + hexcode)
    return palette

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

    # open file in append mode
    myFile = open(csv_file, 'a')
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



# collect input from textbox
def submit(input):
    global fig1
    global current_palette
    global text_box #TODO: empty value in textbox

    # check input
    try:
        input = int(input)
    except:
        print("invalid input")
    else:
        if 1 <= input <= 3:
            print("input:", input)
            # TODO: save values to csv
            update_csv(input, current_palette)

            # call random function to generate new palette
            palette = new_palette()

            # update plot with new palette colors
            show_palette(fig1, palette)
            # set global var 
            current_palette = palette
            print("current palette:")
            print(current_palette)
            # update plot
            plt.draw()
        else:
            print("pick a number between 1 and 3")


#TODO: set up user-profiles
user_list = get_users()
print("Current user profiles:", user_list)
user_id = input("Please enter user_id: ")

# choose among exsisting user profiles
while (user_id in user_list) != True:
    user_id = input("Please enter user_id: ")

# set file to write collected data to
csv_file = str(user_id) + ".csv"

# create plot figure
fig1 = plt.figure()

# generate first palette and show
current_palette = new_palette()
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
