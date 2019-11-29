import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

def new_palette():
    palette = []
    # for i in range (0 to 2):
    #     i = i + 1# random hex number between 000000 to ffffff
    # return palette

def show_palette(fig, colors):
    ax_list = fig.get_axes()
   
    # Generate subplots for each of the colors
    i = 1
    for c in colors:
        if not ax_list: # empty list generate axes objects
            subplot_pos = 130 + i #calculate position index of subplot
            i = i + 1
            fig1.add_subplot(subplot_pos, facecolor=c, aspect='equal') # create an axes object as subplot
            frame1 = plt.gca() #get current matplotlib axes object
            #hide x and y axis in subplot
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)
        else:
            # update the color of existing axes objects
            i = 0
            for ax in ax_list:
                ax.set_facecolor(colors[i])
                i = i + 1                
    #adjust the spacing between subplots to zero
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
    fig1.canvas.draw()

# collect input from textbox
def submit(input=None):
    # collect input and store value in csv
    # call random function to generate new palette
    # update plot with new palette colors

    plt.draw()

fig1 = plt.figure()

colors = ["#20cccf", "#2040cf", "#ac1b66"]
colors2 = ["#ff0000", "#00ff00", "#0000ff"]
show_palette(fig1, colors)
# show figure
fig1.canvas.draw()
show_palette(fig1, colors2)
# redraw figure
fig1.canvas.draw()
ax_list = fig1.get_axes()
print(ax_list)

axbox = plt.axes([0.125, 0.05, 0.777, 0.075])
text_box = TextBox(axbox, "Rate 1 to 5", initial=None)
text_box.on_submit(submit)
plt.show()