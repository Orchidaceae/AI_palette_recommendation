import matplotlib.pyplot as plt

fig1 = plt.figure()

colors = ["#20cccf", "#2040cf", "#ac1b66"]

# Generate subplots for each of the colors
i = 1
for c in colors:
    subplot_pos = 130 + i #calculate position index of subplot
    i = i + 1
    fig1.add_subplot(subplot_pos, facecolor=c, aspect='equal')
    frame1 = plt.gca() #get current matplotlib axes object
    #hide x and y axis on subplot
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

#adjust the spacing between subplots to zero
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
#show plot
plt.show()
