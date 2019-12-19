import csv
import numpy as np

# Normalize input for the NN by converting the hex color codes into 3 RGB triplets.
# Expand the NN input layer to 9 inputs.
# Each of the new 9 network inputs receives one color channel from each of the RGB triplets.

def hex_to_rgb(hex):
    return [int(hex[i:i+2], 16) / 255.0 for i in (1, 3, 5)]

# takes a palette and returns a list of rgb components as normalized floats
def rgb_palette_from(list):
    c1 = hex_to_rgb(list[1])
    c2 = hex_to_rgb(list[2])
    c3 = hex_to_rgb(list[3])
    label = int(list[0]) - 1  # -1 from labels -> range [0 1 2]
    return [label] + c1 + c2 + c3

# takes a palette and returns a list of integers
def palette_to_int_list(list):
    c1 = list[1]
    c2 = list[2]
    c3 = list[3]
    # remove the first char '#' from the color strings and convert them to integers
    return [int(list[0]), int(c1[1:], 16), int(c2[1:], 16), int(c3[1:], 16)]

"import data from named csv-file as numpy vectors"
def import_csvdata(filename):
    path = "Data_mining/" + filename + ".csv"
    with open(path, "rt") as f:
        reader = csv.reader(f, delimiter=",")
        array = []
        for row in reader:
            if not row:
                continue
            else:
                #ip = palette_to_int_list(row) #returns a rated palette as an list of integers
                ip = rgb_palette_from(row) # return a 10 element vector with ratings and normalized rgb values of palette
                array.append(ip)
    # return a 2d numpy array with one 10 element vector per row
    return np.array(array)


#TODO: save return array to a file
""">>> from tempfile import TemporaryFile
>>> outfile = TemporaryFile()
>>>
>>> x = np.arange(10)
>>> np.save(outfile, x)
>>>
>>> _ = outfile.seek(0) # Only needed here to simulate closing & reopening file
>>> np.load(outfile)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"""
    

# How to use import_csvdata()
# data = import_csvdata("test")
# print(np.shape(data))
# print(data)
# print("Palette ratings - Column 0: ",data[:,0])
# print("Palette c1 (rgb) - Column 1 to 3: ",data[:,1:4])
# print("One palette - Row 0: ",data[0])
# print("Samples only: ",np.delete(data, 0, axis=1))

"""
Data format of the numpy array:
labels:     samples:       
[[0.         0.70196078 0.94117647 0.77254902 0.1254902  0.89019608 0.09019608 0.15686275 0.87058824 0.0627451 ]
 [0.         0.74901961 0.59215686 0.58039216 0.05882353 0.88235294 0.29019608 0.71372549 0.50196078 0.87843137]
 [1.         0.39215686 0.30196078 0.69411765 0.30196078 0.87058824 0.67843137 0.68627451 0.89803922 0.85490196]
 [0.         0.56078431 0.4        0.8        0.94509804 0.76862745 0.56470588 0.88627451 0.56862745 0.14509804]
 [2.         0.05882353 0.99215686 0.14901961 0.4745098  0.2        0.19215686 0.69019608 0.78039216 0.55294118]]
"""