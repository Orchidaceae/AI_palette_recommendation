import csv
import numpy as np

# takes a palett and returns a list of integers
def palette_2_int_list(list):
    c1 = list[1]
    c2 = list[2]
    c3 = list[3]
    #print([int(list[0]), int(c1[1:], 16), int(c2[1:], 16), int(c3[1:], 16)])
    # remove the first char '#' from the color strings and convert them to integers
    return [int(list[0]), int(c1[1:], 16), int(c2[1:], 16), int(c3[1:], 16)]

"import data from named csv-file as numpy vectors"
def import_csvdata(filename):
    path = "Data_mining/" + filename + ".csv"
    with open(path, "rt") as f:
        reader = csv.reader(f, delimiter=",")
        r = []
        c1 = []
        c2 = []
        c3 = []
        for row in reader:
            ip = palette_2_int_list(row) #returns a rated palette as an list of integers
            # list each element separetly 
            r.append(ip[0])
            c1.append(ip[1])
            c2.append(ip[2])
            c3.append(ip[3])
    # make a 2d numpy array with one attribute per row
    return np.array([r,c1,c2,c3])

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
data = import_csvdata("lovisa")
print(np.shape(data))
# data2 = np.transpose(data)
# print(data)
# print(data2)
# #s = slice()#start:stop:step
# print("Palette ratings - Row 0: ",data[0])
# print("One palette - Column 0: ",data[:,0])