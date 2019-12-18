import csv
import numpy as np

# takes a palett and returns a list of integers
def palette_2_int_list(list):
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
        r = []
        c1 = []
        c2 = []
        c3 = []
        for row in reader:
            if not row:
                continue
            else:
                ip = palette_2_int_list(row) #returns a rated palette as an list of integers
                # list each element separetly 
                r.append(ip[0])
                c1.append(ip[1])
                c2.append(ip[2])
                c3.append(ip[3])
    # make a 2d numpy array with one attribute per row
    array = np.array([r,c1,c2,c3])
    return np.transpose(array)


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
# print(data)
# print(np.shape(data))
# print("Palette ratings - Column 0: ",data[:,0])
# print("Palette c1 - Column 1: ",data[:,1])
# print("One palette - Row 0: ",data[0])
# # data2 = np.transpose(data)
# print(data)
# print(data2)
# # #s = slice()#start:stop:step

"""
Data format of the numpy array:
    labels          samples
[[       1 11792581  2155287  2678288]
 [       1 12556180  1040714 11960544]
 [       2  6573489  5103277 11527642]
 [       1  9397964 15844496 14848293]
 [       1  1047846  7942961 11585421]]
"""