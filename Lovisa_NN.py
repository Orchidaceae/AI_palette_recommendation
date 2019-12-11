from csv_reader import import_csvdata #used to read data from csv files as numpy arrays
import numpy as np

data = import_csvdata("lovisa")
print(np.shape(data))
print("Palette ratings - Row 0: ",data[0])
print("Palette c1 - Row 1: ",data[1])
print("One palette - Column 0: ",data[:,0])