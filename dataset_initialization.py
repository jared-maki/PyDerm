import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv

DATADIR = '/home/jared/Desktop/ham10000_dataset/Images/'
CATEGORIES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
training_data = []
CONST_CSVFILE = '/home/jared/Desktop/ham10000_dataset/HAM10000_metadata.csv'
IMG_SIZE = 50

table = []

with open (CONST_CSVFILE, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        table.append(row)


def create_training_data():
    for row in range(table.__len__()):
        for category in CATEGORIES:
            class_num = CATEGORIES.index(category)
            key = table[row][2]
            if (key == category):
                try:
                    img_array = cv2.imread((DATADIR + table[row][1] + '.jpg'), cv2.IMREAD_GRAYSCALE)          # pylint: disable=no-member
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))     # pylint: disable=no-member
                    training_data.append([new_array, class_num])
                except Exception as e:      # pylint: disable=unused-variable
                    pass

        # path = os.path.join(DATADIR, category)  # Path to given category's directory
    # print(len(file_list))
    # for imgid in file_list:
    #     try:
    #         img_array = cv2.imread((DATADIR + imgid + '.jpg'))          # pylint: disable=no-member
    #         new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))     # pylint: disable=no-member
                
    #          training_data.append([new_array, class_num])
    #     except Exception as e:      # pylint: disable=unused-variable
    #         pass
create_training_data()
print((len(training_data)))

import random
random.shuffle(training_data)

X = []
y = []



for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(len(X))
print(len(y))

import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()