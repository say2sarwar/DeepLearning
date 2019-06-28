import pandas as pd
data = pd.read_csv("emnist-balanced-train.csv")
data_t = pd.read_csv('emnist-balanced-test.csv')
data_t.head()
a=list(data)[0]
b= list(data_t)[0]
indexNames = data_t[data_t[b] > 35].index
data_t.drop(indexNames, inplace=True)
indexNames0 = data_t[data_t[b] <10].index
data_t.drop(indexNames0, inplace=True)

indexNames = data[data[a] > 35].index
data.drop(indexNames, inplace=True)
indexNames0 = data[data[a] <10].index
data.drop(indexNames0, inplace=True)
data_t.shape, data.shape

train_data = data.iloc[:, 1:]
train_labels = data.iloc[:, 0]
test_data = data_t.iloc[:, 1:]
test_labels = data_t.iloc[:, 0]

train_labels = pd.get_dummies(train_labels)
test_labels = pd.get_dummies(test_labels)
train_labels.head()

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.4
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
train_data = train_data.values
train_labels = train_labels.values
test_data = test_data.values
test_labels = test_labels.values
del data, data_t
import matplotlib.pyplot as plt

plt.imshow(test_data[45].reshape([28, 28]), cmap='Greys_r')
plt.show()

import numpy as np
def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    image = noisy('gauss',image)
    return (image.reshape([28 * 28]))/255.0
train_data = np.apply_along_axis(rotate, 1, train_data)
test_data = np.apply_along_axis(rotate, 1, test_data)
plt.imshow(test_data[45].reshape([28, 28]), cmap='Greys_r')
plt.title("Corrected Image")
plt.show()
plt.figure()