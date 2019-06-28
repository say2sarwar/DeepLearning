#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import time
import numpy as np
np.random.seed(int(time.time()))
import pandas as pd
import matplotlib.pyplot as plt
global SOLVER 


def split_test(ratio=0.8):
    x = pd.read_csv('Data.csv')
    train = x.sample(frac=ratio, random_state=100)
    test = x.drop(train.index)
    ytrain = train['Activities_Types']
    xtrain = train.drop('Activities_Types', axis=1)
    ytest = test['Activities_Types']
    xtest = test.drop('Activities_Types', axis = 1)
    
    #print(train.shape, test.shape)
    return xtrain.values, ytrain.values, xtest.values, ytest.values


class Layer:

    def __init__(self):
        pass
    
    def forward(self, x):
        return x

    def backward(self, z, output_gradient):
        num_units = z.shape[1]
        
        dem_input = np.eye(num_units)
        
        return np.dot(output_gradient, dem_input) # chain rule

class Softmax(Layer):
    def __init__(self):
        pass
    def forward(self, x):
        exp_scores = np.exp(z)
        return exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
class DenseAdam(Layer):
    def __init__(self, Neuron_number_input, Neuron_number_output, learning_rate_in=0.001,
                 beta_1=0.9,beta_2=0.999,eps=1e-8):
        self.learning_rate_in = learning_rate_in
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(Neuron_number_input+Neuron_number_output)), 
                                        size = (Neuron_number_input,Neuron_number_output))
        self.biases = np.zeros(Neuron_number_output)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.learning_rate = float(learning_rate_in)
        self.t = 0
        self.ms =np.zeros((Neuron_number_input, Neuron_number_output))
        self.vs = np.zeros((Neuron_number_input,Neuron_number_output))
        self.updates = np.zeros((Neuron_number_input,Neuron_number_output))
        
    def forward(self,x):
        return np.dot(x,self.weights) + self.biases
    def Adam(self, z, output_gradient):
        
        gradient_weights = np.dot(z.T, output_gradient)
        gradient_biases = output_gradient.mean(axis=0)*z.shape[0]
        
        
        self.t += 1
        self.ms = self.beta_1 * self.ms + (1 - self.beta_1) * gradient_weights
        self.vs = self.beta_2 * self.vs + (1 - self.beta_2) * np.power(gradient_weights, 2)
        m_hat = self.ms / (1 - np.power(self.beta_1, self.t))
        v_hat = self.vs / (1 - np.power(self.beta_2, self.t))
        self.weights = self.weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


       
        self.biases = self.biases - self.learning_rate * gradient_biases
    
    def backward(self,z,output_gradient):
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        self.Adam(z,output_gradient)
        
        return input_gradient

class ReLU(Layer):
    def __init__(self):
        pass
    
    def forward(self, x):
        relu_f = np.maximum(0,x)
        return relu_f
    
    def backward(self, z, output_gradient):
        relu_g = z > 0
        return output_gradient*relu_g


class DenseGD(Layer):
    def __init__(self, Neuron_number_input, Neuron_number_output, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(Neuron_number_input+Neuron_number_output)), 
                                        size = (Neuron_number_input,Neuron_number_output))
        self.biases = np.zeros(Neuron_number_output)
        
    def forward(self,x):
        return np.dot(x,self.weights) + self.biases
    def SGD(self, z, output_gradient):
        
        gradient_weights = np.dot(z.T, output_gradient)
        gradient_biases = output_gradient.mean(axis=0)*z.shape[0]

        self.weights = self.weights - self.learning_rate * gradient_weights
        self.biases = self.biases - self.learning_rate * gradient_biases
    
    def backward(self,z,output_gradient):
        input_gradient = np.dot(output_gradient, self.weights.T)
       
        self.SGD(z,output_gradient)
        
        return input_gradient

class DenseAdaGrad(Layer):
    def __init__(self, Neuron_number_input, Neuron_number_output, learning_rate=0.1,eps=0.001):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(Neuron_number_input+Neuron_number_output)), 
                                        size = (Neuron_number_input,Neuron_number_output))
        self.biases = np.zeros(Neuron_number_output)
        self.eps = eps
        
    def forward(self,x):
        return np.dot(x,self.weights) + self.biases
    def AdaG(self, z, output_gradient):
        
        gradient_weights = np.dot(z.T, output_gradient)
        gradient_biases = output_gradient.mean(axis=0)*z.shape[0]
        factor = self.learning_rate*gradient_weights/np.sqrt(self.eps+np.sum(gradient_weights)**2)
        

        self.weights = self.weights - factor
        self.biases = self.biases - self.learning_rate * gradient_biases
    
    def backward(self,z,output_gradient):
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        self.AdaG(z,output_gradient)
        
        return input_gradient



def softmax_crossentropy(hold,ydash):
    hold_ans = hold[np.arange(len(hold)),ydash]
    
    xentropy = - hold_ans + np.log(np.sum(np.exp(hold),axis=-1))
    
    return xentropy

def grad_softmax_crossentropy(hold,ydash):
    ones_ans = np.zeros_like(hold)
    ones_ans[np.arange(len(hold)),ydash] = 1
    
    softmax = np.exp(hold) / np.exp(hold).sum(axis=-1,keepdims=True)
    
    return (- ones_ans + softmax) / hold.shape[0]


X_train, Target_train, X_val, Target_Validation = split_test(0.8)
Target_train = np.subtract(Target_train,1)
Target_Validation = np.subtract(Target_Validation, 1)


def forward(model, X):
    activations = []
    input = X
    for l in model:
        activations.append(l.forward(input))
        
        input = activations[-1]
    
    return activations

def predict(model,X):
    hold = forward(model,X)[-1]
    #print(hold.argmax())
    return hold.argmax(axis=-1)
def LOSS(model,X,y):
    layer_activations = forward(model,X)
    layer_inputs = [X]+layer_activations
    hold = layer_activations[-1]
    loss = softmax_crossentropy(hold, y)
    return np.mean(loss)
    

def train(model,X,y):
    
    
    layer_activations = forward(model,X)
    layer_inputs = [X]+layer_activations  #
    hold = layer_activations[-1]
    
   
    loss = softmax_crossentropy(hold,y)
    loss_G = grad_softmax_crossentropy(hold,y)
    
    
    for layer_index in range(len(model))[::-1]:
        layer = model[layer_index]
        
        loss_G = layer.backward(layer_inputs[layer_index],loss_G) 
        
    return np.mean(loss)



from tqdm import trange, tnrange
def Batch(inputs, targets, Batch_size=128, shuffle=False):
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for index_counter in range(0, len(inputs) - Batch_size + 1, Batch_size):
        if shuffle:
            clip = indices[index_counter:index_counter + Batch_size]
        else:
            clip = slice(index_counter, index_counter + Batch_size)
        yield inputs[clip], targets[clip]

model = []
model.append(DenseGD(X_train.shape[1],150))
model.append(ReLU())
model.append(DenseGD(150,200))
model.append(ReLU())
model.append(DenseGD(200,100))
model.append(ReLU())
model.append(DenseGD(100,6))


Training_Accuracy = []
validation_accuracy = []
Training_loss =[]
Validation_loss = []
for epoch in trange(400, desc='progress'):

    for x_batch,y_batch in Batch(X_train,Target_train,Batch_size=64,shuffle=True):
        loss=train(model,x_batch,y_batch)
    
    Validation_loss.append(LOSS(model,X_val, Target_Validation))
    Training_loss.append(loss)    
    Training_Accuracy.append(np.mean(predict(model,X_train)==Target_train))
    validation_accuracy.append(np.mean(predict(model,X_val)==Target_Validation))

print("Epoch: ",epoch)
print("Training accuracy:",Training_Accuracy[-1])
print("Validation accuracy:",validation_accuracy[-1])
plt.plot(Training_Accuracy,label='train accuracy')
plt.title("Model Accuracy")
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.plot(validation_accuracy,label='val accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()
# Loss ploting
print("Training Loss: ", Training_loss[-1])
print('Validation Loss: ', Validation_loss[-1])
plt.plot(Training_loss, label='Training loss')
plt.plot(Validation_loss, label='Validation Loss')
plt.title("Model Loss")
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.grid()
plt.show()



from sklearn.metrics import classification_report
Target_hat= predict(model,X_val)
target_names =['1','2','3','4','5','6']
print(classification_report(Target_Validation,Target_hat,target_names=target_names))

test_file = pd.read_csv('Test_no_Ac.csv')
test_file_yhat=predict(model,test_file)+1
import csv
with open('file.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Index', 'Lable'])
    for i in range(len(test_file_yhat)):
    
        writer.writerow([i,test_file_yhat[i]])
    
f.close()

print("Final test file has been saved")

model = []
model.append(DenseAdam(X_train.shape[1],100))
model.append(ReLU())
model.append(DenseAdam(100,200))
model.append(ReLU())
model.append(DenseAdam(200,100))
model.append(ReLU())
model.append(DenseAdam(100,6))


Training_Accuracy = []
validation_accuracy = []
Training_loss =[]
Validation_loss = []
for epoch in trange(100, desc='progress'):

    for x_batch,y_batch in Batch(X_train,Target_train,Batch_size=128,shuffle=True):
        loss=train(model,x_batch,y_batch)
    
    Validation_loss.append(LOSS(model,X_val, Target_Validation))
    Training_loss.append(loss)    
    Training_Accuracy.append(np.mean(predict(model,X_train)==Target_train))
    validation_accuracy.append(np.mean(predict(model,X_val)==Target_Validation))

print("Epoch: ",epoch)
print("Training accuracy:",Training_Accuracy[-1])
print("Validation accuracy:",validation_accuracy[-1])
plt.plot(Training_Accuracy,label='train accuracy')
plt.plot(validation_accuracy,label='val accuracy')
plt.title("Model Accuracy")
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()
# Loss ploting
print("Training Loss: ", Training_loss[-1])
print('Validation Loss: ', Validation_loss[-1])
plt.plot(Training_loss, label='Training loss')
plt.plot(Validation_loss, label='Validation Loss')
plt.title("Model Loss")
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.grid()
plt.show()

from sklearn.metrics import classification_report
Target_hat= predict(model,X_val)
target_names =['1','2','3','4','5','6']
print(classification_report(Target_Validation,Target_hat,target_names=target_names))


True_targets=['dws', 'ups', 'sit','std','wlk','jog']
y_True = [True_targets[i] for i in Target_Validation]
len(y_True)

import pandas as pd


feat_cols = [ 'Feature'+str(i) for i in range(X_val.shape[1]) ]

df = pd.DataFrame(X_val,columns=feat_cols)
df['label'] = y_True
df['label'] = df['label'].apply(lambda i: str(i))

from sklearn.decomposition import PCA


pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 

from plotnine import *



chart = (ggplot( df, aes(x='pca-one', y='pca-two', color='label') )+ geom_point(size=2,alpha=1)+ ggtitle("First and Second Principal Components  by Label"))


ggplot.save(chart,'PCA.png')
print("Image save as PCA.png in current dir")

import time

from sklearn.manifold import TSNE

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
tsne_results = tsne.fit_transform(df[feat_cols].values)

df_tsne = df.copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') )         + geom_point(size=2,alpha=1)         + ggtitle("t-SNE dimensions colored by Label")
ggplot.save(chart,'tsne.png')

print("plot has been save in current dir as t-SNE.png")
