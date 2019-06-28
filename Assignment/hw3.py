#!/usr/bin/env python
# coding: utf-8



import data_prepro as dataset
import cv2
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os


from numpy.random import seed
seed(10)
from tensorflow import set_random_seed
set_random_seed(2)

train_path='train'
batch_size = 128


classes = os.listdir(train_path)
num_classes = len(classes)
validation_size = 0
img_size = 128
num_channels = 3

data = dataset.read_train_sets(train_path, img_size, classes, Val_size=validation_size)
test = dataset.read_train_sets('test', img_size, classes, Val_size= validation_size)
print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))
print("Number of files in Training-set:\t\t{}".format(len(test.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(test.valid.labels)))


print("If Using Regularization chane Variable here")
condition = input("Type Y or N: >")
y_label =[]
for root, _, files in os.walk('test'):
    current_directory_path = os.path.abspath(root)
    class_label = [current_directory_path.split('/')[-1]]
    #print(class_label)
    if class_label != ['test']:
        y_label.append(class_label)
    

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')


y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
alpha = 0.001

filter_size_conv1 = 3 
num_filters_conv1 = 64

filter_size_conv2 = 3
num_filters_conv2 = 128

filter_size_conv3 = 3
num_filters_conv3 = 128
    
fc_layer_size = 2048

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters, name):  
    
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)

    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME', name=name)

    layer += biases

    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    layer = tf.nn.relu(layer)

    return layer, weights

    

def create_flatten_layer(layer):
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer, weights


layer_conv1,w_c1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1, name='conv1')
layer_conv2, w_c2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2, name='conv2')

layer_conv3, w_c3 = create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3,name='conv3')
          
layer_flat = create_flatten_layer(layer_conv3)

layer_fc1, w_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2, w_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                    labels=y_true)
if condition == 'y' or condition == 'Y':
    l2_loss = alpha*(tf.nn.l2_loss(w_c1)+tf.nn.l2_loss(w_c2)+tf.nn.l2_loss(w_c3)+
                tf.nn.l2_loss(w_fc1)+tf.nn.l2_loss(w_fc2))
    cross_entropy_l2 = tf.add(cross_entropy, l2_loss, name='loss')
else:
    cross_entropy_l2 = cross_entropy

cost = tf.reduce_mean(cross_entropy_l2)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 
saver = tf.train.Saver(max_to_keep=4)


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Test Accuracy: {2:>6.1%},  Test Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

loss_training = []
loss_test = []
accuracy_training = []
accuracy_test = []
import matplotlib.pyplot as plt
        
total_iterations = 0
def plotIT(Validation_loss, Training_loss, validation_accuracy, Training_Accuracy):
    print("Training accuracy:",Training_Accuracy[-1])
    print("Test accuracy:",validation_accuracy[-1])
    plt.plot(Training_Accuracy,label='train accuracy')
    plt.title("Model Accuracy")
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.plot(validation_accuracy,label='Test accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    # Loss ploting
    print("Training Loss: ", Training_loss[-1])
    print('Test Loss: ', Validation_loss[-1])
    plt.plot(Training_loss, label='Training loss')
    plt.plot(Validation_loss, label='Test Loss')
    plt.title("Model Loss")
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

model_save_name='caltec_model/'

def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = test.train.next_batch(batch_size)
       

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}
        

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            training_loss = session.run(cost, feed_dict=feed_dict_tr)
            epoch = int(i / int(data.train.num_examples/batch_size))
            
            acc = session.run(accuracy, feed_dict=feed_dict_tr)
            val_acc = session.run(accuracy, feed_dict=feed_dict_val)
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            
            loss_test.append(val_loss)
            loss_training.append(training_loss)
            accuracy_test.append(val_acc)
            accuracy_training.append(acc)
            #saver.save(session, '/caltec_model/my_test_model')
            
            saver.save(session, os.path.join(model_save_name, 'model'))
            #saver.save(session, '/') 
    plotIT(loss_test, loss_training, accuracy_test, accuracy_training)
    total_iterations += num_iteration

train(num_iteration=5000)





def plotWeight(weights_input):
    w = session.run(weights_input)
    weight_id = tf.reshape(w, [-1])
    weight_id = session.run(weight_id)
    weights = np.ones_like(weight_id)/float(len(weight_id))
    plt.hist(weight_id, 150, weights=weights)
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.show()

plotWeight(w_c1)
plotWeight(w_c3)
plotWeight(w_fc1)
plotWeight(w_fc2)



def plot_conv_layer(layer, image):
   
    feed_dict1 = {x: [image]}
    values = session.run(layer, feed_dict=feed_dict1)

    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    
    for i in range(0,16,1):
        img = values[0,:, :, i]
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    fig, axes = plt.subplots(8, 8)
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            img = values[0, :, :, i]

            ax.imshow(img, interpolation='nearest')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
image1 = cv2.imread('144.jpg')
plot_conv_layer(layer=layer_conv1, image=image1)



plot_conv_layer(layer=layer_conv3, image=image1)


pred_y=session.run(y_pred, feed_dict={x:[image1]})
pred_y = pred_y.astype(int)
result = np.where(pred_y == 1)
y_predict = y_label[int(result[1])]
plt.figure()
plt.title('Predicted: %s True label: pigeon'%y_predict)
image2 = cv2.imread('1.jpg')
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.show()

# normalization, regularization
sess = tf.Session()


layer_conv1,w_c1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1, name='conv1')
layer_drop1 = tf.nn.dropout(layer_conv1, keep_prob=0.5)
layer_conv2, w_c2 = create_convolutional_layer(input=layer_drop1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2, name='conv2')
layer_drop2 = tf.nn.dropout(layer_conv2, keep_prob=0.5)

layer_conv3, w_c3 = create_convolutional_layer(input=layer_drop2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3,name='conv3')
layer_drop3 = tf.nn.dropout(layer_conv3, keep_prob=0.5)
          
layer_flat = create_flatten_layer(layer_drop3)

layer_fc1, w_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)
layer_drop2 = tf.nn.dropout(layer_fc1, keep_prob=0.5)

layer_fc2, w_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
sess.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                    labels=y_true)
if condition == 'y' or condition == 'Y':
    l2_loss = alpha*(tf.nn.l2_loss(w_c1)+tf.nn.l2_loss(w_c2)+tf.nn.l2_loss(w_c3)+
                tf.nn.l2_loss(w_fc1)+tf.nn.l2_loss(w_fc2))
    cross_entropy_l2 = tf.add(cross_entropy, l2_loss, name='loss')
else:
    cross_entropy_l2 = cross_entropy

cost = tf.reduce_mean(cross_entropy_l2)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess.run(tf.global_variables_initializer()) 
saver = tf.train.Saver(max_to_keep=4)

loss_training = []
loss_test = []
accuracy_training = []
accuracy_test = []
import matplotlib.pyplot as plt
        
total_iterations = 0


def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = test.train.next_batch(batch_size)
       

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}
        

        sess.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = sess.run(cost, feed_dict=feed_dict_val)
            training_loss = sess.run(cost, feed_dict=feed_dict_tr)
            epoch = int(i / int(data.train.num_examples/batch_size))
            
            acc = sess.run(accuracy, feed_dict=feed_dict_tr)
            val_acc = sess.run(accuracy, feed_dict=feed_dict_val)
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            
            loss_test.append(val_loss)
            loss_training.append(training_loss)
            accuracy_test.append(val_acc)
            accuracy_training.append(acc)
            #saver.save(sess, '/caltec_model/my_test_model')
            
            saver.save(sess, os.path.join(model_save_name, 'model'))
            #saver.save(sess, '/') 
    plotIT(loss_test, loss_training, accuracy_test, accuracy_training)
    total_iterations += num_iteration

train(num_iteration=5000)