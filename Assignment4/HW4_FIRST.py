#!/usr/bin/env python
# coding: utf-8

# In[39]:


import tensorflow as tf


imdb = tf.keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3





train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=120)

test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=120)


# In[ ]:





# In[40]:




import tensorflow as tf
#from future import absolute_import, division, print_function
from sklearn.metrics import roc_curve
from tensorflow import keras as tff
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)


# In[41]:


vocab_size = 10000

model = tff.Sequential()
model.add(tff.layers.Embedding(vocab_size, 20))
model.add(tff.layers.GlobalAveragePooling1D())
model.add(tff.layers.Dropout(0.8))
model.add(tff.layers.Dense(20, activation=tf.nn.relu))
model.add(tff.layers.Dropout(0.8))
model.add(tff.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()


# In[42]:


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['acc'])
x_val = train_data[:5000]
partial_x_train = train_data[5000:]

y_val = train_labels[:5000]
partial_y_train = train_labels[5000:]
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# In[43]:


results = model.evaluate(test_data, test_labels)

print(results)
print("Test Accuracy: %0.4f%%"%(results[1]*100))
history_dict = history.history
history_dict.keys()


# In[46]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[47]:


test_pred = model.predict(test_data).ravel()


# In[48]:



fpr, tpr, thresholds = roc_curve(test_labels, test_pred)
from sklearn.metrics import auc
auc = auc(fpr, tpr)


# In[49]:


plt.figure()
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(fpr, tpr, 'm', label='AUC (Area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (RNN) Test')
plt.legend(loc='best')
plt.show()


# In[50]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
precision, recall, _ = precision_recall_curve(test_labels, test_pred)
auprc = average_precision_score(test_labels, test_pred)
plt.plot(precision, recall, 'b', label='AUPRC (Area = {:.3f})'.format(auprc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve (RNN) Test')
plt.legend(loc='best')
plt.show()


# In[72]:


unit = input("Please Type GRU or LSTM :> ")

model_lstm = tff.Sequential()
model_lstm.add(tff.layers.Embedding(vocab_size, 20))
model_lstm.add(tff.layers.Dropout(0.9))
if unit == 'lstm'.lower():
    #model_lstm.add(tff.layers.CuDNNLSTM(100,return_sequences=True))
    model_lstm.add(tff.layers.CuDNNLSTM(100))
else:
    model_lstm.add(tff.layers.GRU(100))
model_lstm.add(tff.layers.Dropout(0.2))
model_lstm.add(tff.layers.Dense(1, activation=tf.nn.sigmoid))

model_lstm.summary()


# In[68]:


model_lstm.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
x_val = train_data[:5000]
partial_x_train = train_data[5000:]

y_val = train_labels[:5000]
partial_y_train = train_labels[5000:]
history = model_lstm.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# In[69]:


results = model_lstm.evaluate(test_data, test_labels)

print(results)
print("Test Accuracy: %0.4f%%"%(results[1]*100))
history_dict = history.history
history_dict.keys()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss (%s) '%unit.upper())
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy (%s)'%unit.upper())
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[70]:


test_pred = model_lstm.predict(test_data).ravel()
fpr, tpr, thresholds = roc_curve(test_labels, test_pred)
from sklearn.metrics import auc
auc = auc(fpr, tpr)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, 'm', label='AUC (Area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (%s) Test'%unit.upper())
plt.legend(loc='best')
plt.show()

precision, recall, _ = precision_recall_curve(test_labels, test_pred)
auprc = average_precision_score(test_labels, test_pred)
plt.plot(precision, recall, 'b', label='AUPRC (Area = {:.3f})'.format(auprc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve (%s) Test'%unit.upper())
plt.legend(loc='best')
plt.show()


# In[252]:


'lsTM'.lower()


# In[ ]:




