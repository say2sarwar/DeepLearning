#!/usr/bin/env python
# coding: utf-8
import numpy as np
# In[1]:
def preprocess(text, language='english', lower=True):
    """Tokenize and lower the text"""
    words = []
    tokenized_text = []

    for line in text:
        tokenized = nltk.word_tokenize(line, language=language)
        if lower:
            tokenized = [word.lower() for word in tokenized]
        tokenized_text.append(tokenized)
        for word in tokenized:
            words.append(word)

    most_common = Counter(words).most_common()

    return tokenized_text, most_common
# In[75]:


eng_sent = []
fra_sent = []
test_eng = []
test_fra = []
test_f = open('test.txt', encoding='utf-8-sig')
for i in test_f:
    test_eng.append(i.strip())
eng_chars = set()
fra_chars = set()
nb_samples = 137760
ff = open('fr.txt', 'r',encoding='utf-8-sig')
for i in ff:
    tmp = '\t' + i.strip() + '\n'
    fra_sent.append(tmp)
    for char in tmp:
        if (char not in fra_chars):
            fra_chars.add(char)
f = open('en.txt', 'r',encoding='utf-8-sig')
for i in f:
    eng_sent.append(i.strip())
    for char in i:
        if (char not in eng_chars):
            eng_chars.add(char)
eng_sent = np.array(eng_sent)
fra_sent = np.array(fra_sent)
test_eng = np.array(test_eng)

print(fra_chars)


def create_vocab(most_common_words, specials, threshold=0):

    word2ind = {}
    ind2word = {}
    i = 0

    for sp in specials:
        word2ind[sp] = i
        ind2word[i] = sp
        i += 1

    for word, freq in most_common_words:
        if freq >= threshold:
            word2ind[word] = i
            ind2word[i] = word
            i += 1

    assert len(word2ind) == len(ind2word)

    return word2ind, ind2word, len(word2ind)


from keras.models import Model
from keras.layers import Input, LSTM, Dense, LSTM
import numpy as np




fra_chars = sorted(list(fra_chars))
eng_chars = sorted(list(eng_chars))


eng_index_to_char_dict = {}


eng_char_to_index_dict = {}

for k, v in enumerate(eng_chars):
    eng_index_to_char_dict[k] = v
    eng_char_to_index_dict[v] = k

fra_index_to_char_dict = {}


fra_char_to_index_dict = {}
for k, v in enumerate(fra_chars):
    fra_index_to_char_dict[k] = v
    fra_char_to_index_dict[v] = k


# In[81]:


max_len_eng_sent = max([len(line) for line in eng_sent])
max_len_test_sent = max([len(line) for line in test_eng])
max_len_fra_sent = max([len(line) for line in fra_sent])


# In[82]:


max_len_eng_sent
max_len_fra_sent
tokenized_eng_sentences = np.zeros(shape = (nb_samples,max_len_eng_sent,len(eng_chars)), dtype='float32')
tokenized_test_sentences = np.zeros(shape = (nb_samples, max_len_test_sent, len(eng_chars)), dtype='float32')
tokenized_fra_sentences = np.zeros(shape = (nb_samples,max_len_fra_sent,len(fra_chars)), dtype='float32')
target_data = np.zeros((nb_samples, max_len_fra_sent, len(fra_chars)),dtype='float32')


for i in range(nb_samples):
    for k,ch in enumerate(eng_sent[i]):
        tokenized_eng_sentences[i,k,eng_char_to_index_dict[ch]] = 1
        
    for k,ch in enumerate(fra_sent[i]):
        tokenized_fra_sentences[i,k,fra_char_to_index_dict[ch]] = 1

        # decoder_target_data will be ahead by one timestep and will not include the start character.
        if k > 0:
            target_data[i,k-1,fra_char_to_index_dict[ch]] = 1
# Test vectorize the english sentences
for i in range(len(test_eng)):
    for k, ch in enumerate(test_eng[i]):
        tokenized_test_sentences[i,k,eng_char_to_index_dict[ch]] = 1

encoder_input = Input(shape=(None,len(eng_chars)))
encoder_LSTM = LSTM(512,return_state = True)
encoder_outputs, encoder_h, encoder_c = encoder_LSTM (encoder_input)
encoder_states = [encoder_h, encoder_c]

decoder_input = Input(shape=(None,len(fra_chars)))
decoder_LSTM = LSTM(512,return_sequences=True, return_state = True)
decoder_out, _ , _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
decoder_dense = Dense(len(fra_chars),activation='softmax')
decoder_out = decoder_dense (decoder_out)

model = Model(inputs=[encoder_input, decoder_input],outputs=[decoder_out])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['acc'])
history = model.fit(x=[tokenized_eng_sentences,tokenized_fra_sentences], 
          y=target_data,
          batch_size=64,
          epochs=50,
          validation_split=0.1)


encoder_model_inf = Model(encoder_input, encoder_states)

# Decoder inference model
decoder_state_input_h = Input(shape=(512,))
decoder_state_input_c = Input(shape=(512,))
decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input, 
                                                 initial_state=decoder_input_states)

decoder_states = [decoder_h , decoder_c]

decoder_out = decoder_dense(decoder_out)

decoder_model_inf = Model(inputs=[decoder_input] + decoder_input_states,
                          outputs=[decoder_out] + decoder_states )


def decode_seq(inp_seq):
    
    # Initial states value is coming from the encoder 
    states_val = encoder_model_inf.predict(inp_seq)
    
    target_seq = np.zeros((1, 1, len(fra_chars)))
    target_seq[0, 0, fra_char_to_index_dict['\t']] = 1
    
    translated_sent = ''
    stop_condition = False
    
    while not stop_condition:
        
        decoder_out, decoder_h, decoder_c = decoder_model_inf.predict(x=[target_seq] + states_val)
        
        max_val_index = np.argmax(decoder_out[0,-1,:])
        sampled_fra_char = fra_index_to_char_dict[max_val_index]
        translated_sent += sampled_fra_char
        
        if ( (sampled_fra_char == '\n') or (len(translated_sent) > max_len_fra_sent)) :
            stop_condition = True
        
        target_seq = np.zeros((1, 1, len(fra_chars)))
        target_seq[0, 0, max_val_index] = 1
        
        states_val = [decoder_h, decoder_c]
        
    return translated_sent


# In[88]:


for seq_index in range(100):
    inp_seq = tokenized_test_sentences[seq_index:seq_index+1]
    translated_sent = decode_seq(inp_seq)
    print('-')
    print('Source (English)\n', eng_sent[seq_index])
    print('\nTranslation (French)\n', translated_sent)
    test_fra.append(translated_sent)


# In[89]:


history_dict = history.history
history_dict.keys()


# In[90]:


import matplotlib.pyplot as plt

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


# In[93]:


len(test_fra)


# In[96]:


f = open('test_106761503.txt', 'w', encoding='utf-8')
for i in test_fra:
    f.write(i)
f.close()
