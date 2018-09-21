import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.callbacks import Callback
from random import randint

with open('sampleInterview.txt', 'r') as file:
    # any file can be read into script
    #  encoding='utf-8' - ignore this line... #signpost
    corpus = file.read()

chars = list(set(corpus))
data_size, vocab_size = len(corpus), len(chars)

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

sentence_length = 50    # we chunk sentences by char from corpus in sets of 50
sentences = []
next_chars = []
for i in range(data_size - sentence_length):    # ancestral sampling
    sentences.append(corpus[i: i + sentence_length])
    # we start at an index and then call the next 50 as specified above
    next_chars.append(corpus[i + sentence_length])
    # this maps from sentences in 50 chars to the next single char in sequence

# more preprocessing - we createinputs for our lstm
num_sentences = len(sentences)

#%%

# the shape of input has to be batch size  - entire number of examples
# sequence length and data size (embedding dims-this case our one hot vector)
X = np.zeros((num_sentences, sentence_length, vocab_size), dtype=np.bool)

# we ap chink of 50sentences to a single char
y = np.zeros((num_sentences, vocab_size), dtype=np.bool)

# where do we place the one hot encoding?..
# we go through all our examples and then set it to be the
# so for a particular example, set y to be the next charc
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_idx[char]] = 1    # the one hot encoding - index of correct char
    y[i, char_to_idx[next_chars[i]]] = 1

#%%

# we now feed the above into our lstm
model = Sequential()
model.add(LSTM(256, input_shape=(sentence_length, vocab_size)))
model.add(Dense(vocab_size))    # compute distribution across the data
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#%%
def sample_from_model(model, sample_length=100):
    seed = randint(0, data_size - sentence_length)
    seed_sentence = corpus[seed: seed + sentence_length]

    X_pred = np.zeros((1, sentence_length, vocab_size), dtype=np.bool)
    for t, char in enumerate(seed_sentence):
        X_pred[0, t, char_to_idx[char]] = 1

    generated_text = ''
    for i in range(sample_length):
        prediction = np.argmax(model.predict(X_pred))

        generated_text += idx_to_char[prediction]

        activation = np.zeros((1, sentence_length, vocab_size), dtype=np.bool)
        activation[0, 0, prediction] = 1
        X_pred = np.concatenate((X_pred[:, 1:, :], activation), axis=1)

    return generated_text

#%%
class SamplerCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        generated_text = sample_from_model(self.model)
        print('\nGenerated text')
        print('-' * 32)
        print(generated_text)

sampler_callback = SamplerCallback()
model.fit(X, y, epochs=2, batch_size=256, callbacks=[sampler_callback])

generated_text = sample_from_model(model, sample_length=1000)
print('\nGenerated text')
print('-' * 32)
print(generated_text)
