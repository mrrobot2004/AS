import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

sentence = "the sun rises early"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([sentence])
sequence = tokenizer.texts_to_sequences([sentence])[0]

X = np.array(sequence)
generator = TimeseriesGenerator(X, X, length=3, batch_size=1)

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=8, input_length=3))
model.add(LSTM(16))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(generator, epochs=200, verbose=0)

input_seq = np.array([sequence[:3]])
prediction = model.predict(input_seq)
predicted_index = np.argmax(prediction)
predicted_word = tokenizer.index_word[predicted_index]

print("Predicted 4th word:", predicted_word)