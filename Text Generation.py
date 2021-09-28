# Recurrent Neural Network - Text Generation

# Part 1 - Data Preprocessing

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import os
import time

# Importing the training set
specificFile = 'words70to80'
# Flag for if training or generating
training = False
# Flags for if code will be running on its own to save run data and turn off computer
away = True
turnOff = True
# Flag for saving a specific weight from the training checkpoints if model deteriorates
saveNewWeight = False
EPOCHS = 16
# Variable for array of starter words (Increasing reduces time but increases computational load)
number_of_lines = 2000

start = time.time()
fileName = 'C://Users//Michael JITN//Documents//School//Masters Code//DeepLearningEntropy//'+specificFile+'.txt'
fileGrab = open(fileName, 'r')
startingWords = []
for i in range(number_of_lines):
    line = fileGrab.readline()
    startingWords.append(line.strip())
nonempty_lines = [line.strip("\n") for line in fileGrab if line != "\n"]
line_count = len(nonempty_lines)+number_of_lines
print(f'Number of lines: {line_count} lines')
fileGrab.close()
text = open(fileName, 'rb').read().decode(encoding='utf-8')
length = len(text)
print(f'Length of text: {length} characters')
end = time.time()
importingData = end - start
# The unique characters in the file
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

# Vectorize the Text
ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)

chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)


def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
end = time.time()
vectorizingData = end - start
# Part 2 Creating the RNN
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x


model = MyModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)


for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)

model.summary()

# Part 3 - Training the RNN
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", mean_loss)

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


checkpointPath = f'./modelweights/{specificFile}/model(1024)({EPOCHS})-{specificFile}'

if (training):
  history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
  model.save_weights(checkpointPath)

  if (away):
    end = time.time()
    total = end - start
    line = f"File: {specificFile} \nImporting Data Time: {importingData} \nVectorizing Data Time: {vectorizingData} \nTraining Run Time: {total} seconds\n\n"
    record = open('./runtime.txt', "a", encoding='utf-8')
    record.write(line)
    record.close()
else:
  if (saveNewWeight):
    model.load_weights(f'./training_checkpoints/ckpt_{EPOCHS}')
    model.save_weights(checkpointPath)
    exit()
  
  model.load_weights(checkpointPath)


  # Step 4 - Generating Text
  print("Generating Text Begun - ")

  class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
      super().__init__()
      self.temperature = temperature
      self.model = model
      self.chars_from_ids = chars_from_ids
      self.ids_from_chars = ids_from_chars

      # Create a mask to prevent "[UNK]" from being generated.
      skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
      sparse_mask = tf.SparseTensor(
          # Put a -inf at each bad index.
          values=[-float('inf')]*len(skip_ids),
          indices=skip_ids,
          # Match the shape to the vocabulary
          dense_shape=[len(ids_from_chars.get_vocabulary())])
      self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
      # Convert strings to token IDs.
      input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
      input_ids = self.ids_from_chars(input_chars).to_tensor()

      # Run the model.
      # predicted_logits.shape is [batch, char, next_char_logits]
      predicted_logits, states = self.model(inputs=input_ids, states=states,
                                            return_state=True)
      # Only use the last prediction.
      predicted_logits = predicted_logits[:, -1, :]
      predicted_logits = predicted_logits/self.temperature
      # Apply the prediction mask: prevent "[UNK]" from being generated.
      predicted_logits = predicted_logits + self.prediction_mask

      # Sample the output logits to generate token IDs.
      predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
      predicted_ids = tf.squeeze(predicted_ids, axis=-1)

      # Convert from token ids to characters
      predicted_chars = self.chars_from_ids(predicted_ids)

      # Return the characters and model state.
      return predicted_chars, states


  one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

  start = time.time()
  states = None

  next_char = tf.constant(startingWords)
  result = [next_char]
  avg_length = length/line_count
  avg_length_int = int(avg_length)
  saturation = 10
  outputPath = f"./Generated Files/PRED{specificFile}-{saturation*100}.txt"
  f = open(outputPath, "a", encoding='utf-8')
  number_of_guesses = int(avg_length*line_count*saturation/number_of_lines)
  partialGuess = int(number_of_guesses/100)
  print(number_of_guesses)
  for n in range(number_of_guesses):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)
    if (n%partialGuess==(partialGuess-1)):
      temp = tf.strings.join(result)
      for pred in range(number_of_lines):
        output = temp[pred].numpy().decode('utf-8')
        f.write(output+"\n")
      print(f'{n} completed')
      result.clear()
  f.close()
  if (away):
    end = time.time()
    total = end - start
    line = f"File: {specificFile} \nImporting Data Time: {importingData} \nVectorizing Data Time: {vectorizingData} \nGenerating Data: {number_of_guesses} \nGenerating Time: {total} seconds\n\n"
    record = open('./runtime.txt', "a", encoding='utf-8')
    record.write(line)
    record.close()
  else:
    print('\nRun time:', end - start)
  
if (turnOff):
      os.system("shutdown /s /t 1")