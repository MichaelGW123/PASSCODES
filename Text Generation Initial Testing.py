# Recurrent Neural Network - Text Generation

# Part 1 - Data Preprocessing

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import os
import time

# Importing the training set
start = time.time()
specificFile = 'words60to65'
fileName = 'C://Users//Michael JITN//Documents//School//Masters Code//'+specificFile+'.txt'
fileGrab = open(fileName, 'r')
startingWords = []
number_of_lines = 50
for i in range(number_of_lines):
    line = fileGrab.readline()
    startingWords.append(line.strip())
nonempty_lines = [line.strip("\n") for line in fileGrab if line != "\n"]
line_count = len(nonempty_lines)+50
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
for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))

seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(1):
  print(chars_from_ids(seq))

for seq in sequences.take(5):
  print(text_from_ids(seq).numpy())


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


split_input_target(list("Tensorflow"))

dataset = sequences.map(split_input_target)
for input_example, target_example in dataset.take(1):
    print("Input :", text_from_ids(input_example).numpy())
    print("Target:", text_from_ids(target_example).numpy())

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
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)


for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
print()
print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())

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

EPOCHS = 16

checkpointPath = './modelweights/' + specificFile + '/model(1024)(16)-' + specificFile
training = True
away = True
turnOff = True

if (training):
  history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
  model.save_weights(checkpointPath)

  if (away):
    end = time.time()
    total = end - start
    line = f"File: {specificFile} \nImporting Data Time: {importingData} \nVectorizing Data Time: {vectorizingData} \nTraining Run Time: {total} seconds\n\n"
    record = open('./runtime.txt', "a", encoding='utf-8')
    record.write(line)
    if (turnOff):
      os.system("shutdown /s /t 1")
else:
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
  next_char = tf.constant([startingWords[0], startingWords[1], startingWords[2], startingWords[3], startingWords[4]
  , startingWords[5], startingWords[6], startingWords[7], startingWords[8], startingWords[9]
  , startingWords[10], startingWords[11], startingWords[12], startingWords[13], startingWords[14]
  , startingWords[15], startingWords[16], startingWords[17], startingWords[18], startingWords[19]
  , startingWords[20], startingWords[21], startingWords[22], startingWords[23], startingWords[24]
  , startingWords[25], startingWords[26], startingWords[27], startingWords[28], startingWords[29]
  , startingWords[30], startingWords[31], startingWords[32], startingWords[33], startingWords[34]
  , startingWords[35], startingWords[36], startingWords[37], startingWords[38], startingWords[39]
  , startingWords[40], startingWords[41], startingWords[42], startingWords[43], startingWords[44]
  , startingWords[45], startingWords[46], startingWords[47], startingWords[48], startingWords[49]
  , startingWords[50], startingWords[51], startingWords[52], startingWords[53], startingWords[54]
  , startingWords[55], startingWords[56], startingWords[57], startingWords[58], startingWords[59]
  , startingWords[60], startingWords[61], startingWords[62], startingWords[63], startingWords[64]
  , startingWords[65], startingWords[66], startingWords[67], startingWords[68], startingWords[69]
  , startingWords[70], startingWords[71], startingWords[72], startingWords[73], startingWords[74]
  , startingWords[75], startingWords[76], startingWords[77], startingWords[78], startingWords[79]
  , startingWords[80], startingWords[81], startingWords[82], startingWords[83], startingWords[84]
  , startingWords[85], startingWords[86], startingWords[87], startingWords[88], startingWords[89]
  , startingWords[90], startingWords[91], startingWords[92], startingWords[93], startingWords[94]
  , startingWords[95], startingWords[96], startingWords[97], startingWords[98], startingWords[99]
  , startingWords[100], startingWords[101], startingWords[102], startingWords[103], startingWords[104]
  , startingWords[105], startingWords[106], startingWords[107], startingWords[108], startingWords[109]
  , startingWords[110], startingWords[111], startingWords[112], startingWords[113], startingWords[114]
  , startingWords[115], startingWords[116], startingWords[117], startingWords[118], startingWords[119]
  , startingWords[120], startingWords[121], startingWords[122], startingWords[123], startingWords[124]
  , startingWords[125], startingWords[126], startingWords[127], startingWords[128], startingWords[129]
  , startingWords[130], startingWords[131], startingWords[132], startingWords[133], startingWords[134]
  , startingWords[135], startingWords[136], startingWords[137], startingWords[138], startingWords[139]
  , startingWords[140], startingWords[141], startingWords[142], startingWords[143], startingWords[144]
  , startingWords[145], startingWords[146], startingWords[147], startingWords[148], startingWords[149]
  , startingWords[150], startingWords[151], startingWords[152], startingWords[153], startingWords[154]
  , startingWords[155], startingWords[156], startingWords[157], startingWords[158], startingWords[159]
  , startingWords[160], startingWords[161], startingWords[162], startingWords[163], startingWords[164]
  , startingWords[165], startingWords[166], startingWords[167], startingWords[168], startingWords[169]
  , startingWords[170], startingWords[171], startingWords[172], startingWords[173], startingWords[174]
  , startingWords[175], startingWords[176], startingWords[177], startingWords[178], startingWords[179]
  , startingWords[180], startingWords[181], startingWords[182], startingWords[183], startingWords[184]
  , startingWords[185], startingWords[186], startingWords[187], startingWords[188], startingWords[189]
  , startingWords[190], startingWords[191], startingWords[192], startingWords[193], startingWords[194]
  , startingWords[195], startingWords[196], startingWords[197], startingWords[198], startingWords[199]])
  result = [next_char]

  f = open(r"Generated Files\\PREDwords (NC).txt", "a", encoding='utf-8')
  avg_length = length/line_count
  saturation = 10
  number_of_guesses = avg_length*line_count*saturation
  for n in range(number_of_guesses):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)
    if (n%10000==9999):
      temp = tf.strings.join(result)
      for pred in range(50):
        output = temp[pred].numpy().decode('utf-8')
        f.write(output+"\n")
      print(f'{n} completed')
      result.clear()

  end = time.time()
  f.close()
  print('\nRun time:', end - start)