# Michael Williamson
# Masters Research
# Revised Version
# Recurrent Neural Network - Text Generation


########## Part 0 - Importing the libraries ###########
# Part 1 - Data Preprocessing

# Importing the libraries
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

# Imported training set
specific_file = 'entropy_bin_00_output'

training = True
generating = True
modelVer = 5
EPOCHS = 10
save_new_weight = False

# Variable for array of starter words (Increasing reduces time but increases computational load)
number_of_lines = 2000

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 512
if (modelVer == 1):
    rnn_units = 1024

start = time.time()
# Set the file name for opening the file
current_dir = os.getcwd()
file_end = '/Source_Files/text_data/'+specific_file+'.txt'
file_name = current_dir + file_end
# Checkpoint to load if saving
check_to_load = "./training_checkpoints/{modelVer}_Hidden_Layers/ckpt_{rnn_units}_{EPOCHS}"

if (not os.path.exists(file_name)):
    print("File Name specified does not exist.")
    exit()
if (not os.path.exists(check_to_load) and save_new_weight):
    print("Checkpoint to load does not exist.")
    exit()

# Open the file to grab the first 'number_of_lines' you decided to populate the model with when generating
file_grab_start_words = open(file_name, 'r')
starting_words = []
for i in range(number_of_lines):
    line = file_grab_start_words.readline()
    starting_words.append(line.strip())
# count the rest of the lines in the file
nonempty_lines = [line.strip("\n")
                  for line in file_grab_start_words if line != "\n"]
line_count = len(nonempty_lines)+number_of_lines
file_grab_start_words.close()
print(f'Number of lines: {line_count} lines')

text = open(file_name, 'rb').read().decode(encoding='utf-8')
length = len(text)
print(f'Length of text: {length} characters')
average_length = length/line_count
print(f'Average Length of each line: {average_length}')
end = time.time()
importing_data_time = end - start

# The unique characters in the file
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

# Vectorize the Text

# preprocessing.StringLookup converts each character into a numeric ID
ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# For each input sequence the corresponding target contains the same length of text but it contains the following chars
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = round(average_length)
examples_per_epoch = len(text)//(seq_length+1)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

# Create Training Batches
# Batch size
BATCH_SIZE = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (dataset.shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))


end = time.time()
vectorizing_data_time = end - start
print(vectorizing_data_time)

# Part 2 Creating the RNN

# Length of the vocabulary in chars
vocab_size = len(vocab)

# Creating the model, adding the necessary layers


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_gru_layers=1):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru_layers = []
        for _ in range(num_gru_layers):
            self.gru_layers.append(tf.keras.layers.GRU(
                rnn_units, return_sequences=True, return_state=True))
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)

        if states is None:
            states = [gru.get_initial_state(x) for gru in self.gru_layers]

        for gru, state in zip(self.gru_layers, states):
            x, state = gru(x, initial_state=state, training=training)
            states.append(state)

        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


# Usage
model = MyModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    num_gru_layers=modelVer)


for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)

model.summary()

# Part 3 - Training the RNN
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()
print("Prediction shape: ", example_batch_predictions.shape,
      " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", mean_loss)

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = f'./training_checkpoints/{modelVer}_Hidden_Layers/{rnn_units}/'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
print(checkpoint_prefix)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# Path of file for the checkpoints
checkpoint_path = f'./model_weights/{modelVer}_Hidden_Layers/{specific_file}/model({rnn_units})({EPOCHS})-{specific_file}'

# If training, fit the model, save the weights, then save the runtime statistics
if (training):
    es = EarlyStopping(monitor='loss', min_delta=0.0015,
                       mode='min', verbose=1, patience=3)
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[
                        es, checkpoint_callback])
    early_stop = len(history.history['loss'])
    checkpoint_path = f'./model_weights/{modelVer}_Hidden_Layers/{specific_file}/model({rnn_units})({early_stop})-{specific_file}'
    model.save_weights(checkpoint_path)

    # Review models loss and training for evaluation
    print(history.history['loss'])
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    graph_directory_path = f'Training_Graphs/{modelVer}_Hidden_Layers/'
    if not os.path.exists(graph_directory_path):
        os.makedirs(graph_directory_path)
    plt.savefig(
        f'{graph_directory_path}{specific_file}_{rnn_units}_training.png')
    end = time.time()
    total = end - start
    line = f"File: {specific_file} \nImporting Data Time: {importing_data_time} \nVectorizing Data Time: {vectorizing_data_time} \nLayers: {modelVer}\n Neurons: {rnn_units}\nTotal Run Time: {total} seconds\nLoss: {history.history['loss'][-4:]}\n\n"
    record = open('./training_checkpoints/runtime.txt', "a", encoding='utf-8')
    record.write(line)
    record.close()
if (generating):
    if (save_new_weight):  # If we are saving a weight from the checkpoint, simply load, save, exit
        model.load_weights(
            f'./training_checkpoints/{modelVer}_Hidden_Layers/ckpt_{rnn_units}_{EPOCHS}').expect_partial()
        model.save_weights(checkpoint_path)

    # If we are not saving a new weight then the model is loaded and good to generate
    model.load_weights(checkpoint_path)

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
            predicted_ids = tf.random.categorical(
                predicted_logits, num_samples=1)
            predicted_ids = tf.squeeze(predicted_ids, axis=-1)

            # Convert from token ids to characters
            predicted_chars = self.chars_from_ids(predicted_ids)

            # Return the characters and model state.
            return predicted_chars, states

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    start = time.time()
    states = None

    # Sets the tensorflow object to the array of starting words, equal to the first number_of_lines passwords in the file
    next_char = tf.constant(starting_words)
    result = [next_char]
    # Saturation is the multiple of the original file. So in the case of saturation 10, we generate 10 times the trained file
    saturation = 10
    number_of_guesses = int(length*saturation/number_of_lines)
    partialGuess = int(number_of_guesses/100)
    print(number_of_guesses)
    out_directory_path = f'./Generated_Files/{modelVer}_Hidden_Layers/'
    if not os.path.exists(out_directory_path):
        os.makedirs(out_directory_path)
    output_path = f'{out_directory_path}PRED{specific_file}-{saturation*100}(RNN).txt'
    f = open(output_path, "a", encoding='utf-8')
    for n in range(number_of_guesses):
        next_char, states = one_step_model.generate_one_step(
            next_char, states=states)
        result.append(next_char)
        if (partialGuess != 0 and n % partialGuess == (partialGuess-1)):  # Helps keep track of progress
            # Joins the tensorflow objects into strings
            temp = tf.strings.join(result)
            # loops through the tf matrix to decode and write the strings to the output file
            for pred in range(number_of_lines):
                output = temp[pred].numpy().decode('utf-8')
                f.write(output+"\n")
            print(f'{n} completed')
            # clears the strings already written to the file, 'next_char' maintains the last character for predictions
            result.clear()
        else:
            temp = tf.strings.join(result)
            for pred in range(number_of_lines):
                output = temp[pred].numpy().decode('utf-8')
                f.write(output+"\n")
            print(f'{n} completed')
    f.close()
    end = time.time()
    total = end - start
    line = f"File: {specific_file} \nImporting Data Time: {importing_data_time} \nVectorizing Data Time: {vectorizing_data_time} \nGenerating Data: {number_of_guesses} \nGenerating Time: {total} seconds\n\n"

    record = open('./Generated_Files/runtime.txt', "a", encoding='utf-8')
    record.write(line)
    record.close()
    print(line)

# This executes the python script so that whether you want to shutdown after a run
# completes can be determined at any point and doesn't have to be fixed to whatever
# was decided at execution.
os.system("python Helpers/shutdown_comp.py")
