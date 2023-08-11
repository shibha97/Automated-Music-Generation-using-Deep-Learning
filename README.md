# Automated Music Generation using Deep Learning

Generate musical content using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM).


# Requirements:
- Python 3.x
- Music21
- Keras
- Tensorflow
- h5py

# Data:

Use your own data to create unique musical pieces. Recommendation: Use monophonic soundtracks for best results.

# Model:
The model is an RNN with LSTM layers implemented in Keras and TensorFlow. Key components:
- 3 LSTM layers
- 2 Dense layers
- Dropout layers for regularization
- RMSprop optimizer
- Categorical cross-entropy loss function
The model is trained for 200 epochs.

# Encoding:

The music data is encoded into integers for the input, with chords represented as strings of note IDs separated by periods. The output is one-hot encoded for the training.

# Generation:

The trained model generates sequences of 500 notes, starting from random indices in the input data. The output integers are decoded back into Music21 note and chord objects and written to a MIDI file.
