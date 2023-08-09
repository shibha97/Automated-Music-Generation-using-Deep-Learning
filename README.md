# Automated-Music-Generation-using-Deep-Learning

This paper discusses how to create computer-generated monophonic musical content using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM). The goal is to generate music that has a melodic quality and can be played from sheet music by humans.

Data
The input data is piano music from the Final Fantasy video game soundtracks, selected for their melodic nature. There are 92 songs in the dataset with 352 different notes and chords.

Model
The model is an RNN with LSTM layers implemented in Keras and TensorFlow. Key components:
3 LSTM layers
2 Dense layers
Dropout layers for regularization
RMSprop optimizer
Categorical cross-entropy loss function
The model is trained for 200 epochs.

Encoding
The music data is encoded into integers for the input, with chords represented as strings of note IDs separated by periods. The output is one-hot encoded for the training.

Generation
The trained model generates sequences of 500 notes, starting from random indices in the input data. The output integers are decoded back into Music21 note and chord objects and written to a MIDI file.

Results
The generated music exhibits basic melodic structure and alternates between notes and chords. There is room for improvement by training on more data, adding complexity, and experimenting with parameters.
