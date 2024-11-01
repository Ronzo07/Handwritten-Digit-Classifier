import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Function to create and compile a feed-forward neural network
def create_model():
    model = Sequential([
        # Flatten layer converts each 28x28 pixel image into a 1D array of 784 pixels
        Flatten(input_shape=(28, 28)),  # Specify input shape to match MNIST image dimensions (28 by 28)
        
        # First Dense layer with 128 neurons and ReLU activation
        # This is a fully connected layer that captures complex patterns in the data
        Dense(128, activation='relu'),
        
        # Second Dense layer with 64 neurons and ReLU activation for further transformation
        Dense(64, activation='relu'),
        
        # Output layer with 10 neurons for the 10 classes (digits 0-9)
        # Softmax activation outputs probabilities for each class, summing to 1
        Dense(10, activation='softmax')
    ])
    
    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(optimizer='adam',                 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']) # To print accuracy metric during training
    return model
