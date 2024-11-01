# MNIST Digit Classification with Feed-Forward Neural Network

## Project Overview

This project implements a simple feed-forward neural network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## Dataset

The dataset used in this project is the **MNIST** dataset, which consists of 70,000 grayscale images of handwritten digits (0-9). Each image is 28x28 pixels, and the dataset includes 60,000 training images and 10,000 testing images.

- **Source**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- **Input**: 28x28 pixel images
- **Output**: Corresponding labels (digits 0-9)

## Project Structure

The project consists of the following files:

1. **`data_loader.py`**: Contains the function to load and preprocess the MNIST dataset.
   - **Function**: `load_data()` - Loads the MNIST dataset and normalizes pixel values.

2. **`neural_network.py`**: Defines the architecture of the feed-forward neural network.
   - **Function**: `create_model()` - Creates and compiles the neural network model.

3. **`main.ipynb`**: Jupyter notebook for training and evaluating the model.
   - Contains code to load data, create the model, train it on the dataset, and evaluate its performance.

## How to Use

To run this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ronzo07/Handwritten-Digit-Classifier.git
   cd Handwritten-Digit-Classifier
   ```

2. **Install required packages**:
   Ensure you have Python and TensorFlow installed. You can install the necessary libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the notebook cells**:
   Run the cells in the notebook to load the dataset, create the model, train it, and evaluate its performance. The final model accuracy will be printed after evaluation on the test set.

## Explanation of the Model

The feed-forward neural network architecture consists of:
- **Input Layer**: A flattening layer that converts each 28x28 image into a 1D array of 784 pixels.
- **Hidden Layers**: 
  - The first dense layer with 128 neurons and ReLU activation captures complex patterns in the data.
  - The second dense layer with 64 neurons and ReLU activation further transforms the data representation.
- **Output Layer**: A dense layer with 10 neurons (one for each digit) using softmax activation to produce probabilities for classification.

The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function, suitable for multi-class classification tasks.
