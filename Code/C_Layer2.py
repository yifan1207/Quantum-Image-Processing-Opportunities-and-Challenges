# Import the necessary modules
import tensorflow as tf
from tensorflow.keras import layers 

# Define the input shape
input_shape = (n, n, 64) # n is the number of pixels in one dimension of the image

# Create a sequential model
model = tf.keras.Sequential()

# Add the second convolutional layer
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))

# Print the model summary
model.summary()
