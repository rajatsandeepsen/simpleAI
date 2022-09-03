
# install tensorflow using- pip install tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense

import numpy as np
from pickletools import optimize


# creating one Hidden Layer with Single Node
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# no idea. Sounds important for Learning
model.compile(optimizer='sgd', loss='mean_squared_error')

# input data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

# output data
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# the input values and data is based on the Eqn of "2X-1 = Y"

# training for 500 times
model.fit(xs, ys, epochs=500)

# this a  sample input X value to predict the Y
print(model.predict([15.0]))
# this should give an Approximate value of 28.99997 something


# training more than 500 didn't make any further contribution without more data.

# printing the array (the model)
print(np.vstack(model, model))
# [array([[1.9980245]], dtype=float32), array([-0.9938751], dtype=float32)]

# did you see that.... 1.9980245 is almost equal to 2X
# and -0.9938751 is almost equal to -1


print(model.predict([100.0]))
# [[198.80856]]
