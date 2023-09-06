import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
import tensorflow as tf
from datagen import DataGenerator

INIT_LR = 1e-4


def build_model():
    model = Sequential()

    # 5x5 Convolutional layers with stride of 2x2
    model.add(Conv2D(24, (5, 5), strides=(2, 2),
                     activation='elu', input_shape=(66, 200, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))

    # 3x3 Convolutional layers with stride of 1x1
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    # Flatten before passing to the fully connected layers
    model.add(Flatten())

    # Three fully connected layers
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(.25))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(.25))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(.25))

    # Output layer with linear activation
    model.add(Dense(1, activation="linear"))

    return model


xs = []
ys = []

with open('driving_dataset/data.txt') as f:
    for line in f:
        xs.append(line.split()[0])
        ys.append(float(line.split()[1]) * 3.14159265 / 180)

num_images = len(xs)

partition = {'train': xs[:int(len(xs)*0.8)],
             'validation': xs[-int(len(xs) * 0.2):]}
labels = dict(zip(xs, ys))

checkpoint_path = "save/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Parameters for datagenerator
params = {'dim': (66, 200, 3),
          'batch_size': 32,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition["train"], labels, **params)
validation_generator = DataGenerator(partition["validation"], labels, **params)

model = build_model()
model.compile(optimizer='adam', loss="mse")

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)
# train model for 10 epochs
model.fit_generator(generator=training_generator, epochs=10,
                    validation_data=validation_generator, callbacks=[cp_callback])

# Save trained model
model.save('save/model.h5')
