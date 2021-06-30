import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

dataset_train = data_gen.flow_from_directory(
    'E:\\ProjekatORI\\Project\\Traffic_Sign_Classification\\data',
    color_mode="rgb",
    shuffle=True,
    seed=1337,
    subset="training",
    batch_size=32,
    target_size=(55, 55),
    class_mode="categorical")

dataset_val = data_gen.flow_from_directory(
    'E:\\ProjekatORI\\Project\\Traffic_Sign_Classification\\data',
    color_mode="rgb",
    shuffle=True,
    seed=1337,
    subset="validation",
    batch_size=32,
    target_size=(55, 55),
    class_mode="categorical")

model = Sequential()
model.add(Conv2D(6, kernel_size=3, activation="relu", input_shape=(55, 55, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(6, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(10, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation="relu"))
model.add(Dense(12, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x=dataset_train, validation_data=dataset_val, batch_size=32, epochs=10)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



