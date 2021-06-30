import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

test_gen = ImageDataGenerator()

test_dataset = test_gen.flow_from_directory(
    'E:\\ProjekatORI\\Project\\Traffic_Sign_Classification\\test_data',
    color_mode="rgb",
    target_size=(55, 55)
)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Results\n###############\n")
results = loaded_model.evaluate(test_dataset, batch_size=32)
print("test loss, test acc:", results)
