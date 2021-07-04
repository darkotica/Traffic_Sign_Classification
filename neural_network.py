import cv2 as cv
from skimage import exposure
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def AHE(img):
    img = img/255
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq


data_gen = ImageDataGenerator(validation_split=0.2, preprocessing_function=AHE)

dataset_train = data_gen.flow_from_directory(
    'data',
    color_mode="rgb",
    shuffle=True,
    seed=1337,
    subset="training",
    batch_size=32,
    target_size=(55, 55),
    class_mode="categorical")

dataset_val = data_gen.flow_from_directory(
    'data',
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
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(10, kernel_size=3, activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation="relu"))
model.add(Dense(12, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x=dataset_train, validation_data=dataset_val, batch_size=32, epochs=10)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

# serialize model to JSON
model_json = model.to_json()
with open("model_clahe_new.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_clahe_new.h5")
print("Saved model to disk")



