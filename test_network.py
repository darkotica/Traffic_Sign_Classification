from skimage import exposure
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator


def AHE(img):
    img = img/255
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq


test_gen = ImageDataGenerator(preprocessing_function=AHE)

test_dataset = test_gen.flow_from_directory(
    "test_data\\",
    color_mode="rgb",
    target_size=(55, 55)
)

json_file = open('model_clahe.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_clahe.h5")
print("Loaded model from disk")

loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Results\n###############\n")
results = loaded_model.evaluate(test_dataset, batch_size=32)
print("test loss, test acc:", results)
