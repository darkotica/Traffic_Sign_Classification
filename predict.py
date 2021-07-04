from keras.models import model_from_json
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np


def load_image(filename, show=False):
    img = image.load_img(filename, target_size=(55, 55))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    if show:
        img_tensor2 = img_tensor / 255.
        plt.imshow(img_tensor2[0])
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":
    json_file = open('model_clahe.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_clahe.h5")
    print("Loaded model from disk")

    loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # image path
    img_path = ''

    if len(img_path) == 0:
        print("Niste izabrali putanju!")
        quit()

    # load a single image
    new_image = load_image(img_path, True)

    # check prediction
    pred = loaded_model.predict(new_image)

    # for output
    label_map = {0: "znak 60",
                 1: "znak 100",
                 2: "zabranjeno preticanje",
                 3: "raskrsnica na gl putu",
                 4: "pravo prvenstva",
                 5: "raskrsnica sa glavnim putem",
                 6: "stop znak",
                 7: "zabranjen smer",
                 8: "opasnost na putu",
                 9: "radovi na putu",
                 10: "jelen",
                 11: "obavezno pravo",
                 }
    print("Predictions:")
    for i in range(0, 12):
        print("\t{} : {:.5f} %".format(label_map[i], pred.item(i)*100))
