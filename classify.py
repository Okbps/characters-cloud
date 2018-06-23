import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, default='models/characters.model', help="path to trained model")
ap.add_argument("-l", "--labelbin", required=True, default='models/mlb.pickle', help="path to label binarizer")
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-dim", "--dimensions", type=str, default='96,96,3', help="image dimensions")
args = vars(ap.parse_args())

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2

IMAGE_DIMS = tuple(map(int, args['dimensions'].split(',')))

image = cv2.imread(args["image"])

image = cv2.resize(image, IMAGE_DIMS[:2])
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model(args["model"])
mlb = pickle.loads(open(args["labelbin"], "rb").read())

print("[INFO] classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1]

for (i, j) in enumerate(idxs):
    print("{}: {:.4f}".format(mlb.classes_[j], proba[j]))
