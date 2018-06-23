import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d",   "--dataset",        type=str,   default='data/ClothingAttributeDataset', help="path to input dataset")
ap.add_argument("-m",   "--model",          type=str,   default='models/characters.model', help="path to output model")
ap.add_argument("-l",   "--labelbin",       type=str,   default='models/mlb.pickle', help="path to output label binarizer")
ap.add_argument("-p",   "--plot",           type=str,   default="notebooks/plot.png", help="path to output accuracy/loss plot")
ap.add_argument("-dim", "--dimensions",     type=str,   default='96,96,3', help="image dimensions")
ap.add_argument("-r",   "--rows",           type=int,   default=1856, help="number of rows")
ap.add_argument("-e",   "--epochs",         type=int,   default=50, help="number of epochs")
ap.add_argument("-b",   "--batch",          type=int,   default=8, help="batch size")
ap.add_argument("-lr",  "--learn",          type=float, default=1e-3, help="initial learning rate")
ap.add_argument("-a",   "--architecture",   type=str,   default='smallervggnet', help="CNN architecture to build")

args = vars(ap.parse_args())

import numpy as np
import matplotlib.pyplot as plt
from utils.common import  get_explanation
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.optimizers import Adam
import cv2
import pandas as pd
import json
import pickle

IMAGE_DIMS = tuple(map(int, args['dimensions'].split(',')))
labels = []
data = []

df_labels = pd.read_csv(args['dataset'] + '/labels.csv', nrows=args['rows'])
df_labels = df_labels.fillna(value=0.0)
col_names = [x for x in df_labels.columns if x!='image']
df_labels[col_names] = df_labels[col_names].astype(dtype='int32')

imagePaths = args['dataset'] + '/images/' + df_labels['image'].values

with open(args['dataset'] + '/label_values.json') as json_file:
    labels_explain = json.load(json_file)

for x in range(df_labels.shape[0]):
    labels.append([get_explanation(labels_explain, c, str(df_labels.loc[x, c])) for c in col_names])

labels = np.array(labels)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

data = np.array(data, dtype="float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.1, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

if args['architecture']=='extendedvggnet':
    from classifiers.extendedvggnet import ExtendedVGGNet

    model = ExtendedVGGNet.build(
        width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
        depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
        finalAct="sigmoid")
else:
    from classifiers.smallervggnet import SmallerVGGNet

    model = SmallerVGGNet.build(
        width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
        depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
        finalAct="sigmoid")


opt = Adam(lr=args['learn'], decay=args['learn']/args['epochs'])

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=args['batch']),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // args['batch'],
    epochs=args['epochs'], verbose=1)

print("[INFO] serializing network...")
model.save(args["model"])

print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()

plt.style.use("ggplot")
plt.figure()
N = args['epochs']
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
