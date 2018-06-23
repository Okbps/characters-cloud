from keras.preprocessing.image import array_to_img
from matplotlib import pyplot as plt
import cv2

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(array_to_img(cv2.cvtColor(ims[i], cv2.COLOR_BGR2RGB)))
