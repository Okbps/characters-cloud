# import the necessary packages
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Flatten, Dropout

class ExtendedVGGNet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
        vgg16_model = VGG16(include_top=False, weights='imagenet', input_shape=(width, height, depth))

        for layer in vgg16_model.layers:
            layer.trainable = False

        x = vgg16_model.output
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(classes)(x)
        predictions = Activation(finalAct)(x)

        model = Model(inputs=vgg16_model.input, outputs=predictions)

        return model
