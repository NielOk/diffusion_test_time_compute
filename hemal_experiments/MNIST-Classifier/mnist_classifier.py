from __future__ import print_function

import os
import sys
import numpy as np
import datetime
import dateutil.tz
import argparse

from tensorflow.examples.tutorials.mnist import input_data

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image


parser = argparse.ArgumentParser()
parser.add_argument("--train", help="True for training a new model [False]", action='store_true')
parser.add_argument("--predict", help="True for predicting with an existing model [False]", action='store_true')
parser.add_argument("--epochs", help="Epochs to train [50]", type=int, default=50)
parser.add_argument("--learning_rate", help="Learning rate for the optimizer [0.001]", type=float, default=1e-3)
parser.add_argument("--batch_size", help="The size of batch images [64]", type=int, default=64)
parser.add_argument("--optimizer", help="Optimizer to use. Can be one of: SGD, RMSprop, Adadelta, Adam [Adam]",
                    type=str, default="Adam", choices=set(("SGD", "RMSprop", "Adadelta", "Adam")))
parser.add_argument("--val_size", help="The size of the validation set [5000]", type=int, default=5000)
parser.add_argument("--log_dir", help="Directory name to save the checkpoints and logs [log_dir]",
                    type=str, default="log_dir")
parser.add_argument("--data_set_path", help="Path where data set for training is stored. [mnist_data]",
                    type=str, default="mnist_data")
parser.add_argument("--model", help="Path to model used for prediction. [weights.hdf5]", type=str, default="weights.hdf5")
parser.add_argument("--img_path", help="Path to images to predict. []", type=str,)
FLAGS = parser.parse_args()


def create_log_dir():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    log_dir = FLAGS.log_dir+"/" + str(sys.argv[0][:-3]) + "_" + timestamp
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    # save command line arguments
    with open(log_dir + "/hyperparameters_"+timestamp+".csv", "wb") as f:
        for arg in FLAGS.__dict__:
            f.write(arg + "," + str(FLAGS.__dict__[arg]) + "\n")

    return log_dir


# use mnist data from the specified folder (download if not already there)
def load_mnist_data(path, val_size):
    mnist = input_data.read_data_sets(path, validation_size=val_size, one_hot=True)

    x_train = mnist.train.images
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    y_train = mnist.train.labels

    x_val = mnist.validation.images
    x_val = np.reshape(x_val, (-1, 28, 28, 1))
    y_val = mnist.validation.labels

    x_test = mnist.test.images
    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    y_test = mnist.test.labels

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# build the classification model
def build_model(optimizer, learning_rate, input_shape=(28, 28, 1)):
    weight_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, input_shape=input_shape, padding="same", strides=2,
                     kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=2, strides=1))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=4, padding="same", strides=2, kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=2, strides=1))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=4, padding="same", strides=2, kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=2, strides=1))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=4, padding="same", strides=2, kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    lr = learning_rate
    optimizers = {"SGD": keras.optimizers.SGD(lr=lr), "RMSprop": keras.optimizers.RMSprop(lr=lr),
                  "Adadelta": keras.optimizers.Adadelta(lr=lr), "Adam": keras.optimizers.Adam(lr=lr)}

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizers[optimizer],
                  metrics=['accuracy'])

    return model


# training the model
def train_model(log_dir):
    train_data, val_data, test_data = load_mnist_data(path=FLAGS.data_set_path, val_size=FLAGS.val_size)

    model = build_model(optimizer=FLAGS.optimizer, learning_rate=FLAGS.learning_rate)

    # callback for the training process
    save_model = keras.callbacks.ModelCheckpoint(log_dir+"/weights.hdf5", monitor='val_acc', mode='max', verbose=0,
                                                 save_best_only=True, save_weights_only=False, period=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='max')
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10, batch_size=32, write_graph=True,
                                              write_grads=False, write_images=False, embeddings_freq=0,
                                              embeddings_layer_names=None, embeddings_metadata=None)

    # train model
    model.fit(train_data[0], train_data[1],
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              verbose=1,
              validation_data=val_data,
              callbacks=[early_stopping, save_model, tensorboard])

    # calculate and store test set performance on the model with best validation error
    print("Calculating performance on test set...")
    model = keras.models.load_model(log_dir+"/weights.hdf5")
    score = model.evaluate(test_data[0], test_data[1], verbose=0)
    print('Test loss: {:.4f}'.format(score[0]))
    print('Test accuracy: {:.4f}'.format(score[1]))
    with open(log_dir+"/test_acc-{:.4f}_test_loss-{:.4f}.txt".format(score[1], score[0]), "wb") as file:
        file.write('Test accuracy: {:.4f}\n'.format(score[1]))
        file.write('Test loss: {:.4f}'.format(score[0]))


# predict image classes
def predict(model, img_path, batch_size):
    model = keras.models.load_model(model)
    # normalize image pixel values into range [0,1]
    img_generator = image.ImageDataGenerator(preprocessing_function=lambda img: img/255.0)
    validation_generator = img_generator.flow_from_directory(directory=img_path, target_size=(28,28), shuffle=False,
                                                             batch_size=batch_size, color_mode="grayscale",)

    score = model.evaluate_generator(validation_generator)
    print("Accuracy: {:.4f}".format(score[1]))


if FLAGS.train:
    log_dir = create_log_dir()
    train_model(log_dir)
elif FLAGS.predict:
    assert FLAGS.img_path is not None, "Please specify the directory in which the images are stored via \"--img_path\"."
    assert os.path.exists(FLAGS.img_path), "The specified path to the images does not exit: {}". \
        format(FLAGS.img_path)
    predict(FLAGS.model, FLAGS.img_path, FLAGS.batch_size)
else:
    print("No valid option chosen. Choose either \"--train\" or \"--predict\".")
    print("Use \"--help\" for an overview of the command line arguments.")