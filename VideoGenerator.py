import glob
import csv
import numpy as np
import os

# Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

NUMBER_OF_CLASSES = 2 
CLASSES = {
    'label1': 0,
    'label2': 1
}

class VideoGenerator:

    def __init__(self, width, height, frames, batch_size,
                 shuffle=True, inputdir="./data/", fileext=".npy"):
        self.dim_x = width
        self.dim_y = height
        self.dim_z = frames
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.inputdir = inputdir
        self.fileext = fileext

        self.filenames_train, self.filenames_test, self.filenames_eval = self.load_filenames()

    def generate(self, train_or_test_or_eval):
        filenames = self.filenames_train
        if train_or_test_or_eval == "test":
            filenames = self.filenames_test
        elif train_or_test_or_eval == "eval":
            filenames = self.filenames_eval
        while True:
            indexes = self.__get_exploration_order(filenames)
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                filenames_batch = [filenames[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]
                X, y = self.__data_generation(filenames_batch)
                yield X, y

    def __get_exploration_order(self, filenames):
        indexes = np.arange(len(filenames))
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, filenames_batch):
        X = np.empty((self.batch_size, self.dim_z, self.dim_y, self.dim_x, 1))
        y = np.empty(self.batch_size, dtype=int)

        for i, filename in enumerate(filenames_batch):
            X[i, :, :, :, 0] = np.load(filename)
            y[i] = CLASSES[os.path.basename(os.path.dirname(filename))]

        return X, self.sparsify(y)

    @staticmethod
    def sparsify(y):
        return np.array([[1 if y[i] == j else 0 for j in range(NUMBER_OF_CLASSES)]
                         for i in range(y.shape[0])])

    def load_filenames(self):
        train = glob.glob("{}train/**/*{}".format(self.inputdir, self.fileext), recursive=True)
        test = glob.glob("{}test/**/*{}".format(self.inputdir, self.fileext), recursive=True)
        eval = glob.glob("{}eval/**/*{}".format(self.inputdir, self.fileext), recursive=True)
        return train, test, eval
