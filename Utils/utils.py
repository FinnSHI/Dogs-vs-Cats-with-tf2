import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import os


class DataGenerator(tf.keras.utils.Sequence):
    # def __init__(self, config, dataset, shuffle=True, is_train=True):
    def __init__(self, dataset, shuffle, classes, width, height, channels, batch_size, is_train=True):
        self.dataset = dataset
        self.is_train = is_train
        self.len_dataset = len(dataset)
        self.indices = np.arange(self.len_dataset)
        self.shuffle = shuffle
        self.classes = classes
        self.width = width
        self.height = height
        self.channels = channels
        self.batch_size = batch_size

        if self.shuffle:
            self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):  # Returns number of batches per epoch
        return int(np.floor(self.len_dataset / self.batch_size))

    def __getitem__(self, index):  # Generated batch for given index

        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        id = [self.dataset[k] for k in indices]
        if self.is_train:
            x, y = self.__data_generation(id)
            return x, y
        else:
            x = self.__data_generation(id)
            return x

    def __data_generation(self, ids):

        x_batch = []
        y_batch = []
        if self.is_train:
            for instance in ids:
                # image, label = read_image(instance, size=(self.width, self.height), to_aug=self.aug)
                image, label = label_image(instance, size=(self.width, self.height))
                x_batch.append(image)
                y_batch.append(label)
            x_batch = np.asarray(x_batch, dtype=np.float32)
            y_batch = np.asarray(y_batch, dtype=np.float32)
            return x_batch, y_batch
        else:
            batch = []
            for img in ids:
                image = read_image_test(img, size=(self.width, self.height))
                batch.append(image)
            return np.asarray(batch)


# read the dataset
def read_dataset(source_path, shuffle=True):  # shuffle can mess imgs up
    imgs = os.listdir(source_path)
    preprocessed_paths = []
    for img in imgs:
        preprocessed_paths.append(os.path.join(source_path, img))
    preprocessed_paths = np.asarray(preprocessed_paths)
    if shuffle:
        np.random.shuffle(preprocessed_paths)  # mess up the order
    return preprocessed_paths


# read images for testing
def read_image_test(path, size):
    path = Path(path)
    if not path.exists():
        raise Exception("Image Not Found")
    else:
        image = cv2.imread(str(path))
        # By default cv2 reads in Blue,Green,Red. Format to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size)
        image = scaled_img(image)
        return np.asarray(image)


# read images for training
# def read_image(path, size, to_aug=False):
def label_image(path, size):
    path = Path(path)
    if not path.exists():
        raise Exception("Image Not Found")
    else:
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image,
                             cv2.COLOR_BGR2RGB)  # By default cv2 reads in Blue,Green,Red Format to convert in RGB
        image = cv2.resize(image, size)
        image = scaled_img(image)
        # cat 0, dog 1
        label = 0 if 'cat' in str(os.path.basename(path)) else 1
        return image, label


# scale images
def scaled_img(image):
    image = np.asarray(image)
    mean = np.mean(image, axis=0, keepdims=True)
    return (image - mean)


def get_callbacks(checkpoint_path, checkpoint_best_path, logs_path):
    callbacks = []

    early_stop = tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')
    callbacks.append(early_stop)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss',
                                                    save_best_only=False,
                                                    save_weights_only=True,
                                                    filepath=checkpoint_path)
    callbacks.append(checkpoint)

    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss',
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         filepath=checkpoint_best_path)
    callbacks.append(checkpoint_best)

    logs_path = logs_path
    board = tf.keras.callbacks.TensorBoard(logs_path, write_graph=True, write_images=True)
    callbacks.append(board)

    # if self.config['callbacks']['scheduler']['onecycle']['to_use']:
    #     iterations = np.ceil(self.train_size / self.batch_size) * self.epochs
    #     max_rate = self.config['callbacks']['scheduler']['onecycle']['max_rate']
    #     one_cycle_callback = one_cycle(iterations=iterations, max_rate=max_rate)
    #     callbacks.append(one_cycle_callback)
    #
    # if self.config['callbacks']['scheduler']['exponential_scheduler']['to_use']:
    #     s = self.config['callbacks']['scheduler']['exponential_scheduler']['params']
    #     exponential_scheduler_callback = exponential_scheduler(s=s)
    #     callbacks.append(exponential_scheduler_callback)
    return callbacks


def show_some_image_prediction(images, labels, path_to_save=None):
    n_rows = 10
    n_cols = 5

    plt.figure(figsize=(20, 20))
    for row in range(n_rows):
        for col in range(n_cols):
            idx = n_cols * row + col
            plt.subplot(n_rows, n_cols, idx + 1)
            img = cv2.imread(images[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = labels[idx]
            plt.imshow(img)
            plt.title(label, fontsize=12)
            plt.axis('off')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.show()