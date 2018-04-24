import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from PIL import Image


def load_images():

    bc_benign = ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"]
    bc_malignant = ["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma"]
        #,"papillary_carcinoma"]

    image_path = "BreaKHis_v1/histology_slides/breast"
    image_path = os.path.abspath(image_path)

    for m in bc_malignant:

        images = []
        labels = []

        path = os.path.join(image_path, "malignant", "SOB", m, "*/*/*.png")
        print(path)
        file_list = glob.glob(path)

        for file in file_list:
            image = Image.open(file)
            im_resize = image.resize((46, 70), Image.ANTIALIAS)
            pixels = list(im_resize.getdata())
            width, height = im_resize.size
            img = [pixels[i * width:(i + 1) * width] for i in range(height)]
            images.append(img)
            labels.append(1)

    for b in bc_benign:

        path = os.path.join(image_path, "benign", "SOB", b, "*/*/*.png")
        print(path)
        file_list = glob.glob(path)
        for file in file_list:
            image = Image.open(file)
            im_resize = image.resize((46, 70), Image.ANTIALIAS)
            pixels = list(im_resize.getdata())
            width, height = im_resize.size
            img = [pixels[i * width:(i + 1) * width] for i in range(height)]
            images.append(img)
            labels.append(0)
    # le = LabelEncoder()
    # le.fit(labels)
    # labels = le.transform(labels)
    x = np.array(images)
    print(x)
    y = np.array(labels)
    print(y)
    np.save('x.npy', x)
    np.save('y.npy', y)


def create_data_sets(training_size, batch_size, is_training=False):

    if not(os.path.exists('x.npy') and os.path.exists('y.npy')):
        load_images()
    x = np.load('x.npy')
    y = np.load('y.npy')
    t_index = int(x.shape[0] * training_size)
    print(t_index)
    x, y = shuffle(x, y)
    print(x.shape)
    train_x = x[:t_index]
    print(train_x.shape)
    train_y = y[:t_index]
    if is_training:
        return train_x, train_y

    test_x = x[t_index:]
    test_y = y[t_index:]
    num_test_batch = (x.shape[0] - t_index) // batch_size

    return test_x, test_y, num_test_batch


def split_training(training_size, validation_size, batch_size, is_training=True):

    x, y = create_data_sets(training_size, batch_size, is_training)
    print(len(x))
    v_index = int(x.shape[0] * validation_size)
    print(v_index)
    train_x = x[v_index:]
    train_y = y[v_index:]
    val_x = x[:v_index]
    val_y = y[:v_index]
    print(train_y.shape)

    num_tr_batch = (x.shape[0] - v_index) // batch_size
    print(num_tr_batch)
    num_vl_batch = v_index // batch_size
    print(num_vl_batch)

    return train_x, train_y, num_tr_batch, val_x, val_y, num_vl_batch


def batch(batch_size):

    train_x, train_y, num_tr_batch, val_x, val_y, num_vl_batch = split_training(.8, .125, batch_size)
    data_queues = tf.train.slice_input_producer([train_x, train_y])
    x, y = tf.train.shuffle_batch(data_queues,
                                  batch_size=batch_size,
                                  capacity=batch_size * 2,
                                  min_after_dequeue=batch_size,
                                  allow_smaller_final_batch=False)
    return x, y
