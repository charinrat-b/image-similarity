import matplotlib.pyplot as plt
import numpy as  np
from keras.applications.vgg16 import VGG16
from keras.applications import vgg16
from keras import models, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from annoy import AnnoyIndex
import random
import pandas as pd
from scipy import spatial
from keras.models import load_model
%matplotlib inline


def create_model():

    model=Sequential()
    model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(100,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(20,activation='softmax'))
    return model

def load_model_from_path(filepath):
    model = load_model(filepath)
    return model

def load_images_preds(numpy_filepath):
    data = np.load(numpy_filepath)
    img = data['images']
    preds = data['preds']
    return img, preds

def show_img(array):
    array = array.reshape(224,224,3)
    numpy_image = img_to_array(array)
    plt.imshow(np.uint8(numpy_image))
    plt.show()

def load_images_from_file(filepath):

    img = load_img(filepath,  target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape((1,) + img.shape)
    return img

def get_nearest_neighbor_and_similarity(preds1, K):
    dims = 4096
    n_nearest_neighbors = K+1
    trees = 10000
    file_index_to_file_vector = {}

    # build ann index
    t = AnnoyIndex(dims)
    for i in range(preds1.shape[0]):

        file_vector = preds1[i]
        file_index_to_file_vector[i] = file_vector
        t.add_item(i, file_vector)
    t.build(trees)

    for i in range(preds1.shape[0]):
        master_vector = file_index_to_file_vector[i]

        named_nearest_neighbors = []
        similarities = []
        nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)
    for j in nearest_neighbors:
#         print (j)
        neighbor_vector = preds1[j]
        similarity = 1 - spatial.distance.cosine(master_vector, neighbor_vector)
        rounded_similarity = int((similarity * 10000)) / 10000.0
        similarities.append(rounded_similarity)
    return similarities, nearest_neighbors


def get_similar_images(similarities, nearest_neighbors, images1):
    j = 0
    for i in nearest_neighbors:
        show_img(images1[i])
        print (similarities[j])
        j+=1

def main(new_image_file, model_file, image_pred_file,K):
    model2 = create_model()
    model = load_model_from_path(model_file)
    images, preds = load_images_preds(image_pred_file)
    new_im = load_images_from_file(new_image_file)
    new_im_pred = model.predict(new_im)
    images1 = np.append(images, new_im.reshape(1,1,224,224,3), axis=0)
    preds1 = np.append(preds, new_im_pred, axis=0)
    similarities, nearest_neighbors = get_nearest_neighbor_and_similarity(preds1,K)
    get_similar_images(similarities, nearest_neighbors, images1)


if __name__ == '__main__':
    main()
