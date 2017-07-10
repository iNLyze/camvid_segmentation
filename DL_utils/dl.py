# %% LIBRARIES


import os
import re
import time
import random
import glob
import numpy as np
from numpy.random import random, permutation, randn, normal, uniform, choice
from six.moves import cPickle as pickle
import bcolz
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, Convolution3D, UpSampling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Input
from keras.layers import merge
from keras.models import load_model
#from SpatialPyramidPooling import SpatialPyramidPooling
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models  import  model_from_json
from keras.preprocessing import image, sequence
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K


from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage.exposure import rescale_intensity
from skimage.transform import resize
from skimage import io
from skimage.util import crop
from skimage import measure
from skimage.morphology import binary_dilation
from PIL import Image

#import cv2

import pandas as pd
from pandas import read_json

import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import json

from shutil import copyfile


# %% GLOBAL VARIABLES
IMAGE_ROW_N = 224
IMAGE_COL_N = 224

IMAGE_SIZE = (IMAGE_ROW_N, IMAGE_COL_N)


# %% UTILITY FUNCTIONS

# Create dirs
def create_dir(directory, verbose=True):
    try:
        os.mkdir(directory)
        if verbose:
            print(directory+' created.')
    except Exception as e:
        print('Unable to create dir ', e)
    
def create_class_subdir(directory, classes):
    for subdir in classes:
        try:
            os.mkdir(directory+subdir)
        except Exception as e:
            print('Unable to create dir ' + subdir, e)
            

# Recreate file structure in new validation_set subdir and save validation files there
def copy_file_list(source_list, destination_dir):
    for i, img in enumerate(source_list):
        try:
            copyfile(img, destination_dir+img.split('/')[-1])
            if i%100 == 0: print('Processed {} of {}'.format(i, len(source_list)))
        except Exception as e:
            print('Unable to copy file ' + img, e)

def move_file_list(source_list, destination_dir):
    for i, img in enumerate(source_list):
        try:
            move(img, destination_dir+img.split('/')[-1])
            if i%100 == 0: print('Processed {} of {}'.format(i, len(source_list)))
        except Exception as e:
            print('Unable to copy file ' + img, e)

def count_files(directory):
    for d in os.listdir(directory):
        print("Patient '{}' has {} scans".format(d, len(os.listdir(directory + d))))
    print('----')
    n_subdir = len(os.listdir(directory))
    n_files = len(glob.glob(directory+'*/*.dcm'))
    print('Total patients {} Total DCM files {}'.format(n_subdir, n_files) )
    return n_subdir, n_files

# Loading and saving notebook variables
def maybe_pickle_variables(object, variable_name, force=False):
  filename = variable_name + '.pickle'
  if os.path.exists(filename) and not force:
      print(filename + ' already present - Skipping pickling.')
  else:
      print('Pickling ' + filename)
      try:
          with open(filename, 'wb') as file:
              pickle.dump(object, file, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
          print('Unable to save data to', filename, ':', e)
  return filename

def maybe_load_variables(variable_name, encoding='iso-8859-1'):
    data = []
    filename = variable_name + '.pickle'
    if os.path.exists(filename):
        print('Loading ' + filename)
        try:
            with open(filename, 'rb') as file:
                data = pickle.load(file, encoding=encoding)
        except Exception as e:
            print('Unable to load file ', filename, ':', e)
    return data

# Get data
def get_batches(dirname, gen=image.ImageDataGenerator(),
                shuffle=False, batch_size=1,
                class_mode='categorical',
                target_size=IMAGE_SIZE):
    #gen.preprocessing_function=vgg_preprocess
    #gen.rescale=1./255 # not needed for VGG, but Inception would require it
    #gen.featurewise_center=True
    #gen.zca_whitening=True
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)
            
            
# Wrapper around ImageDataGenerator, also returns classes for fitting
def get_data(path, target_size=IMAGE_SIZE, return_type='uint8'):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return (np.concatenate([batches.next() for i in range(batches.nb_sample)]).astype(return_type),
            batches.classes,
            to_categorical(batches.classes),
            batches.filenames
            ) # .astype('float32') optional behind batches.next()

# Save arrays using bcolz for I/O efficiency on image data
def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

# Load arrays using bcolz
def load_array(fname):
    return bcolz.open(fname)[:]


def list_to_enum(input_list, categorical_output=True):
    """
    Turns a list of class names, e.g. ['apples', 'pears', 'oranges', 'pears']
    into a numpy array of numbers, e.g. [0, 1, 2, 1]
    returns array, number_of_classes
    """
    d = {} # Saves mapping class:number
    output = np.zeros(len(input_list)) # Array with integers in place of class names
    num_classes = 0 # Number of classes found
    for i, name in enumerate(input_list): # name is the list item i.e. class name
        if name in d:
            output[i] = d[name]
        else: # Found a new class name not seen before
            d[name] = num_classes
            output[i] = d[name]
            num_classes += 1
    if categorical_output:
        output = to_categorical(output)
    return output, num_classes

# For Submissions
def do_truncate(arr, threshold):
    return np.clip(arr, (1-threshold)/7, threshold)
    
def submission_to_csv(y_test, fname='submission.csv'):
    submission = pd.DataFrame(y_test)
    submission.columns = FISH_CLASSES
    submission.index = [names.split('/')[1] for names in test_filenames]
    submission.index.name = 'image'
    submission.to_csv(WORKING_DIR + '/' + fname)    

# For modeling

def save_model_old(model, path, optional_comment = ''):
    date_string = time.strftime("%Y-%m-%d-%H:%M")
    print(date_string)
    print(optional_comment)
    # Serialize model to disk and save weights
    model_json  =  model.to_json()
    with  open(path+date_string+'_'+optional_comment+".json",  "w")  as  json_file:
        json_file.write(model_json)
    model.save_weights(path+date_string+'_'+optional_comment+".h5")
    print("Saved  model  to  disk")

def save_model(model, path, optional_comment = ''):
    date_string = time.strftime("%Y-%m-%d-%H:%M")
    print(date_string)
    print(optional_comment)
    # Serialize model to disk and save weights
    #model_json  =  model.to_json()
    #with  open(path+date_string+'_'+optional_comment+".json",  "w")  as  json_file:
    #    json_file.write(model_json)
    #model.save_weights(path+date_string+'_'+optional_comment+".h5")
    model.save(path+date_string+'_'+optional_comment+'.h5')
    print("Saved  model  to  disk")

def timestamp(format="%Y-%m-%d-%H:%M"):
    return time.strftime(format)
  
def split_at(model, layer_type):
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers)
                 if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]


# Preprocessing


def preprocess(array, d_mean=0, d_std=1):
    return (array-d_mean)/(d_std*255.)

def deprocess(array, d_mean=0, d_std=1):
    return (array*(d_std*255.))+d_mean

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

# Special Plotting functions
def plot_3d(image, threshold=-300):
    # Plotting CT scans from dicom data
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def plotGrid(image_array, figure_size=(6., 6.), axes_pad=0, inverse_preproc=False, cmap='gray'):
    ## image_array.shape = (observations, height, width, channels)
    grid_size = np.int(np.ceil(np.sqrt(len(image_array))))
    fig = plt.figure(1, figure_size)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(grid_size, grid_size), axes_pad=axes_pad)
    if inverse_preproc:
        for i in range(len(image_array)):
            grid[i].imshow(vgg_preprocess_inverse(image_array[i]), cmap=cmap)
    else:
        for i in range(len(image_array)):
            grid[i].imshow(image_array[i], cmap=cmap)



def show(array, d_mean=0, d_std=1, do_deprocess=True, **kwargs):
    if do_deprocess:
        array = deprocess(array, d_mean, d_std)
    io.imshow(array.astype('uint8'), **kwargs)

# Model selection

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

