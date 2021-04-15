# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:03:38 2021

@author: studperadh6230
"""

import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import glob
import cv2
import ktrain
from keras.optimizers import Adam
import keras
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D,Input,MaxPool2D
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import randint


# Preprocessing parameters
RESCALE = 1.0 / 255
SHAPE = (256, 256)
PREPROCESSING_FUNCTION = None
PREPROCESSING = None
VMIN = 0.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN

# Finetuning parameters
FINETUNE_SPLIT = 0.1
STEP_MIN_AREA = 5
START_MIN_AREA = 5
STOP_MIN_AREA = 1005

# Data augmentation parameters (only for training)
ROT_ANGLE = randint(0,180)
W_SHIFT_RANGE = 0.05
H_SHIFT_RANGE = 0.05
FILL_MODE = "nearest"
BRIGHTNESS_RANGE = [0.95, 1.05]
VAL_SPLIT = 0.2

# Learning Rate Finder parameters
START_LR = 1e-5
LR_MAX_EPOCHS = 10
LRF_DECREASE_FACTOR = 0.85

# Training parameters
EARLY_STOPPING = 12
REDUCE_ON_PLATEAU = 6

# Finetuning parameters
FINETUNE_SPLIT = 0.1
STEP_MIN_AREA = 5
START_MIN_AREA = 5
STOP_MIN_AREA = 1005


def build_model(color_mode):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3

    # define model
    input_img = keras.layers.Input(shape=(*SHAPE, channels))
    # Encode-----------------------------------------------------------
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(
        input_img
    )
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(128, (4, 4), strides=2, activation="relu", padding="same")(
        x
    )
    x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    encoded = keras.layers.Conv2D(1, (8, 8), strides=1, padding="same")(x)

    # Decode---------------------------------------------------------------------
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(
        encoded
    )
    x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (4, 4), strides=2, activation="relu", padding="same")(
        x
    )
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((4, 4))(x)
    x = keras.layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (8, 8), activation="relu", padding="same")(x)

    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(
        channels, (8, 8), activation="sigmoid", padding="same"
    )(x)

    model = keras.models.Model(input_img, decoded)

    return model




#losses 


def ssim_loss(dynamic_range):
    def loss(imgs_true,imgs_pred):
        
        return 1 - tf.image.ssim(imgs_true,imgs_pred,dynamic_range)
    
    return loss

def mssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):
        
        return 1 - tf.image.ssim_multiscale(imgs_true, imgs_pred,dynamic_range)
    
    return loss

def l2_loss(imgs_true,imgs_pred):
    
    return tf.nn.l2_loss(imgs_true-imgs_pred)

loss_func = ssim_loss(1.0-0)
#metrics

def ssim_metric(dynamic_range):
    def ssim(imgs_true,imgs_pred):
        
        return K.mean(tf.image.ssim(imgs_true, imgs_pred,dynamic_range),axis =-1)
    
    return ssim

def mssim_metric(dynamic_range):
    def mssim(imgs_true,imgs_pred):
        
        return K.mean(tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range),axis=-1)
    
    return mssim


metric = ssim_metric(1.-0.)

#setting loss function

color_mode = 'rgb'

#compiling the model
model = build_model(color_mode)
model.summary()
optimizer = Adam(learning_rate = 1e-5,beta_1 = 0.9,beta_2 = 0.999)

model.compile(loss = loss_func,optimizer = optimizer,metrics = metric)


#data preprocessing
#seperate functions for training data,validation, test and finetuning

def get_train_generator(batch_size, shuffle=True):
        train_dir ='D:/anomaly_detection/train/'
        # This will do preprocessing and realtime data augmentation:
        train_datagen = ImageDataGenerator(
            # standarize input
            featurewise_center=False,
            featurewise_std_normalization=False,
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=ROT_ANGLE,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=W_SHIFT_RANGE,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=H_SHIFT_RANGE,
            # set mode for filling points outside the input boundaries
            fill_mode=FILL_MODE,
            # value used for fill_mode = "constant"
            cval=0.0,
            horizontal_flip=True,
            # randomly change brightness (darker < 1 < brighter)
            brightness_range=BRIGHTNESS_RANGE,
            # set rescaling factor (applied before any other transformation)
            rescale=1./255,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format="channels_last",
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=VAL_SPLIT,
        )

        # Generate training batches with datagen.flow_from_directory()
        train_generator = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=(256,256),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return train_generator

def get_val_generator(batch_size, shuffle=True):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size.
        For test, pass nb_test_images as batch size.
        """
        # For validation dataset, only rescaling
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            data_format="channels_last",
            validation_split=VAL_SPLIT
        )
        # Generate validation batches with datagen.flow_from_directory()
        validation_generator = validation_datagen.flow_from_directory(
            directory='D:/anomaly_detection/train/',
            target_size=(256,256),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode="input",
            subset="validation",
            shuffle=shuffle,
        )
        return validation_generator


def get_test_generator(batch_size, shuffle=False):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size.
        For test, pass nb_test_images as batch size.
        """
        # For test dataset, only rescaling
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            data_format="channels_last",
            preprocessing_function=None,
        )

        # Generate validation batches with datagen.flow_from_directory()
        test_generator = test_datagen.flow_from_directory(
            directory='D:/anomaly_detection/test/',
            target_size=(256,256),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return test_generator

def get_finetuning_generator(batch_size, shuffle=False):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size.
        For test, pass nb_test_images as batch size.
        """
        # For test dataset, only rescaling
        test_data_dir = 'D:/anomaly_detection/test/'
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            data_format="channels_last",
            preprocessing_function=None,
        )

        # Generate validation batches with datagen.flow_from_directory()
        finetuning_generator = test_datagen.flow_from_directory(
            directory=test_data_dir,
            target_size=(256,256),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return finetuning_generator
    




X_train = []

for img in glob.glob('D:/hazelnut/train/good/*png'):
    n= cv2.imread(img,1)
    n = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
    n = cv2.resize(n,(256,256),interpolation=cv2.INTER_AREA)
    X_train.append(n)

X_train = np.asarray(X_train).astype(np.float32)

X_test = []

for img in glob.glob('D:/hazelnut/train/val/*png'):
    n= cv2.imread(img,1)
    n = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
    n = cv2.resize(n,(256,256),interpolation=cv2.INTER_AREA)
    X_test.append(n)

X_test = np.asarray(X_test).astype(np.float32)

X_pred = []

for img in glob.glob('D:/hazelnut/test/crack/*png'):
    n= cv2.imread(img,1)
    n = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
    n = cv2.resize(n,(256,256),interpolation=cv2.INTER_AREA)
    X_pred.append(n)

X_pred = np.asarray(X_pred).astype(np.float32)


from skimage.metrics import structural_similarity
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.util import img_as_ubyte

# Segmentation Parameters
# float + SSIM
THRESH_MIN_FLOAT_SSIM = 0.10
THRESH_STEP_FLOAT_SSIM = 0.002
# float + L2
THRESH_MIN_FLOAT_L2 = 0.005
THRESH_STEP_FLOAT_L2 = 0.0005
# uint8 + SSIM
THRESH_MIN_UINT8_SSIM = 20
THRESH_STEP_UINT8_SSIM = 1
# uint8 + L2 (generally uneffective combination)
THRESH_MIN_UINT8_L2 = 5
THRESH_STEP_UINT8_L2 = 1


class TensorImages:
    def __init__(
        self,
        imgs_input,
        imgs_pred,
        vmin,
        vmax,
        method,
        dtype="float64",
        filenames=None,
    ):
        assert imgs_input.ndim == imgs_pred.ndim == 4
        assert dtype in ["float64", "uint8"]
        assert method in ["l2", "ssim", "mssim"]
        self.method = method
        self.dtype = dtype
        # pixel min and max values of input and reconstruction (pred)
        # depend on preprocessing function, which in turn depends on
        # the model used for training.
        self.vmin = vmin
        self.vmax = vmax
        self.filenames = filenames

        # if grayscale, reduce dim to (samples x length x width)
        if imgs_input.shape[-1] == 1:
            imgs_input = imgs_input[:, :, :, 0]
            imgs_pred = imgs_pred[:, :, :, 0]
            self.cmap = "gray"
        # if RGB
        else:
            self.cmap = None

        # compute resmaps
        self.imgs_input = imgs_input
        self.imgs_pred = imgs_pred
        self.scores, self.resmaps = calculate_resmaps(
            self.imgs_input, self.imgs_pred, method, dtype
        )

        # compute maximal threshold based on resmaps
        self.thresh_max = np.amax(self.resmaps)

        # set parameters for future segmentation of resmaps
        if dtype == "float64":
            self.vmin_resmap = 0.0
            self.vmax_resmap = 1.0
            if method in ["ssim", "mssim"]:
                self.thresh_min = THRESH_MIN_FLOAT_SSIM
                self.thresh_step = THRESH_STEP_FLOAT_SSIM
            elif method == "l2":
                self.thresh_min = THRESH_MIN_FLOAT_L2
                self.thresh_step = THRESH_STEP_FLOAT_L2

        elif dtype == "uint8":
            self.vmin_resmap = 0
            self.vmax_resmap = 255
            if method in ["ssim", "mssim"]:
                self.thresh_min = THRESH_MIN_UINT8_SSIM
                self.thresh_step = THRESH_STEP_UINT8_SSIM
            elif method == "l2":
                self.thresh_min = THRESH_MIN_UINT8_L2
                self.thresh_step = THRESH_STEP_UINT8_L2

    def generate_inspection_plots(self, group, save_dir=None):
        assert group in ["validation", "test"]

        l = len(self.filenames)

        for i in range(len(self.imgs_input)):
            self.plot_input_pred_resmap(index=i, group=group, save_dir=save_dir)


    ### plottings methods for inspection

    def plot_input_pred_resmap(self, index, group, save_dir=None):
        assert group in ["validation", "test"]
        fig, axarr = plt.subplots(3, 1)
        fig.set_size_inches((4, 9))

        axarr[0].imshow(
            self.imgs_input[index], cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
        )
        axarr[0].set_title("input")
        axarr[0].set_axis_off()
        # fig.colorbar(im00, ax=axarr[0])

        axarr[1].imshow(
            self.imgs_pred[index], cmap=self.cmap, vmin=self.vmin, vmax=self.vmax
        )
        axarr[1].set_title("pred")
        axarr[1].set_axis_off()
        # fig.colorbar(im10, ax=axarr[1])

        im20 = axarr[2].imshow(
            self.resmaps[index],
            cmap="inferno",
            vmin=self.vmin_resmap,
            vmax=self.vmax_resmap,
       )
        axarr[2].set_title("resmap_"+ self.method+ "_"+ self.dtype+ "\n{}_".format(self.method)+ f"score = {self.scores[index]:.2E}")
        axarr[2].set_axis_off()
        fig.colorbar(im20, ax=axarr[2])

        plt.suptitle(group.upper() + "\n" + self.filenames[index])

        if save_dir is not None:
            plot_name = get_plot_name(self.filenames[index], suffix="inspection")
            fig.savefig(os.path.join(save_dir, plot_name))
            plt.close(fig=fig)
        return

    def plot_image(self, plot_type, index):
        assert plot_type in ["input", "pred", "resmap"]
        # select image to plot
        if plot_type == "input":
            image = self.imgs_input[index]
            cmap = self.cmap
            vmin = self.vmin
            vmax = self.vmax
        elif plot_type == "pred":
            image = self.imgs_pred[index]
            cmap = self.cmap
            vmin = self.vmin
            vmax = self.vmax
        elif plot_type == "resmap":
            image = self.resmaps[index]
            cmap = "inferno"
            vmin = self.vmin_resmap
            vmax = self.vmax_resmap
            
        # plot image
        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        fig.colorbar(im)
        title = plot_type + "\n" + self.filenames[index]
        plt.title(title)
        plt.show()
        return


def get_plot_name(filename, suffix):
    filename_new, ext = os.path.splitext(filename)
    filename_new = "_".join(filename_new.split("/")) + "_" + suffix + ext
    return filename_new


#### Image Processing Functions

## Functions for generating Resmaps


def calculate_resmaps(imgs_input, imgs_pred, method, dtype="float64"):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """
    # if RGB, transform to grayscale and reduce tensor dimension to 3
    if imgs_input.ndim == 4 and imgs_input.shape[-1] == 3:
        imgs_input_gray = tf.image.rgb_to_grayscale(imgs_input).numpy()[:, :, :, 0]
        imgs_pred_gray = tf.image.rgb_to_grayscale(imgs_pred).numpy()[:, :, :, 0]
    else:
        imgs_input_gray = imgs_input
        imgs_pred_gray = imgs_pred

    # calculate remaps
    if method == "l2":
        scores, resmaps = resmaps_l2(imgs_input_gray, imgs_pred_gray)
    elif method in ["ssim", "mssim"]:
        scores, resmaps = resmaps_ssim(imgs_input_gray, imgs_pred_gray)
    if dtype == "uint8":
        resmaps = img_as_ubyte(resmaps)
    return scores, resmaps


def resmaps_ssim(imgs_input, imgs_pred):
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    scores = []
    for index in range(len(imgs_input)):
        img_input = imgs_input[index]
        img_pred = imgs_pred[index]
        score, resmap = structural_similarity(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            sigma=1.5,
            full=True,
        )
        # resmap = np.expand_dims(resmap, axis=-1)
        resmaps[index] = 1 - resmap
        scores.append(score)
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    return scores, resmaps


def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2
    scores = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return scores, resmaps


## functions for processing resmaps


def label_images(images_th):
    """
    Segments images into images of connected components (regions).
    Returns segmented images and a list of lists, where each list 
    contains the areas of the regions of the corresponding image. 
    
    Parameters
    ----------
    images_th : array of binary images
        Thresholded residual maps.
    Returns
    -------
    images_labeled : array of labeled images
        Labeled images.
    areas_all : list of lists
        List of lists, where each list contains the areas of the regions of the corresponding image.
    """
    images_labeled = np.zeros(shape=images_th.shape)
    areas_all = []
    for i, image_th in enumerate(images_th):
        # close small holes with binary closing
        # bw = closing(image_th, square(3))

        # remove artifacts connected to image border
        cleared = clear_border(image_th)

        # label image regions
        image_labeled = label(cleared)

        # image_labeled = label(image_th)

        # append image
        images_labeled[i] = image_labeled

        # compute areas of anomalous regions in the current image
        regions = regionprops(image_labeled)

        if regions:
            areas = [region.area for region in regions]
            areas_all.append(areas)
        else:
            areas_all.append([0])

    return images_labeled, areas_all

def calculate_largest_areas(resmaps, thresholds):

    # initialize largest areas to an empty list
    largest_areas = []


    for index, threshold in enumerate(thresholds):
        # segment (threshold) residual maps
        resmaps_th = resmaps > threshold

        # compute labeled connected components
        _, areas_th = label_images(resmaps_th)

        # retieve largest area of all resmaps for current threshold
        areas_th_total = [item for sublist in areas_th for item in sublist]
        largest_area = np.amax(np.array(areas_th_total))
        largest_areas.append(largest_area)

        # print progress bar

    return largest_areas

def get_total_number_train_images():
        total_number = 0
        train_data_dir = 'D:/anomaly_detection/train/'
        sub_dir_names = os.listdir(train_data_dir)
        for sub_dir_name in sub_dir_names:
            sub_dir_path = os.path.join(train_data_dir, sub_dir_name)
            filenames = os.listdir(sub_dir_path)
            number = len(filenames)
            total_number = total_number + number
        return total_number

nb_train_images = get_total_number_train_images()
train_generator = get_train_generator(batch_size = 32,shuffle =True)


nb_validation_images = int((0.2*nb_train_images))


validation_generator = get_val_generator(
        batch_size=nb_validation_images, shuffle=False)

    # retrieve preprocessed validation images from generator
imgs_val_input = validation_generator.next()[0]

    # retrieve validation image_names
filenames_val = validation_generator.filenames
#training

hist = model.fit(train_generator, epochs = 10,batch_size = 32,shuffle = True,validation_data = validation_generator)

    # reconstruct (i.e predict) validation images
imgs_val_pred = model.predict(imgs_val_input)


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#testing

def generate_new_name(filename, suffix):
    filename_new, ext = os.path.splitext(filename)
    filename_new = "_".join(filename_new.split("/")) + "_" + suffix + ext
    return filename_new

def get_true_classes(filenames):
    # retrieve ground truth
    y_true = [1 if "good" not in filename.split("/") else 0 for filename in filenames]
    return y_true


def is_defective(areas, min_area):
    """Decides if image is defective given the areas of its connected components"""
    areas = np.array(areas)
    if areas[areas >= min_area].shape[0] > 0:
        return 1
    return 0


def predict_classes(resmaps, min_area, threshold):
    # threshold residual maps with the given threshold
    resmaps_th = resmaps > threshold
    # compute connected components
    _, areas_all = label_images(resmaps_th)
    # Decides if images are defective given the areas of their connected components
    y_pred = [is_defective(areas, min_area) for areas in areas_all]
    return y_pred


def save_segmented_images(resmaps, threshold, filenames, save_dir):
    # threshold residual maps with the given threshold
    resmaps_th = resmaps > threshold
    # create directory to save segmented resmaps
    seg_dir = os.path.join(save_dir, "segmentation")
    if not os.path.isdir(seg_dir):
        os.makedirs(seg_dir)
    # save segmented resmaps
    for i, resmap_th in enumerate(resmaps_th):
        fname = generate_new_name(filenames[i], suffix="seg")
        fpath = os.path.join(seg_dir, fname)
        plt.imsave(fpath, resmap_th, cmap='gray')
    return


def get_total_number_train_images():
        total_number = 0
        train_data_dir = 'D:/anomaly_detection/train/good/'
        sub_dir_names = os.listdir(train_data_dir)
        for sub_dir_name in sub_dir_names:
            sub_dir_path = os.path.join(train_data_dir, sub_dir_name)
            filenames = os.listdir(sub_dir_path)
            number = len(filenames)
            total_number = total_number + number
        return total_number


method = 'ssim'

    # instantiate TensorImages object to compute validation resmaps
tensor_val = TensorImages(
        imgs_input=imgs_val_input,
        imgs_pred=imgs_val_pred,
        vmin=VMIN,
        vmax=VMAX,
        method=method,
        dtype='uint8',
        filenames=filenames_val,
    )


def get_total_number_test_images():
        total_number = 0
        test_data_dir = 'D:/hazelnut/test'
        sub_dir_names = os.listdir(test_data_dir)
        for sub_dir_name in sub_dir_names:
            sub_dir_path = os.path.join(test_data_dir, sub_dir_name)
            filenames = os.listdir(sub_dir_path)
            number = len(filenames)
            total_number = total_number + number
        return total_number

nb_test_images = get_total_number_test_images()

finetuning_generator = get_finetuning_generator(
        batch_size=nb_test_images, shuffle=False)

    # retrieve preprocessed test images from generator
imgs_test_input = finetuning_generator.next()[0]
filenames_test = finetuning_generator.filenames

#finetuning datasplit

assert "good" in finetuning_generator.class_indices
index_array = finetuning_generator.index_array
classes = finetuning_generator.classes
_, index_array_ft, _, classes_ft = train_test_split(
        index_array,
        classes,
        test_size=FINETUNE_SPLIT,
        random_state=42,
        stratify=classes,
    )

    # get correct classes corresponding to selected images
good_class_i = finetuning_generator.class_indices["good"]
y_ft_true = np.array(
        [0 if class_i == good_class_i else 1 for class_i in classes_ft])

    # select test images for finetuninig
imgs_ft_input = imgs_test_input[index_array_ft]
filenames_ft = list(np.array(filenames_test)[index_array_ft])

    # reconstruct (i.e predict) finetuning images
imgs_ft_pred = model.predict(imgs_ft_input)

    # instantiate TensorImages object to compute finetuning resmaps
tensor_ft = TensorImages(
        imgs_input=imgs_ft_input,
        imgs_pred=imgs_ft_pred,
        vmin=VMIN,
        vmax=VMAX,
        method='ssim',
        dtype='uint8',
        filenames=filenames_ft,
    )


#compute thersholds

    # initialize finetuning dictionary
dict_finetune = {
        "min_area": [],
        "threshold": [],
        "TPR": [],
        "TNR": [],
        "FPR": [],
        "FNR": [],
        "score": [],
    }

    # initialize discrete min_area values
min_areas = np.arange(
        start=START_MIN_AREA,
        stop=STOP_MIN_AREA,
        step=STEP_MIN_AREA,
    )

    # initialize thresholds
thresholds = np.arange(
        start=tensor_val.thresh_min,
        stop=tensor_val.thresh_max + tensor_val.thresh_step,
        step=tensor_val.thresh_step,
    )

    # compute largest anomaly areas in resmaps for increasing thresholds
print("step 1/2: computing largest anomaly areas for increasing thresholds...")
largest_areas = calculate_largest_areas(
        resmaps=tensor_val.resmaps, thresholds=thresholds,
    )

    # select best minimum area and threshold pair to use for testing
print("step 2/2: selecting best minimum area and threshold pair for testing...")


for i, min_area in enumerate(min_areas):
        # compare current min_area with the largest area
        for index, largest_area in enumerate(largest_areas):
            if min_area > largest_area:
                break

        # select threshold corresponding to current min_area
        threshold = thresholds[index]

        # apply the min_area, threshold pair to finetuning images
        y_ft_pred = predict_classes(
            resmaps=tensor_ft.resmaps, min_area=min_area, threshold=threshold
        )

        # confusion matrix
        tnr, fpr, fnr, tpr = confusion_matrix(
            y_ft_true, y_ft_pred, normalize="true"
        ).ravel()

        # record current results
        dict_finetune["min_area"].append(min_area)
        dict_finetune["threshold"].append(threshold)
        dict_finetune["TPR"].append(tpr)
        dict_finetune["TNR"].append(tnr)
        dict_finetune["FPR"].append(fpr)
        dict_finetune["FNR"].append(fnr)
        dict_finetune["score"].append((tpr + tnr) / 2)



    # get min_area, threshold pair corresponding to best score
max_score_i = np.argmax(dict_finetune["score"])
max_score = float(dict_finetune["score"][max_score_i])
best_min_area = int(dict_finetune["min_area"][max_score_i])
best_threshold = float(dict_finetune["threshold"][max_score_i])




        # initialize preprocessor

def get_test_generator(batch_size, shuffle=False):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size.
        For test, pass nb_test_images as batch size.
        """
        # For test dataset, only rescaling
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            data_format="channels_last")
        
        test_generator = test_datagen.flow_from_directory(
            directory='D:/anomaly_detection/test/',
            target_size=(256,256),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return test_generator

        
        
        # get test generator
nb_test_images = get_total_number_test_images()
test_generator = get_test_generator(batch_size=nb_test_images, shuffle=False)

        # retrieve test images from generator
imgs_test_input = test_generator.next()[0]

        # retrieve test image names
filenames = test_generator.filenames

        # predict on test images
imgs_test_pred = model.predict(imgs_test_input)

        # instantiate TensorImages object
tensor_test = TensorImages(
            imgs_input=imgs_test_input,
            imgs_pred=imgs_test_pred,
            vmin=VMIN,
            vmax=VMAX,
            method='ssim',
            dtype='uint8',
            filenames=filenames,
        )

#  CLASSIFICATION 

        # retrieve ground truth
y_true = get_true_classes(filenames)

        # predict classes on test images
y_pred = predict_classes(
            resmaps=tensor_test.resmaps, min_area=min_area, threshold=threshold
        )

        # confusion matrix
tnr, fp, fn, tpr = confusion_matrix(y_true, y_pred, normalize="true").ravel()

        # initialize dictionary to store test results
test_result = {
            "min_area": min_area,
            "threshold": threshold,
            "TPR": tpr,
            "TNR": tnr,
            "score": (tpr + tnr) / 2,
            "method": 'ssim',
            "dtype": 'uint8',
        }

model_dir_name = 'D:/'
input_directory = 'D:/'

loss = 'ssim'
subdir = 'crack'

save_dir = os.path.join(
            os.getcwd(),
            "results",
            input_directory,
            'test',
            model_dir_name,
            subdir,
        )



#save_segmented_images(tensor_test.resmaps, threshold, filenames, save_dir)

plt.imshow(tensor_test.resmaps[57])
cv2.imwrite('D:/hazelnut/ss.bmp',tensor_test.resmaps[35])

for i in range(0,70):
    im = tensor_test.resmaps[i]
    im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    #cv2.imwrite('D:/anomaly_detection/seg/'+'s'+str(i)+'.bmp',im)
    #plt.imshow(im,cmap=None)
    plt.imsave('D:/anomaly_detection/seg/p'+str(i)+'.png',tensor_test.resmaps[i])
    
    

