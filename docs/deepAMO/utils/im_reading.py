import os, sys
sys.path.append('/mip/opt/lib/python3.6/site-packages/')
sys.path.append('/netscratch/garyli2/cnn_ho/')
import numpy as np
import warnings
import SimpleITK as sitk
import cv2
from scipy import misc
from scipy import ndimage
from utils.imageutils import pad_resize, resize_by_spacing
from PIL import Image
import pickle

from NumpyIm import ArrayFromIm


import matplotlib.pyplot as plt
def load_TriadImage_from_folder(folder_path,new_size, HE=False, Truc=False, Aug=False):
    """loads images in the folder_path and returns a ndarray and threshold the label image"""

    image_list = []
    label_list = []
    #counter = 0
    for image_name in os.listdir(folder_path):

        image_original = ArrayFromIm(folder_path+image_name)

        #need these to have the image in an upright orientation due to wirdness introduced by ArrayFromIm

        image_original = np.swapaxes(image_original,0,2)
        image_original = np.fliplr(image_original)
        image_original = ndimage.rotate(image_original, 90)



        triad = image_original[:,:,:-1]
        # plt.imshow(image)
        # plt.show()
        #print("triad shape,",triad.shape)
        triad = cv2.resize(triad, new_size)

        #activate below to turn on windowing to [0,255]
        #image = ((image - np.amin(image)) * (1 / (np.amax(image) - np.amin(image))) * 255).astype('uint8')

        label = image_original[:,:,-1]
        label_flag = int(np.amax(label))
        label = cv2.resize(label, new_size).astype('uint8')

        #activate below for binary-class segmentation
        super_threshold_indices = label < 1
        label[super_threshold_indices] = 0
        #label = label.ravel()



        if HE == True:
            image = cv2.equalizeHist(triad)
        elif Truc == True:
            clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8,8))
            image_0 = clahe.apply(triad[:,:,0])
            image_1 = clahe.apply(triad[:, :, 1])
            image_2 = clahe.apply(triad[:, :, 2])
            triad = np.stack([image_0,image_1,image_2])
            triad = np.swapaxes(triad, 0, 2)
            #ret, image = cv2.threshold(image,200,255,cv2.THRESH_TRUNC)
        else:
            triad = triad
        # if label_flag == 1:
        #     label_from_categorical_before_reshape = to_categorical(label, 1)
        #
        #     label_from_categorical = to_categorical(label, 1).reshape(1, 9216, 1)
        #     label_from_reshape = label.reshape(1,9216,1)
        #     diff = np.sum(label_from_categorical - label_from_reshape)

#image augmentation method in the FusionNet paper

        if Aug == True and label_flag == 1:
            image_aug_1 = ndimage.rotate(triad, -90)
            image_aug_2 = np.flipud(image_aug_1)
            image_aug_3 = ndimage.rotate(triad, -180)
            image_aug_4 = np.flipud(image_aug_3)
            image_aug_5 = ndimage.rotate(triad, -270)
            image_aug_6 = np.flipud(image_aug_5)
            image_aug_7 = np.flipud(triad)

            label_aug_1 = ndimage.rotate(label, -90)
            label_aug_1 = label_aug_1.astype(int)
            label_aug_2 = np.flipud(label_aug_1)
            label_aug_2 = label_aug_2.astype(int)
            label_aug_3 = ndimage.rotate(label, -180)
            label_aug_3 = label_aug_3.astype(int)
            label_aug_4 = np.flipud(label_aug_3)
            label_aug_4 = label_aug_4.astype(int)
            label_aug_5 = ndimage.rotate(label, -270)
            label_aug_5 = label_aug_5.astype(int)
            label_aug_6 = np.flipud(label_aug_5)
            label_aug_6 = label_aug_6.astype(int)
            label_aug_7 = np.flipud(label)
            label_aug_7 = label_aug_7.astype(int)

            image_list.append(triad)
            image_list.append(image_aug_1)
            image_list.append(image_aug_2)
            image_list.append(image_aug_3)
            image_list.append(image_aug_4)
            image_list.append(image_aug_5)
            image_list.append(image_aug_6)
            image_list.append(image_aug_7)

            label_list.append(label)
            label_list.append(label_aug_1)
            label_list.append(label_aug_2)
            label_list.append(label_aug_3)
            label_list.append(label_aug_4)
            label_list.append(label_aug_5)
            label_list.append(label_aug_6)
            label_list.append(label_aug_7)
        else:
            image_list.append(triad)
            label_list.append(label)

    image_array = np.asarray(image_list)
    label_array = np.asarray(label_list)

    return image_array,label_array


def imageGenerator_from_txt(imageFolder, image_list, label_list, batch_size, HE=False, Truc=False, Aug=True):
    """loads images in the list_train.txt and returns a ndarray and threshold the label image to discrete nunmber of 1-5"""
    images1 = None
    images2 = None
    images3 = None
    images4 = None
    images5 = None
    images6 = None
    images7 = None
    images8 = None
    images9 = None
    labels = None

    while 1:

        for i in range(int(len(image_list) / batch_size)):
            del images1
            del images2
            del images3
            del images4
            del images5
            del images6
            del images7
            del images8
            del images9

            del labels

            images1 = []
            images2 = []
            images3 = []
            images4 = []
            images5 = []
            images6 = []
            images7 = []
            images8 = []
            images9 = []
            labels = []

            for j in range(batch_size):
                if os.path.exists(
                        imageFolder + image_list[i * batch_size + j]):
                    images = ArrayFromIm(imageFolder + image_list[i * batch_size + j])
                    images = np.swapaxes(images,0,2)
                    #print images.shape
                    label = label_list[i * batch_size + j]
                    #print image_list[i * batch_size + j]

                    image1 = images[:, :, 0]
                    image2=  images[:, :, 1]
                    image3 = images[:, :, 2]
                    image4 = images[:, :, 3]
                    image5 = images[:, :, 4]
                    image6 = images[:, :, 5]
                    image7 = images[:, :, 6]
                    image8 = images[:, :, 7]
                    image9 = images[:, :, 8]

                    images1.append(image1)
                    images2.append(image2)
                    images3.append(image3)
                    images4.append(image4)
                    images5.append(image5)
                    images6.append(image6)
                    images7.append(image7)
                    images8.append(image8)
                    images9.append(image9)

                    labels.append(label)

        images1 = np.expand_dims(np.asarray(images1),axis=-1)
        images2 = np.expand_dims(np.asarray(images2),axis=-1)
        images3 = np.expand_dims(np.asarray(images3),axis=-1)
        images4 = np.expand_dims(np.asarray(images4),axis=-1)
        images5 = np.expand_dims(np.asarray(images5),axis=-1)
        images6 = np.expand_dims(np.asarray(images6),axis=-1)
        images7 = np.expand_dims(np.asarray(images7),axis=-1)
        images8 = np.expand_dims(np.asarray(images8),axis=-1)
        images9 = np.expand_dims(np.asarray(images9),axis=-1)


        label_array = np.asarray(labels)

        yield [images1,images2,images3,images4,images5,images6,images7,images8,images9],label_array



def load_image_from_txt(image_directory, input_list,new_size, HE=False, Truc=False, Aug=True):
    """loads images in the list_train.txt and returns a ndarray and threshold the label image to discrete nunmber of 1-5"""
    os.chdir(image_directory)
    image_list_coronal = []
    image_list_saggital = []
    image_list_transaxial = []
    label_list = []

    with open(input_list) as f:
        for line in f:
            base = line.split(' ')[0]
            im_name = os.path.splitext(base)[0] + '.im'
            label = round(int(line.split(' ')[1]))

            image = ArrayFromIm(im_name)
            divider = image.shape[0]/3
            coronal_images = image[0:divider,:,:]
            coronal_images = coronal_images.swapaxes(0, 2)
            saggital_images = image[divider:divider*2,:,:]
            saggital_images = saggital_images.swapaxes(0, 2)
            transaxial_images = image[divider*2:divider * 3, :, :]
            transaxial_images = transaxial_images.swapaxes(0, 2)
            #coronal_images = cv2.resize(coronal_images, new_size)
            #transaxial_images = cv2.resize(transaxial_images, new_size)
            #saggitial_images = cv2.resize(saggitial_images, new_size)
            image_list_transaxial.append(transaxial_images)
            image_list_saggital.append(saggital_images)
            image_list_coronal.append(coronal_images)
            label_list.append(label)

    image_array_coronal = np.asarray(image_list_coronal)
    image_array_saggital = np.asarray(image_list_saggital)
    image_array_transaxial = np.asarray(image_list_transaxial)
    label_array = np.asarray(label_list)

    return image_array_coronal,image_array_saggital,image_array_transaxial,label_array
