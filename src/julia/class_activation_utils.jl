using CUDA
using Flux
using StatsBase
using ProgressMeter
using CSV, DataFrames
using PyCall
using FileIO
using Images, ImageView

# change working directory to source file location 
#cd(@__DIR__)

py"""
import keras.backend as K
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
import scipy
# py"ResNet50(weights='imagenet').summary()" get the names 
resmodel = ResNet50(weights='imagenet')
def get_resnet_embedding_model(model):
    Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)

"""
resmodel = py"resmodel"

# LOAD IMAGES AND RESHAPE 
py"""

def pyload(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    return(img)

def pypreprocess(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return(x)

def get_img(impath):
    '''
    this function loads and preprocesses an image
    
    usage:
        basepath = os.getcwd()+"/images/brazil_images_public"
        impath = os.listdir(basepath)[1]
        get_img(basepath+"/"+impath)
    '''
    img = pyload(impath)
    img = pypreprocess(img)
    return(img)
"""
# Define two julia utility functions 
get_example(path) = py"get_img"(path * StatsBase.sample(readdir("images/brazil_images_public")))
get_img(path) = py"get_img"(path)
# img = get_example("images/brazil_images_public")

## Class activation maps 
# This is a modified version of the activation maps
# This grabs the embedding containing maximum weight for every class 
# The interpretation would be the most important features for each class
# In other words, the features with the most information 
py"""
def get_class_activation_map(model, img):
    '''
    this function computes the class activation map
    
    Inputs:
        1) model (tensorflow model) : trained model
        2) img (numpy array of shape (224, 224, 3)) : input image
    '''
    # Get the 2048 input weights to the softmax of the winning class.
    nn_model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    class_weights_winner = nn_model.predict(img)[0]
    
    # get the final conv layer
    final_conv_layer = model.get_layer("conv5_block3_out")
    
    # create a function to fetch the final conv layer output maps (should be shape (1, 7, 7, 2048)) 
    get_output = K.function([model.layers[0].input],[final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    
    # squeeze conv map to shape image to size (7, 7, 2048)
    conv_outputs = np.squeeze(conv_outputs)
    
    # bilinear upsampling to resize each filtered image to size of original image 
    mat_for_mult = scipy.ndimage.zoom(conv_outputs, (32, 32, 1), order=1) # dim: 224 x 224 x 2048
    
    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((224*224, 2048)), class_weights_winner).reshape(224,224) # dim: 224 x 224
    
    # return class activation map
    return final_output
"""
# julia function to call class activation mapping 
get_class_activation_map(model, img) = py"get_class_activation_map"(model, img)

# Class Activation Paint Images - feed class activation into an image 
function cam_paint(img, cam)
    # channelview of image 
    channeled_img = channelview(RGB.(Gray.(img)))
    # cam reshaped
    cam_reshaped = imresize(cam, size(channeled_img)[2:3])
    channeled_img[1, :, :] .= 0.0
    channeled_img[1, :, :] = 1.0 .* cam_reshaped .> median(cam_reshaped)
    colorview(RGB, channeled_img)
end

# Class Activation Paint Images - feed class activation into an image 
function cam_heat(img, cam)
    # channelview of image 
    channeled_img = channelview(Gray.(img))
    # cam reshaped
    cam_reshaped = imresize(cam, size(channeled_img))
    out = (channeled_img .+ cam_reshaped/max(cam...))'
    return out
end

############## END CODE HERE 

### EXAMPLES 

#img = load("../../images/most_ambiguous_female_alessandra-amatto-d.jpg")
