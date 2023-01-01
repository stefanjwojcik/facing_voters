# Feature analysis:

using CUDA
using Metalhead
using Flux
using StatsBase
using ScikitLearn
using ProgressMeter
using CSV, DataFrames
using PyCall
using FileIO
using Images

# change working directory to source file location 
cd("src/julia")


# LOAD IMAGES AND RESHAPE 
py"""
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np

def pyload(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    return(img)

def pypreprocess(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return(x)

"""
# create julia equivalent python functions 
pyload(image_path) = py"pyload"(image_path)
pypreprocess(img) = py"pypreprocess"(img)
pysqueeze(predictions) = py"np.squeeze"(predictions)

## RESNET MODEL 

py"""
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
# py"ResNet50(weights='imagenet').summary()" get the names 
_ = ResNet50(weights='imagenet')
nn_model = Model(inputs=_.input, outputs=_.get_layer('avg_pool').output)

"""
resmodel(img) = py"nn_model.predict"(img)


# GETTING THE FEATURES 
training_images = CSV.read("../../data/brazil_public_training_images_names.csv", DataFrame)
cd("../../images/brazil_images_public")
features = @showprogress [pysqueeze(resmodel(pypreprocess(pyload(x)))) for x in training_images.path ]
features = mapreduce(permutedims, vcat, features)
label_m_f(key) = contains(key, r"Fem") ? "Woman" : "Man";
y = label_m_f.(training_images.path);

# THE PIPELINE - (RE)TRAINING THE SVM

#using resnet and the svm already trained (resvm)
@sk_import svm: LinearSVC
@sk_import calibration: CalibratedClassifierCV

resvm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
calsvm = CalibratedClassifierCV(resvm)
calsvm.fit(features, y)
# TEST prediction on one example at a time 
calsvm.predict_proba(reshape(features[1, :], (1, 2048)))


# flow: load images in julia - 
#example_path = "images/brazil_images_public/age46_Masc_joao-correa-psd-d.jpg"
#im = load(example_path);
#rawimg = channelview(im);
# get the number of divisions by 5
function maskiterators(rawimg, kernelsize) # segment with [y:z, p:q]
    h, w = size(rawimg)
    hstart = 1 + Int(round((h % kernelsize) / 2))
    wstart = 1 + Int(round((w % kernelsize) / 2))
    its = Iterators.Stateful(Base.product(wstart:kernelsize:w-kernelsize, hstart:kernelsize:h-kernelsize))
    #hzontal = Iterators.Stateful(wstart:kernelsize:w-kernelsize)
    #vert = Iterators.Stateful(hstart:kernelsize:h-kernelsize)
    return its
end

# add noise to a particular section of the photo 
function noisemask!(rawimg, hzontal, vert, kernelsize)
    rawimg[vert:vert+kernelsize, hzontal:hzontal+kernelsize] .= rand(kernelsize+1, kernelsize+1)
end

# convert and save the noised image 
function convert_and_save(rawimg)
    tempfile = tempname()
    save(tempfile*".png", Gray.(rawimg))
    return(tempfile*".png")
end

########
#rawimg_orig = deepcopy(rawimg)

# Function to calculate relative certainty (distance from .5)
# Takes the prediction vector and returns a vector 
function rel_certainty(preds)
    return map(x -> abs(.5 - x) / .5, preds) # the distance from .5 for all predictions
end

# create masking prediction model:
function mask_analysis(raw_img, kernel_size, calsvm)
    coords = maskiterators(raw_img, kernel_size)
    pred_graph = Tuple[]
    @showprogress for (x,y) in coords
        # copy source image
        imgcopy = deepcopy(raw_img)
        noisemask!(imgcopy, x, y, kernel_size)
        tf = convert_and_save(imgcopy)
        features =  pysqueeze(resmodel(pypreprocess(pyload(tf))))
        preds = calsvm.predict_proba(reshape(features, (1, 2048)))[2] #prob of W
        relpreds = rel_certainty(preds)
        push!(pred_graph, (x,y, relpreds))
    end
    return pred_graph
end

# TESTING 
#basefeat = pysqueeze(resmodel(pypreprocess(pyload(convert_and_save(rawimg_orig)))))
#basepred =  calsvm.predict_proba(reshape(basefeat, (1,2048)))[2]
#preds = mask_analysis(rawimg_orig, 5, calsvm)

# Function to paint most important features of the image 
function paint!(rawimg, preds, kernelsize, threshold)
    channeled_img = channelview(RGB.(Gray.(rawimg)))
    ymax, xmax = size(rawimg)
    for (x,y,z) in preds
        if z <= threshold
            channeled_img[1, y:min(y+kernelsize, ymax), x:min(x+kernelsize, xmax)] .= 1.0
        else 
            channeled_img[1, y:min(y+kernelsize, ymax), x:min(x+kernelsize, xmax)] .= 0.0
        end 
    end
    return colorview(RGB, channeled_img)
end

# Full mega-function
function mask_viz(img, kernelsize, calsvm, quant=.9)
    preds = mask_analysis(img, 5, calsvm)
    thresh = quantile([z for (x,y,z) in preds], quant)
    paint!(img, preds, kernelsize, thresh)
end


# OUTPUT 

## Paint the four representative images 
maf = mask_viz(load("../most_ambiguous_female_alessandra-amatto-d.jpg"), 5, calsvm, .1) 

mam = mask_viz(load("../most_ambiguous_male_acelmo-assuncao-d.jpg"), 5, calsvm, .1) 

mfw = mask_viz(load("../most_feminine_woman_simone-xucra-d.jpg"), 5, calsvm, .1) 

mmm = mask_viz(load("../most_masculine_man_claudio-ocozias-d.jpg"), 5, calsvm, .1) 