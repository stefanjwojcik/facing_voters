#################
### Transfer learning with ResNet
#########################

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
using ScikitLearn.CrossValidation: cross_val_score
@sk_import svm: LinearSVC

include("src/julia/utils.jl")
impath = "images/official_images_data/"

df = CSV.read("data/official_local_election_data_2016.csv", DataFrame);
df = filter(r -> any(startswith.(["PREFEITO", "VEREADOR"], r.DS_CARGO)), df)

# puts together info for each candidate to create the path of the image
df.imglink = map(generate_img_link, eachrow(df))

# be sure to CD into the data folder 
masc_training = CSV.read("data/masculino_training_sample.csv", DataFrame)
fem_training = CSV.read("data/feminino_training_sample.csv", DataFrame)
training = [masc_training; fem_training]
# Generate labels 
y = [repeat(["Man"], nrow(masc_training)); repeat(["Woman"], nrow(fem_training))]

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

#########################
# RESNET - MAIN RESULTS 
###################################
py"""
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.vgg19 import preprocess_input
#from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
# py"ResNet50(weights='imagenet').summary()" get the names 
_ = ResNet50(weights='imagenet')
nn_model = Model(inputs=_.input, outputs=_.get_layer('avg_pool').output)

"""
resmodel(img) = py"nn_model.predict"(img)

###### TEST 

example_path = "images/official_images_data/"*readdir("images/official_images_data/")[1]
pysqueeze(resmodel(pypreprocess(pyload(example_path))))

######### ITERATE AND PRODUCE PREDICTIONS  --------------
#cd("images/official_images_data")
std_features = @showprogress [pysqueeze(resmodel(pypreprocess(pyload(impath * x)))) for x in training.path ]
std_features = mapreduce(permutedims, vcat, std_features)

###### RESNET MODEL W/ SVM

## LOADING SCIKITLEARN 

resvm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
cv = ScikitLearn.CrossValidation.KFold(5000, n_folds=5, random_state = 3304, shuffle=true)
out = cross_val_score(resvm, std_features, y, cv = cv)
############################################
#                    VALIDATION 
############################################

######################### 
# RESNET DIFFERENT LAYER 
##################

py"""
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
# py"ResNet50(weights='imagenet').summary()" get the names 
_ = ResNet50(weights='imagenet')
nn_model_conv = Model(inputs=_.input, outputs=_.get_layer('conv5_block3_add').output)

"""
resmodelconv(img) = py"nn_model_conv.predict"(img)

# Test example: 
pysqueeze(resmodelconv(pypreprocess(pyload(impath * readdir(impath)[1]))))

# now, will take the mean of the 7*7 convolution from this layer 

######### ITERATE AND PRODUCE PREDICTIONS  --------------
feature_process(path)  = mean(pysqueeze(resmodelconv(pypreprocess(pyload(path)))), dims=(1,2))
features = @showprogress [vcat(feature_process(impath * x)...) for x in training.path ]
features = mapreduce(permutedims, vcat, features)

###### CONV RESNET MODEL W/ SVM - results 

## LOADING SCIKITLEARN 
import ScikitLearn: CrossValidation
svm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)

# accuracy is atrocious, why is it sooo bad? when done in python, it is fine 
cv = ScikitLearn.CrossValidation.KFold(5000, n_folds=5, random_state = 3304, shuffle=true)
resnet_conv_out = cross_val_score(svm, features, y, cv = cv)


######################### 
# DIFFERENT MODEL - VGG19
##################

py"""
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
# py"VGG19(weights='imagenet').summary()" get the names 
_ = VGG19(weights='imagenet')
vgg_model = Model(inputs=_.input, outputs=_.get_layer('fc2').output)
"""
vggmodel(img) = py"vgg_model.predict"(img)

# Test example: 
pysqueeze(vggmodel(pypreprocess(pyload(impath * readdir(impath)[1]))))

# now, will take the mean of the 7*7 convolution from this layer 

######### ITERATE AND PRODUCE PREDICTIONS  --------------
feature_process(path)  = pysqueeze(vggmodel(pypreprocess(pyload(path))))
features = @showprogress [feature_process(impath * x) for x in training.path ]
features = mapreduce(permutedims, vcat, features)

## LOADING SCIKITLEARN 
import ScikitLearn: CrossValidation
svm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)

# accuracy is atrocious, why is it sooo bad? when done in python, it is fine 
cv = ScikitLearn.CrossValidation.KFold(5000, n_folds=5, random_state = 3304, shuffle=true)
vgg_out = cross_val_score(svm, features, y, cv = cv)

############################################
##### FEATURE EXPLORATION WITH THE EXAMPLE DATA 
####################################################

# flow: load images in julia - 
example_path = impath*readdir(impath)[1]
im = load(example_path);
rawimg = channelview(im);
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

########
rawimg_orig = deepcopy(rawimg)

# THE PIPELINE - (RE)TRAINING THE SVM

#using resnet and the svm already trained (resvm)
resmodel(img) = py"nn_model.predict"(img)
@sk_import calibration: CalibratedClassifierCV
calsvm = CalibratedClassifierCV(resvm)
calsvm.fit(std_features, y)
calsvm.predict_proba(std_features)
# TEST prediction on one example at a time 
calsvm.predict_proba(reshape(std_features[1, :], (1, 2048)))

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
        out = calsvm.predict_proba(reshape(features, (1, 2048)))[2] #prob of W
        push!(pred_graph, (x,y, out))
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
        if z >= threshold
            channeled_img[1, y:min(y+kernelsize, ymax), x:min(x+kernelsize, xmax)] .= 1.0
        else 
            channeled_img[1, y:min(y+kernelsize, ymax), x:min(x+kernelsize, xmax)] .= 0.0
        end 
    end
    return colorview(RGB, channeled_img)
end

# Full mega-function
function mask_viz(img, kernelsize, calsvm, quant=.9)
    preds = mask_analysis(rawimg_orig, 5, calsvm)
    thresh = quantile([z for (x,y,z) in preds], quant)
    paint!(img, preds, kernelsize, thresh)
end
# Test
#paint!(rawimg, preds, 5, .001)

## Paint the four representative images 

#load(df.imglink[findall(x -> contains.(x, r"CLAUDIO OCOZIAS"), df.NM_URNA_CANDIDATO)][1])

# Alessandra Amatto
maf = mask_viz(load("images/official_images_data/MG-53970-130000012523-128169510264.jpg"), 5, calsvm, .9) 
# Acelmo Assuncao
mam = mask_viz(load("images/official_images_data/MG-48950-130000075502-197497290221.jpg"), 5, calsvm, .9) 
# Simone Xucra
mfw = mask_viz(load("images/official_images_data/MS-90034-120000005513-018097691937.jpg"), 5, calsvm, .9) 
# Claudio Ocozias
mmm = mask_viz(load("images/official_images_data/GO-93718-90000012258-014291981031.jpg"), 5, calsvm, .9) 