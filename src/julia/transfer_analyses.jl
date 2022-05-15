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

# CD into the images folder 
training_images = CSV.read("images/brazil_public_training_images_names.csv", DataFrame)

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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
# py"ResNet50(weights='imagenet').summary()" get the names 
_ = ResNet50(weights='imagenet')
nn_model = Model(inputs=_.input, outputs=_.get_layer('avg_pool').output)

"""
resmodel(img) = py"nn_model.predict"(img)

###### TEST 

example_path = "images/brazil_images_public/age46_Masc_joao-correa-psd-d.jpg"
pysqueeze(resmodel(pypreprocess(pyload(example_path))))

######### ITERATE AND PRODUCE PREDICTIONS  --------------
cd("images/brazil_images_public")
features = @showprogress [pysqueeze(resmodel(pypreprocess(pyload(x)))) for x in training_images.path ]
features = mapreduce(permutedims, vcat, features)
label_m_f(key) = contains(key, r"Fem") ? "Woman" : "Man";
y = label_m_f.(training_images.path);

###### RESNET MODEL W/ SVM

## LOADING SCIKITLEARN 
import ScikitLearn: CrossValidation
@sk_import svm: LinearSVC
import ScikitLearn: CrossValidation
resvm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)

# accuracy is atrocious, why is it sooo bad? when done in python, it is fine 
RSK = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=3403)
out = cross_val_score(resvm, features, y, cv = RSK.split(features,  y))

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
pysqueeze(resmodelconv(pypreprocess(pyload("age46_Masc_joao-correa-psd-d.jpg"))))

# now, will take the mean of the 7*7 convolution from this layer 

######### ITERATE AND PRODUCE PREDICTIONS  --------------
cd("images/brazil_images_public")
feature_process(path)  = mean(pysqueeze(resmodelconv(pypreprocess(pyload(path)))), dims=(1,2))
features = @showprogress [vcat(feature_process(x)...) for x in training_images.path ]
features = mapreduce(permutedims, vcat, features)
label_m_f(key) = contains(key, r"Fem") ? "Woman" : "Man";
y = label_m_f.(training_images.path);

###### CONV RESNET MODEL W/ SVM - results 

## LOADING SCIKITLEARN 
import ScikitLearn: CrossValidation
svm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)

# accuracy is atrocious, why is it sooo bad? when done in python, it is fine 
RSK = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=3403)
resnet_conv_out = cross_val_score(svm, features, y, cv = RSK.split(features,  y))


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
pysqueeze(vggmodel(pypreprocess(pyload("age46_Masc_joao-correa-psd-d.jpg"))))

# now, will take the mean of the 7*7 convolution from this layer 

######### ITERATE AND PRODUCE PREDICTIONS  --------------
cd("images/brazil_images_public")
feature_process(path)  = pysqueeze(vggmodel(pypreprocess(pyload(path))))
features = @showprogress [feature_process(x) for x in training_images.path ]
features = mapreduce(permutedims, vcat, features)
label_m_f(key) = contains(key, r"Fem") ? "Woman" : "Man";
y = label_m_f.(training_images.path);

###### CONV RESNET MODEL W/ SVM

## LOADING SCIKITLEARN 
import ScikitLearn: CrossValidation
svm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)

# accuracy is atrocious, why is it sooo bad? when done in python, it is fine 
RSK = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=3403)
vgg_out = cross_val_score(svm, features, y, cv = RSK.split(features,  y))

############################################
##### FEATURE EXPLORATION WITH THE EXAMPLE DATA 
####################################################

# flow: load images in julia - 
example_path = "images/brazil_images_public/age46_Masc_joao-correa-psd-d.jpg"
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

# convert and save the noised image 
function convert_and_save(rawimg)
    tempfile = tempname()
    save(tempfile*".png", Gray.(rawimg))
    return(tempfile*".png")
end

########
rawimg_orig = deepcopy(rawimg)

# THE PIPELINE - (RE)TRAINING THE SVM

#using resnet and the svm already trained (resvm)
resmodel(img) = py"nn_model.predict"(img)
@sk_import calibration: CalibratedClassifierCV
calsvm = CalibratedClassifierCV(resvm)
calsvm.predict_proba(features)
# TEST prediction on one example at a time 
calsvm.predict_proba(reshape(features[1, :], (1, 2048)))

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
maf = mask_viz(load("most_ambiguous_female_alessandra-amatto-d.jpg"), 5, calsvm, .9) 

mam = mask_viz(load("most_ambiguous_male_acelmo-assuncao-d.jpg"), 5, calsvm, .9) 

mfw = mask_viz(load("most_feminine_woman_simone-xucra-d.jpg"), 5, calsvm, .9) 

mmm = mask_viz(load("most_masculine_man_claudio-ocozias-d.jpg"), 5, calsvm, .9) 