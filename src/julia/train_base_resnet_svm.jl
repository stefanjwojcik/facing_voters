# THE BASE RESNET MODEL 

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


impath = "images/official_images_data/"
include("src/julia/utils.jl")

# CD into the images folder 
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

# py"ResNet50(weights='imagenet').summary()" get the names 
_ = ResNet50(weights='imagenet')
nn_model = Model(inputs=_.input, outputs=_.get_layer('avg_pool').output)
    
"""
# create julia equivalent python functions 
pyload(image_path) = py"pyload"(image_path)
pypreprocess(img) = py"pypreprocess"(img)
pysqueeze(predictions) = py"np.squeeze"(predictions)

resmodel(img) = py"nn_model.predict"(img)
@sk_import calibration: CalibratedClassifierCV

## 
features = @showprogress [pysqueeze(resmodel(pypreprocess(pyload(impath * x)))) for x in training.path ]
features = mapreduce(permutedims, vcat, features)

# Generate resnet svm (resvm)
import ScikitLearn: CrossValidation
@sk_import svm: LinearSVC
import ScikitLearn: CrossValidation
resvm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
cv = ScikitLearn.CrossValidation.KFold(5000, n_folds=5, random_state = 3304, shuffle=true)
out = cross_val_score(resvm, features, y, cv = cv)

calsvm = CalibratedClassifierCV(resvm)
calsvm.fit(features, y)

#output the GCS scores for the training data 
#probs = calsvm.predict_proba(features)

# Merging of probabilities and training 
#probs = DataFrame(probs = probs[:, 2], path = training_images.path)

## RUNNING ON THE FULL SAMPLE: 

getfeatures(x) = pysqueeze(resmodel(pypreprocess(pyload(x))))
calib_svm_pred(x) = calsvm.predict_proba(reshape(getfeatures(x), (1,2048)))[2]

######### Estimates of body  
fem_body = Float64[]
@showprogress for img in readdir(impath)
    push!(fem_body, calib_svm_pred(impath * img))
end

body_out = DataFrame(imglink = readdir(impath), fem_body = fem_body)
CSV.write("data/fem_body_officialV2.csv", body_out)
######### Estimates with face locations 

face_locs = CSV.read("data/official_face_coords.csv", DataFrame)

training.gender = y
train_face = leftjoin(training, face_locs, on = [:path => :imglink])

# TRAINING THE FACE MODEL 
face_features = Matrix{Float32}(undef, 5000, 2048)
@showprogress for row in eachrow(train_face)
    if row.t == row.b == row.l == row.r == 0
        face_features[rownumber(row), :] .= getfeatures(impath * row.path)
    else 
        loaded_img = load(impath* row.path)[row.t + 1:row.b, row.l + 1:row.r]
        img_saved = convert_and_save(loaded_img; ftype=".jpg")
        feat = getfeatures(img_saved)
        face_features[rownumber(row), :] .= feat
    end
end

# Generate resnet svm (resvm)
resvm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
#cv = ScikitLearn.CrossValidation.KFold(5000, n_folds=5, random_state = 3304, shuffle=true)
#out = cross_val_score(resvm, face_features, train_face.gender, cv = cv)

calsvm = CalibratedClassifierCV(resvm)
calsvm.fit(face_features, train_face.gender)
calib_svm_pred(x) = calsvm.predict_proba(reshape(getfeatures(x), (1,2048)))[2]

fem_face = Float64[]

@showprogress for row in eachrow(face_locs)
    if isfile(impath * row.imglink)
        if row.t == row.b == row.l == row.r == 0
            push!(fem_face, calib_svm_pred(impath * row.imglink))
        else
            loaded_img = load(impath* row.imglink)[row.t + 1:row.b, row.l + 1:row.r]
            img_saved = convert_and_save(loaded_img; ftype=".jpg")
            push!(fem_face, calib_svm_pred(img_saved))
        end
    else 
        push!(fem_face, .5000)
        println("pushing .5000 to $(row.imglink)")
    end
end

face_out = DataFrame(imglink = face_locs.imglink, fem_face = fem_face)
CSV.write("data/fem_face_officialV2.csv", face_out)

