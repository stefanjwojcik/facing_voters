### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 889dbca6-5fa2-4474-b6c9-65a40c9b72eb
begin 
	cd("/home/ubuntu/facing_voters/")
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ 24fa47b4-a61c-4620-a634-db1c084573dd
begin
		# THE BASE RESNET MODEL 
		using DataFrames
		using PlutoUI, CUDA
		#using Plots
		using Metalhead
		using Flux
		using StatsBase
		using ScikitLearn
		using ProgressMeter
		using CSV
		using PyCall
		using FileIO
		using Images
end

# ╔═╡ 626935f8-a2bb-438d-8ab4-7c0653ec5faa
begin 
	using Plots
	
	# change working directory to source file location 
	#cd(@__DIR__)
	
	py"""
	import keras.backend as K
	import scipy
	from tensorflow.keras.applications.vgg19 import preprocess_input
	from tensorflow.keras.models import Model
	import numpy as np
	from tensorflow.keras.preprocessing import image
	from tensorflow.keras.applications.resnet50 import ResNet50
	# py"ResNet50(weights='imagenet').summary()" get the names 
	resmodel = ResNet50(weights='imagenet')
	def get_resnet_embedding_model(model):
	    Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
	
	"""
	#resmodel = py"resmodel"
	
	# LOAD IMAGES AND RESHAPE 
	py"""
	def pypreprocess(img):
	    x = image.img_to_array(img)
	    x = np.expand_dims(x, axis=0)
	    x = preprocess_input(x)
	    return(x)
	def pyload(image_path):
	    img = image.load_img(image_path, target_size=(224, 224))
	    return(img)
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

	# heat map function 
	function cam_heat_composite(imgpath, model)
	    # load image 
	    img = load(imgpath)
	    # get class activation map 
	    cam = get_class_activation_map(model, get_img(imgpath))
	    # return heatmap 
	    return heatmap(cam_heat(img, cam), legend=false)
	end

	####### THE FINAL IMAGES 
	maf = cam_heat_composite("images/most_ambiguous_female_alessandra-amatto-d.jpg", py"resmodel") 
	mam = cam_heat_composite("images/most_ambiguous_male_acelmo-assuncao-d.jpg", py"resmodel") 
	mfw = cam_heat_composite("images/most_feminine_woman_simone-xucra-d.jpg", py"resmodel") 
	mmm = cam_heat_composite("images/most_masculine_man_claudio-ocozias-d.jpg", py"resmodel") 

	plot(maf, mam, mfw, mmm)
	
end


# ╔═╡ 2ad9d26b-8ee7-4a71-a5cc-20e7f91e41cb
begin 
	function convert_and_save(rawimg; ftype=".png")
    	tempfile = tempname()
    	save(tempfile*ftype, Gray.(rawimg))
    	return(tempfile*ftype)
	end

	using ScikitLearn.CrossValidation: cross_val_score
	
	
	impath = "images/official_images_data/"
	
	# CD into the images folder and read into memory
	masc_training = CSV.read("data/masculino_training_sample.csv", DataFrame)
	fem_training = CSV.read("data/feminino_training_sample.csv", DataFrame)
	training = [masc_training; fem_training]
	
	# Generate labels for the training data
	y = [repeat(["Man"], nrow(masc_training)); repeat(["Woman"], nrow(fem_training))]
	
	# LOAD IMAGES AND RESHAPE 
	py"""
	from tensorflow.keras.preprocessing import image
	from tensorflow.keras.applications.vgg19 import preprocess_input
	from tensorflow.keras.models import Model
	from tensorflow.keras.applications.resnet50 import ResNet50
	import numpy as np
	# loading images 
	def pyload(image_path):
	    img = image.load_img(image_path, target_size=(224, 224))
	    return(img)
	# use python preprocessing function 
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

	getfeatures(x) = pysqueeze(resmodel(pypreprocess(pyload(x))))
	calib_svm_pred(x) = calsvm.predict_proba(reshape(getfeatures(x), (1,2048)))[2]
	
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
	body_out = cross_val_score(resvm, features, y, cv = cv)

	# Trained/calibrated model :
	calsvm = CalibratedClassifierCV(resvm)
	calsvm.fit(features, y)
	
	#############################################
	######### Estimates with face locations 
	###############################
	
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
	cv = ScikitLearn.CrossValidation.KFold(5000, n_folds=5, random_state = 3304, shuffle=true)
	face_out = cross_val_score(resvm, face_features, train_face.gender, cv = cv)
	
end

# ╔═╡ 0670606b-8ce9-4d09-9c64-2dd3326df80f
begin
#using DataFrames, PlutoUI, HypertextLiteral, ImageView, Plots
PlutoUI.TableOfContents(title="Table of Contents", indent=true)
end


# ╔═╡ f6786e81-6b56-44a7-907c-71b386622130
md"""
# Replication Notebook 2: Facing Voters (Wojcik and Mullenax)

This is Notebook 2 of 2. It contains the code to replicate the results in the paper.

The lines below load Julia and Python libraries required for the analysis.
"""


# ╔═╡ 194abb04-cbfe-4512-9599-285e5a6d35b1
cd("/home/ubuntu/facing_voters/")

# ╔═╡ 6a24643d-1da6-42f1-9790-9c39eec98696
md"""
### Table 1
"""

# ╔═╡ 2aa4009f-f506-4eb8-b577-b67e52abc948
DataFrame(
	Cross_validation = ["CV Fold 1", "CV Fold 2", "CV Fold 3", "CV Fold 4", "CV Fold 5"],
	Face_Only = face_out, 
	Full_Image = body_out
)

# ╔═╡ f8a08cf3-d3f1-4eb4-bd4c-12117e65a691
md"""
### Figure 2: Image Map
"""

# ╔═╡ 1ab99acf-9db4-44f1-9174-a19c1ae1a1c8
Pkg.status()

# ╔═╡ 347cd581-8b9e-45a3-bdec-87b7bc33a353
md"""
### Table 4: Alternative GCS Models
"""

# ╔═╡ 7a7b11d9-aa43-46bb-801c-e30413a8cf86
let 

	######################### 
	# RESNET DIFFERENT LAYER 
	##################
	py"""
	import keras.backend as K
	import scipy
	from tensorflow.keras.applications.vgg19 import preprocess_input
	from tensorflow.keras.models import Model
	import numpy as np
	from tensorflow.keras.preprocessing import image
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
	svm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
	
	cv = ScikitLearn.CrossValidation.KFold(5000, n_folds=5, random_state = 3304, shuffle=true)
	resnet_conv_out = cross_val_score(svm, features, y, cv = cv)
	
	
	######################### 
	# DIFFERENT MODEL - VGG19
	##################
	
	py"""
	from tensorflow.keras.applications.vgg19 import VGG19
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
	svm = LinearSVC(C=.0001, loss="squared_hinge", penalty="l2", multi_class="ovr", random_state = 35552, max_iter=2000)
	 
	cv = ScikitLearn.CrossValidation.KFold(5000, n_folds=5, random_state = 3304, shuffle=true)
	vgg_out = cross_val_score(svm, features, y, cv = cv)

	DataFrame(
	Cross_validation = ["CV Fold 1", "CV Fold 2", "CV Fold 3", "CV Fold 4", "CV Fold 5"],
	VGG19 = vgg_out,
	AlternateLayer = resnet_conv_out
)

end 


# ╔═╡ Cell order:
# ╠═889dbca6-5fa2-4474-b6c9-65a40c9b72eb
# ╠═24fa47b4-a61c-4620-a634-db1c084573dd
# ╠═0670606b-8ce9-4d09-9c64-2dd3326df80f
# ╠═f6786e81-6b56-44a7-907c-71b386622130
# ╠═194abb04-cbfe-4512-9599-285e5a6d35b1
# ╠═2ad9d26b-8ee7-4a71-a5cc-20e7f91e41cb
# ╠═6a24643d-1da6-42f1-9790-9c39eec98696
# ╠═2aa4009f-f506-4eb8-b577-b67e52abc948
# ╠═f8a08cf3-d3f1-4eb4-bd4c-12117e65a691
# ╠═1ab99acf-9db4-44f1-9174-a19c1ae1a1c8
# ╠═626935f8-a2bb-438d-8ab4-7c0653ec5faa
# ╠═347cd581-8b9e-45a3-bdec-87b7bc33a353
# ╠═7a7b11d9-aa43-46bb-801c-e30413a8cf86
# ╠═9ca12f89-6c8c-47c3-98d9-739a7ffcb8e2
