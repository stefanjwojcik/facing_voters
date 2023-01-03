# Fine-tuning a model in Julia. 
# Takes the ResNet pretrained model and uses that as a base model to create an alternative conformity score 

using CUDA
using Metalhead
using Flux
using Flux: @epochs
using StatsBase
using Statistics
using ProgressMeter

## Adding the data-loading utilities 
if !isfile("fine_tune_dataloader.jl")
    include("src/julia/fine_tune_dataloader.jl")
else 
    include("fine_tune_dataloader.jl")
end

# to reclaim gpu memory 
#GC.gc(true); CUDA.reclaim(); 
CUDA.allowscalar(false)

# Load and prepare the paths to the source images
masc_imgs = load_paths("data/masculino_training_sample.csv", "path")
fem_imgs = load_paths("data/feminino_training_sample.csv", "path")

# Load the ResNet model as a pre-trained model
resnet = ResNet(50; pretrain = true)
nn_model = Chain(
  resnet.layers[1],
  resnet.layers[2][1:end-1],
) |> gpu 

# This is basically 100x faster with the GPU 
masc_features = @showprogress [nn_model(normalize(load(x), (224, 224)) |> gpu) for x in masc_imgs] 
fem_features = @showprogress [nn_model(normalize(load(x), (224, 224)) |> gpu) for x in fem_imgs] 
cmasc_features = cpu(masc_features); #moves features to CPU and reclaims space
cfem_features = cpu(fem_features);
masc_features = nothing; fem_features = nothing; # reclaim memory

# Assign earlier model to nothing and garbage collect
resnet = nothing
nn_model = nothing
GC.gc(true); CUDA.reclaim();

#### Create a minor neural network on top of the resnet features 
# This is a simple 2 layer neural network
top_model = Chain(
  Dense(2048, 512, relu),
  Dense(512, 2),
  softmax
) |> gpu

## Create training and validation sets
train_size = round(Int, 0.8 * size(cmasc_features, 1))
test_size = size(cmasc_features, 1) - train_size

# Combine the data into a single array

## TRAINING
tr_embeddings = hcat(hcat(cmasc_features[1:train_size]...), hcat(cfem_features[1:train_size]...))
tr_labels = Flux.onehotbatch(repeat([0, 1], inner=train_size), [0,1])
# TESTING 
te_embeddings = hcat(hcat(cmasc_features[train_size+1:end]...), hcat(cfem_features[train_size+1:end]...))
te_labels = Flux.onehotbatch(repeat([0, 1], inner=test_size), [0,1])

# dataset
dataset = [(tr_embeddings, tr_labels)]

# Define loss functions 
opt = ADAM()
loss(x,y) = Flux.Losses.logitcrossentropy(top_model(x), y)
accuracy(x, y) = Statistics.mean(Flux.onecold(top_model(x)) .== Flux.onecold(y))

# Define trainable parameters - just the top 
ps = Flux.params(top_model)

# Train for two epochs 

############

# Train the model over 100_000 epochs
for epoch in 1:1000
    # Implement Stochastic Gradient Descent 
    Flux.train!(loss, ps, dataset |> gpu, opt)

    # stop if the training accuracy is greater than testing accuracy
    if accuracy(tr_embeddings |> gpu, tr_labels |> gpu) > accuracy(te_embeddings |> gpu, te_labels |> gpu) + 0.02
        break
    end

    # Print loss function values 
    if epoch % 10 == 0
        println("Epoch: $(epoch) Validation Results")
        @show loss(te_embeddings |> gpu, te_labels |> gpu)
        @show accuracy(te_embeddings |> gpu, te_labels |> gpu)
        println()
    end
end