# Fine-tuning a model in JUlia

using CUDA
using Metalhead
using Flux
using Flux: @epochs
using StatsBase
using Statistics
include("dataloader.jl")

# load the model 
nn_model = GoogleNet().layers[1:19]
neurons = size(nn_model(rand(Float32, 224, 224, 3, 1)))[1]

## 
model = Chain(
  GoogleNet().layers[1:end-2],
  Dense(neurons, 1000),  
  Dense(1000, 256),
  Dense(256, 2),        # we get 2048 features out, and we have 2 classes
)

# Testing to get probabilities
model(rand(Float32, 224, 224, 3, 1))

# send model to GPU 
model = model |> gpu
dataset = [gpu.(load_batch(10)) for i in 1:10]

# Define loss functions 
opt = ADAM()
loss(x,y) = Flux.Losses.logitcrossentropy(model(x), y)
accuracy(x, y) = Statistics.mean(Flux.onecold(model(x)) .== Flux.onecold(y))

# Define trainable parameters - just the tip 
ps = Flux.params(model[2:end])

# Train for two epochs 
@epochs 2 Flux.train!(loss, ps, dataset, opt)

# SHOW THE RESULTS 
imgs, labels = gpu.(load_batch(10))
display(model(imgs))

labels

############

(m, n) = size(xtrain)

# Train the model over 100_000 epochs
for epoch in 1:1000
    # Randomly select a entry of training data 
    dataset = [gpu.(load_batch(10)) for x in 1:10]
    xt, yt = dataset[1]

    # Implement Stochastic Gradient Descent 
    Flux.train!(loss, ps, dataset, opt)

    # Print loss function values 
    if epoch % 10 == 0
        println("Epoch: $(epoch)")
        @show loss(xt, yt)
        @show accuracy(xt, yt)
        println()
    end
end