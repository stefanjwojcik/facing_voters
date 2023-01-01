using Flux, Images
using Flux: onehotbatch
using Flux.Data: DataLoader
using StatsBase: sample, shuffle
using CSV, DataFrames, CUDA
using Pipe, Images
#CUDA.allowscalar(false)

if isempty(readdir("data"))
  error("Empty train folder - perhaps you need to run get_data.jl.")
end

# define some constants 
const impath = "images/official_images_data/"

### Create a bunch of CONST files for loading images by batches 

function load_paths(filepath, colname)
  # load paths to images 
  df = CSV.read(filepath, DataFrame)
  # filter out non-image files 
  paths = impath .* df[:, colname]
end

## NORMALIZE IMAGES 
function normalize(img, nsize)
  # normalize images 
  @pipe img |> 
  RGB.(_) |>
  Images.imresize(_, nsize...) |> 
  (channelview(_) .* 255 .- 128)./128 |> 
  Float32.(permutedims(_, (3, 2, 1))[:,:,:,:])
end

# function to load training data, append label prefix
#= function load_images_from_disk(filepath::String, colname, prefix::String)
  df = CSV.read(filepath, DataFrame)
  labels = repeat([prefix], length(df[:, colname]))
  labels = map(x -> occursin("$prefix",x) ? 1 : 0, labels)
  labels = Flux.onehotbatch(labels, [0,1])
  imgs = custom_img_preprocess.(Images.load.(impath .* df[:, colname] ))
  imgs = cat(imgs..., dims = 4)
  # This is requires since the model's input is a 4D array
  # The default is float64 but float32 is commonly used which is why we use it
  Float32.(imgs), labels
end =#

# be sure to CD into the data folder 
# const masc_training = load_training_data("data/masculino_training_sample.csv", "masc")
# const fem_training = load_training_data("data/feminino_training_sample.csv", "fem")
# const training = [masc_training; fem_training]

# # Takes in the number of requested images per batch ("n") and image size
# # Returns a 4D array with images and an array of labels
# # This function is clunky because I need to avoid any scalar indexing. 
# function load_batch(n = 10, nsize = (224,224))
#   if ((n % 2) != 0)
#       print("Batch size must be an even number")
#   end

#   imgs_paths = shuffle(vcat(sample(masc_training, Int(n/2)), sample(fem_training, Int(n/2))))
  
#   # Generate image labels 
#   labels = map(x -> occursin("fem",x) ? 1 : 0, imgs_paths)
  
#   # Convert the text based names to 0 or 1 (one hot encoding)
#   CUDA.@allowscalar labels = Flux.onehotbatch(labels, [0,1])
  
#   # Load all of the images, remove prefixes. 
#   imgs = Images.load.(impath .* replace.(imgs_paths, r"^fem" => "", r"^masc" => "", count=1 ))
#   # Turn to RGB for continuity w/ grey scale images 
#   imgs = map(img -> RGB.(img), imgs)
  
#   # Re-size the images based on imagesize from above (most models use 224 x 224)
#   imgs = map(img -> Images.imresize(img, nsize...), imgs)
  
#   # Change the dimensions of each image, switch to gray scale. Channel view switches to...
#   # a 3 channel 3rd dimension and then (3,2,1) makes those into seperate arrays.
#   # So we end up with [:, :, 1] being the Red values, [:, :, 2] being the Green values, etc
#   imgs = map(img -> permutedims(channelview(img), (3,2,1)), imgs)
#   # Result is two 3D arrays representing each image
  
#   # Concatenate the two images into a single 4D array and add another extra dim at the end
#   # which shows how many images there are per set, in this case, it's 2
#   imgs = cat(imgs..., dims = 4)
#   # This is requires since the model's input is a 4D array
  
#   # Convert the images to float form and return them along with the labels
#   # The default is float64 but float32 is commonly used which is why we use it
#   Float32.(imgs), labels
# end
