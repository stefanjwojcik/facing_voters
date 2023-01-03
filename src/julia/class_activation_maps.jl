# Feature Analysis using Class Activation 

include("src/julia/class_activation_utils.jl")
using Plots

# heat map function 
function cam_heat_composite(imgpath, model)
    # load image 
    img = load(imgpath)
    # get class activation map 
    cam = get_class_activation_map(model, get_img(imgpath))
    # return heatmap 
    return heatmap(cam_heat(img, cam), legend=false)
end

### Testing here 
cam_heat_composite("images/most_ambiguous_female_alessandra-amatto-d.jpg", py"resmodel")

cam = get_class_activation_map(py"resmodel", get_img("images/most_ambiguous_female_alessandra-amatto-d.jpg"));

####### THE FINAL IMAGES 
maf = cam_heat_composite("images/most_ambiguous_female_alessandra-amatto-d.jpg", py"resmodel") 
mam = cam_heat_composite("images/most_ambiguous_male_acelmo-assuncao-d.jpg", py"resmodel") 
mfw = cam_heat_composite("images/most_feminine_woman_simone-xucra-d.jpg", py"resmodel") 
mmm = cam_heat_composite("images/most_masculine_man_claudio-ocozias-d.jpg", py"resmodel") 

# Save the images
png(maf, "images/maf_cam_heat.png")
png(mam, "images/mam_cam_heat.png")
png(mfw, "images/mfw_cam_heat.png")
png(mmm, "images/mmm_cam_heat.png")