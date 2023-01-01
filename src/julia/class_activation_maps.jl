# Feature Analysis using Class Activation 

### Testing here 
img = get_example("images/brazil_images_public/");
img_view = Gray.(img[1,:,:,1]/255)
cam = get_class_activation_map(py"resmodel", img)

# show the resulting img 
cam_paint(img, cam)

maf = cam_paint("images/most_ambiguous_female_alessandra-amatto-d.jpg", py"resmodel", .01) 
mam = cam_paint("images/most_ambiguous_male_acelmo-assuncao-d.jpg", py"resmodel", .01) 
mfw = cam_paint("images/most_feminine_woman_simone-xucra-d.jpg", py"resmodel", .01) 
mmm = cam_paint("images/most_masculine_man_claudio-ocozias-d.jpg", py"resmodel", .01) 