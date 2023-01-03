# Facing Voters: Gender Expression, Gender Stereotypes, and Vote Choice
Replication of scientific findings of the paper entitled "Facing Voters: Gender Expression, Gender Stereotypes, and Vote Choice"

## Replicating Main Paper Results: 

- Figure 1: Histogram of GCS scores - [official_analysis.R](src/R/official_analysis.R)
- Figure 2: Feature mapping of images - [class_activation_maps.jl](src/julia/class_activation_maps.jl)
- Figure 3: Effect of GCS in high information elections - [official_analysis.R](src/R/official_analysis.R)
- Figure 4: Effect of GCS in low information elections - [official_analysis.R](src/R/official_analysis.R)

- Table 1: Cross-validated accuracy scores [train_base_resnet_svm.jl](src/julia/train_base_resnet_svm.jl)
- Table 2: Regression results in high information elections - [official_analysis.R](src/R/official_analysis.R)
- Table 3: Regression results in low information elections - [official_analysis.R](src/R/official_analysis.R)

## Replicating Appendix Results: 
- Table 4: Cross-validated accuracy scores for alternative models - [transfer_analyses.jl(src/julia/transfer_analyses.jl)]
- Figure 5: Intercoder reliability / Human annotation/MTurk plot - [mturk_analysis.jl](src/julia/mturk_analysis.jl)
- Table 5: Alternative models with education included - [official_analysis.R](src/R/official_analysis.R)
- Figure 6: GCS by race and gender - [official_analysis.R](src/R/official_analysis.R)

# Accessing the images and training GCS Vision model from scratch 

## LOAD KEY UTILITIES/LIBRARIES

## RUN SCRIPT TO DOWNLOAD IMAGES AND DATA ZIP FILES 

CD into the main directory for this repo:
```shell
julia --project="" -e "include('get_data.jl');"
```
Creates directories: 
images/brazil_images_public (contains 193108 files)
images (contains brazil_public_testing_image_names.csv and brazil_public_training_image_names.csv)


## RUN SCRIPT TO DO TRANSFER LEARNING - PRODUCE CV RESULTS TABLE 

- Load the training/test data image list 




## RUN ANALYSIS SCRIPT - PRODUCE FINAL TABLES AND FIGURES 


## CONDITIONS OF DATA USAGE 

We make our data available under the condition that researchers not use it to train demographic classifiers of any kind. This includes models that attempt to infer characteristics like age, race, or gender.We make our data available under the condition that researchers not use it to train demographic classifiers of any kind. This includes models that attempt to infer characteristics like age, race, or gender.
