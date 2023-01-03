# Facing Voters: Gender Expression, Gender Stereotypes, and Vote Choice
Replication of scientific findings of the paper entitled "Facing Voters: Gender Expression, Gender Stereotypes, and Vote Choice"

## Replicating Regression Results: 

- Figure 1: Histogram of GCS scores (see [official_analysis.R](src/R/official_analysis.R)
- Figure 2: 
- Figure 3: 
- Figure 4: 

- Table 1: 
- Table 2: 
- Table 3: 


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
