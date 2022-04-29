# facing_voters
Replication of scientific findings of the paper entitled "Facing Voters: Gender Expression, Gender Stereotypes, and Vote Choice"

## LOAD KEY UTILITIES/LIBRARIES

## RUN SCRIPT TO DOWNLOAD IMAGES AND DATA ZIP FILES 

CD into the directory that you want to save the images to. 
```julia
using AWSS3
using AWS

# set up access to AWS (docs -> https://github.com/JuliaCloud/AWSS3.jl)
aws = global_aws_config(; region="us-east-1")

new_bucket = s3_list_objects(aws, "brazil.images.public")
new_keys = [x["Key"] for x in new_bucket]

@showprogress for x in new_keys
    #println(newkey)
    s3_get_file(aws, "brazil.images.public", x, x)
end
```

## RUN SCRIPT TO DO TRANSFER LEARNING - PRODUCE CV RESULTS TABLE 

- Load the training/test data image list 




## RUN ANALYSIS SCRIPT - PRODUCE FINAL TABLES AND FIGURES 
