## Store all image data locally 
# This script can be run non-interactively. 

using AWSS3
using AWS
using ProgressMeter

# base directory path 
base_dir_path = pwd()

# ask to initialize download 
function promptdata()
    inp = ""
    println("This script will download more than 2GB of data. Initiate data download? Y/N")
    inp = readline()
    while inp ∉ ["Y", "N"]
        inp = promptdata()
    end
    return inp
end
# Ask to download data 
download_data = promptdata() == "Y"

if download_data

    println("Saving data to folder $(abspath("images/brazil_images_public"))")

    aws = global_aws_config(; region="us-east-1")

    # Images - to source file location  - assuming current working directory is 'masc_faces'
    if not isdir(abspath("images/brazil_images_public"))
        mkdir(abspath("images"))
        mkdir(abspath("images/brazil_images_public"))
    end

    # make path to save the images 
    image_path = abspath("images/brazil_images_public")
    cd(image_path) #cd to the images path 

    new_bucket = s3_list_objects(aws, "brazil.images.public")
    new_keys = [x["Key"] for x in new_bucket]

    @showprogress for x in setdiff(new_keys, readdir())
        #println(newkey)
        s3_get_file(aws, "brazil.images.public", x, x)
    end

    # go back to base path 
    cd(base_dir_path) # back to base
    cd(abspath("images")) #back to images 

    # Training image names "
    # cmd: aws s3api get-object --bucket brazil.features --key brazil_training_image_names  ~/Downloads/brazil_training_image_names
    # cmd: aws s3api get-object --bucket brazil.features --key brazil_testing_image_names  ~/Downloads/brazil_testing_image_names
    if "brazil_public_training_image_names.csv" ∉ readdir()
        read(`aws s3api get-object --bucket brazil.features --key brazil_public_training_image_names.csv brazil_public_training_image_names.csv`, String)
    end
    if "brazil_public_testing_image_names.csv" ∉ readdir()
        read(`aws s3api get-object --bucket brazil.features --key brazil_public_testing_image_names.csv brazil_public_testing_image_names.csv`, String)
    end

    cd(base_dir_path) #back to base 

    println("Done. Image Data now available")

    #### DONE WITH IMAGE DATA, NOW GET VOTING RESULTS 

    println("Getting Voting Data....")

    if not isdir(abspath("data"))
        mkdir(abspath("data"))
    end

    cd(base_dir_path) # back to base
    cd(abspath("data")) # to data path

    data_to_recover = ["official_local_election_data_2016.csv", ""]

    if "official_local_election_data_2016.csv" ∉ readdir()
        read(`aws s3api get-object --bucket brazil.features --key official_local_election_data_2016.csv official_local_election_data_2016.csv`, String)
    end

    if "masculino_training_sample.csv" ∉ readdir()
        read(`aws s3api get-object --bucket brazil.features --key masculino_training_sample.csv masculino_training_sample.csv`, String)
    end

    if "feminino_training_sample.csv" ∉ readdir()
        read(`aws s3api get-object --bucket brazil.features --key feminino_training_sample.csv feminino_training_sample.csv`, String)
    end

    if "votacao_candidato_munzona_2016_BRASIL.csv" ∉ readdir()
        read(`aws s3api get-object --bucket brazil.features --key votacao_candidato_munzona_2016_BRASIL.csv votacao_candidato_munzona_2016_BRASIL.csv`, String)
    end

    if "fem_face_officialV2.csv" ∉ readdir()
        read(`aws s3api get-object --bucket brazil.features --key fem_face_officialV2.csv fem_face_officialV2.csv`, String)
    end

    println("Done. All Data Downloaded Successfully.")
    
end