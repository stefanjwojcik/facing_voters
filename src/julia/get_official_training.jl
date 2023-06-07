using DataFrames, CSV, AWSS3, AWS, ProgressMeter

aws = global_aws_config(; region="us-east-1")

masc_faces = CSV.read("data/masculino_training_sample.csv", DataFrame)
fem_training = CSV.read("data/feminino_training_sample.csv", DataFrame)

training = [masc_training; fem_training]

cd("images/official_images_data/")

impath = "images/official_images_data/"

@showprogress for x in setdiff(new_keys, readdir())
    #println(newkey)
    s3_get_file(aws, "official.images.data", x, x)
end