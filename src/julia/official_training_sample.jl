# Random sample of M/W for Training
cd("/home/swojcik/github/facing_voters/")
using CSV, Random, DataFrames, Gadfly, Cairo, Fontconfig , Statistics, StatsBase

## CODE TO CREATE MTURK TRAINING IMAGES 
official_images = CSV.read("/home/swojcik/github/facing_voters/data/official_local_election_data_2016.csv", DataFrame);
official_images = filter(r -> any(startswith.(["PREFEITO", "VEREADOR"], r.DS_CARGO)), official_images)

# Create official links 
function img_names_from_official(row::DataFrameRow{DataFrame, DataFrames.Index})
    # Join all information into the link 
    sg_uf = string(row.SG_UF)
    sg_ue = lpad(row.SG_UE, 5, "0")
    sq_ca = string(row.SQ_CANDIDATO)
    nr_tit = lpad(row.NR_TITULO_ELEITORAL_CANDIDATO, 12, "0" )
    nm = join([sg_uf, sg_ue, sq_ca, nr_tit], "-") * ".jpg"
    return(nm)
end

# Get images names as they are saved 
official_images.imgnames = map(img_names_from_official, eachrow(official_images))

# Create tag for whether the file exists on disc and filter 
official_images.downloaded = isfile.("images/official_images_data/" .* official_images.imgnames)

official_images_d = filter(row -> row.downloaded == 1, official_images)

# Select 2500 M and F among those that were downloaded successfully 
wsamp = official_images_d.imgnames[sample(findall(x -> x == "FEMININO", official_images_d.DS_GENERO), 2500, replace=false)];
msamp = official_images_d.imgnames[sample(findall(x -> x == "MASCULINO", official_images_d.DS_GENERO), 2500, replace=false)];

feminino_training_sample = DataFrame(path = wsamp)
masculino_training_sample = DataFrame(path = msamp)

CSV.write("data/feminino_training_sample.csv", feminino_training_sample)
CSV.write("data/masculino_training_sample.csv", masculino_training_sample)

#out = DataFrame(image_url = "https://s3.amazonaws.com/brazil.images.public/" .* samp)
#CSV.write("data/image_urls_prod_1k_oftraining.csv", out)

