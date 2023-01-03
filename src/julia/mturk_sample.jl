
####################
### ORIGINAL MTURK SAMPLE - 3K coded images, ** sent to MTURK May 15, 2022  4:17 AM PDT ** 
#####################
# Random sample of M/W for Mturk
cd("/home/swojcik/github/facing_voters/")
using CSV, Random, DataFrames, Gadfly, Cairo, Fontconfig , Statistics, StatsBase
using AWS, AWSS3, Pipe

## CODE TO CREATE MTURK TRAINING IMAGES 
training_images = CSV.read("data/brazil_public_training_images_names.csv", DataFrame)

label_m_f(key) = contains(key, r"Fem") ? "Woman" : "Man";
y = label_m_f.(training_images.path);

wsamp = training_images.path[rand(findall(x -> x == "Woman", y), 500)];
msamp = training_images.path[rand(findall(x -> x == "Man", y), 500)];
samp = [wsamp; msamp]

out = DataFrame(image_url = "https://s3.amazonaws.com/brazil.images.public/" .* samp)
CSV.write("data/image_urls_prod_1k_oftraining.csv", out)

####################
### END
#####################

#################################
## CODE TO MERGE THE MTURK OFFICIAL RESULTS WITH THE GCS SCORES: 
#########################################
# retrain the model here - using code from transfer_analyses.jl resnet model 
include("src/julia/train_base_resnet_svm.jl");


# Functions to recode qualitative responses in MTURK to numeric 1-5 scale
myfn(x) = replace(x, "." => "_")
recodefn(x) = replace(x, "Very Feminine" => 5, 
                      "Somewhat Feminine" => 4, 
                      "Neither Feminine nor Masculine" => 3, 
                      "Somewhat Masculine" => 2, 
                      "Very Masculine" => 1, 
 )
# Read in mturk results 
mturk = CSV.read("data/mturk_1k_training.csv", DataFrame, silencewarnings=true)
rename!(myfn, mturk) # rename columns to somethign more julia friendly

# recode the mturk results - modify image names, recode the labels to numeric, and take the mean of the 3 ratings per image
mturk_recoded = mturk |> 
        (data -> combine(data, :, :Input_image_url => x -> replace.(x, "https://s3.amazonaws.com/brazil.images.public/" => "" ) ) ) |>
        (data -> select(data, [:Answer_category_label, :Input_image_url_function])) |> 
        (data -> combine(data, :, :Answer_category_label => x -> recodefn(x) )) |> 
        (data -> groupby(data, :Input_image_url_function)) |> 
        (data -> combine(data, :Answer_category_label_function => mean => :turk_gcs ))

## join with the GCS data (left out of this script)
#out = leftjoin(mturk_recoded, 
#            probs |> 
#            (data -> groupby(data, :path)) |> 
#            (data -> combine(data, :probs => mean, renamecols=false)),
#            on = [:Input_image_url_function => :path])

##########
# OLD CODE TO LOOK AT MTURK compared to GCS, now use mturk_analysis.jl
#############

# correlation 
#CSV.write("data/mturk_plot_data.csv", out)
#out = CSV.read("data/mturk_plot_data.csv", DataFrame)
#cor(out.turk_gcs, out.probs)

#plotout = Gadfly.plot(out, x=:turk_gcs, y=:probs, color=[colorant"red"], Geom.point, Geom.smooth)

#l1 = Gadfly.layer(out, x=:turk_gcs, y=:probs, Geom.point, alpha=[.5], 
#        Theme(highlight_width=0mm));
#l2 = layer(out, x=:turk_gcs, y=:probs, color=[colorant"lightsalmon"], Geom.smooth, Theme(line_width=2mm));
#mturk_plot = Gadfly.plot(l1, l2, 
#        Guide.xlabel("MTurk Annotated Expression Scores (higher = more feminine)"), 
#        Guide.ylabel("GCS (higher = more feminine)"), Guide.title("GCS vs Annotated Images"))
#draw(PNG("images/mturk_plot.png", 20cm, 15cm), mturk_plot)


####################
### NEWER MTURK SAMPLE - BASED on OFFICIAL DATA - sent to MTURK May 31, 2022 10:21 AM PDT
#####################

## JOIN OLD MTURK WITH OFFICIAL DATA (FUZZY JOIN)

mturk = CSV.read("data/mturk_plot_data.csv", DataFrame);
old_data = CSV.read("data/df_name_map.csv", DataFrame) ;
old_data.cand_number = old_data.Número;
old_data.imglink = replace.(old_data.imglink, "Male" => "Masc", "Female" => "Fem")
mturk = leftjoin(mturk, old_data, on = [:Input_image_url_function => :imglink]) |> 
        (data -> select(data, [:name, :turk_gcs, :Input_image_url_function, :cand_number]))
# Columns to merge: MTURK 
mturk.firstname = uppercase.(first.(split.(mturk.name, "-")))
mturk.secondname = uppercase.(map(x -> length(x) > 1 ? x[2] : "", split.(mturk.name, "-")))
## DROP non-numeric candidate numbers 
mturk = filter(x-> contains.(x.cand_number, r"[0-9]"), mturk)
mturk.cand_number = parse.(Int64, mturk.cand_number)

# Load official data 
include("src/julia/utils.jl")
impath = "images/official_images_data/"

df = CSV.read("data/official_local_election_data_2016.csv", DataFrame);
df = filter(r -> any(startswith.(["PREFEITO", "VEREADOR"], r.DS_CARGO)), df)
# Columns to merge: OFFICIAL 
df.firstname = first.(split.(df.NM_URNA_CANDIDATO, " "))
df.secondname = map(x -> length(x) > 1 ? x[2] : "", split.(df.NM_URNA_CANDIDATO, " "))
df.imglink = map(generate_img_link, eachrow(df))
df.downloaded = isfile.("images/official_images_data/" .* df.imglink)
df = df |> (data -> filter(:downloaded => x -> x == true, data))

df = df |> 
        (data -> select(data, [:firstname, :secondname, :imglink, :NR_CANDIDATO, :DS_GENERO]))

## NEED to know which of the old MTURK cases failed to merge with the new official data 
## JOIN on FIRST NAME, SECOND NAME, CAND NUMBER 
out = innerjoin(mturk, df, on = [:firstname => :firstname, 
                                :secondname => :secondname, 
                                :cand_number => :NR_CANDIDATO ])

#FIND AND FILTER NEW CASES FOR MTURK
## 316 men and 320 women 
# NEED 193 men and 188 women 
out |> (data -> groupby(data, :DS_GENERO)) |> (data -> combine(data, :turk_gcs => mean))
out |> (data -> groupby(data, :DS_GENERO)) |> (data -> combine(data, :turk_gcs => length))
CSV.write("data/mturk_official_joined_sample.csv", out)

## SAMPLE FROM SET DIFF OF EXISTING DATA 
dfmen = df |> (data -> filter(:DS_GENERO => x -> x == "MASCULINO", data ))
dfmen = StatsBase.sample(setdiff(dfmen.imglink, out.imglink), 193, replace=false)
dfwomen = df |> (data -> filter(:DS_GENERO => x -> x == "FEMININO", data ))
dfwomen = StatsBase.sample(setdiff(dfwomen.imglink, out.imglink), 188, replace=false)

# WRITE THE NEW MTURK SAMPLE TO DISK
CSV.write("data/men_official_mturk_sample.csv", DataFrame(imglink = dfmen))
CSV.write("data/women_official_mturk_sample.csv", DataFrame(imglink = dfwomen))

# Write the MTURK INPUT FILE 
CSV.write("data/mturk_official_input_file.csv", 
       DataFrame(image_url =  "https://s3.amazonaws.com/official.images.data/mturk/" .* [dfmen; dfwomen]) )

### WRITING TO NEW BUCKET 
aws = global_aws_config(; region="us-east-1")
s3_put(aws, "brazil.images.public/official", "this.jpg", "images/official_images_data/"*readdir(impath)[1] )

for link in [dfwomen; dfmen]
        cmd = `aws s3api put-object --bucket official.images.data --key mturk/$link --body images/official_images_data/$link`
        read(cmd, String)
end

##################
# END
##################

########################
## LOAD, JOIN, and PROCESS ALL THE MTURK DATA - for mturk_analysis.jl
##################

# Load the original mturk codes - dematched with the official data above 
# Get their img-links, then merge with the original mturk data to get coder scores 
# Concatenate the original mturk data with the coded official data to create an entire dataset 

# Load the data of dematched id's - those that matched with the official data 
mturk_original_dematched_ids = CSV.read("data/mturk_official_joined_sample.csv", DataFrame) |> 
                (data -> select(data, [:imglink, :Input_image_url_function]));
# Load the coded data 
mturk_original_coded = CSV.read("data/mturk_1k_coded.csv",DataFrame, silencewarnings=true) |> 
        (data -> rename!(data, Symbol.(replace.(string.(names(data)), "." => "_")))) |>
        (data -> combine(data, :, :Input_image_url => x -> replace.(x, "https://s3.amazonaws.com/brazil.images.public/" => "" ), renamecols=false ) ) |>
        (data -> select(data, [:WorkerId, :Input_image_url, :Answer_category_label])) |>
        (data -> combine(data, :, :Answer_category_label => x -> recodefn(x), renamecols=false )) |> 
        (data -> innerjoin(data, mturk_original_dematched_ids, on = [:Input_image_url => :Input_image_url_function])) |> 
        (data -> rename!(data, :Input_image_url => :Input_image_url_old)) |>
        (data -> rename!(data, :imglink => :Input_image_url)) |> 
        (data -> select(data, [:WorkerId, :Input_image_url, :Answer_category_label]));
# Coded official mturk data 
official_coded = CSV.read("data/mturk_official_coded.csv", DataFrame, silencewarnings=true) |> 
        (data -> rename!(data, Symbol.(replace.(string.(names(data)), "." => "_")))) |>
        (data -> select(data, [:WorkerId, :Input_image_url, :Answer_category_label])) |>
        (data -> combine(data, :, :Answer_category_label => x -> recodefn(x), renamecols=false )) |>
        (data -> combine(data, :, :Input_image_url => x -> replace.(x, "https://s3.amazonaws.com/official.images.data/mturk/" => "" ), renamecols=false ) ) ;
# Concatenate the two datasets 
mturk_coded = vcat(mturk_original_coded, official_coded);
CSV.write("data/mturk_coded_1k_final.csv", mturk_coded)

##############
# END
#############

