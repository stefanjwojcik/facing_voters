# FINAL MTURK ANALYSIS.jl
using CSV, DataFrames, Statistics

# Official MTURK Analysis for Facing Voters paper, R+R 2022

# create load and recode function 
function load_recode_data(filename, recode_function)
    mturk_codes = CSV.read(filename, DataFrame, silencewarnings=true) 
    #if recode_function != nothing
    mturk_codes = combine(mturk_codes, :, :Answer_category_label => x -> recode_function(x), renamecols=false ) 
    #end
    # Create a vector of labels for each image 
    all_labels  = []
    for img in unique(mturk_codes.Input_image_url)
            # Get all the labels for this image
            labels = filter(:Input_image_url => x -> x == img, mturk_codes).Answer_category_label
            push!(all_labels, labels)
    end

    # create a data frame of the three codes for each image
    df = DataFrame(image_url = unique(mturk_codes.Input_image_url), 
    l1 = [x[1] for x in all_labels], 
    l2 = [x[2] for x in all_labels],
    l3 = [x[3] for x in all_labels])

    return df
end

# get the number of workers
CSV.read("data/mturk_coded_1k_final.csv", DataFrame) |> 
    (x -> x[!, :WorkerId]) |>
    unique |> 
    length

# Load and recode with the standard recode function 
df = load_recode_data("data/mturk_coded_1k_final.csv", x -> x)

# Use R to calculate intercoder reliability
using RCall
@rlibrary irr
@rlibrary dplyr
@rput df

# Agreement on normally scaled data 
R"irr::agree(dplyr::select(df, l1, l2))$value" 
R"irr::agree(dplyr::select(df, l1, l3))$value"
R"irr::agree(dplyr::select(df, l2, l3))$value"

# Kappa on normally scaled data 
R"irr::kappa2(dplyr::select(df, l1, l2))$value" 
R"irr::kappa2(dplyr::select(df, l1, l3))$value"
R"irr::kappa2(dplyr::select(df, l2, l3))$value"

# Coarsened Data
coarse_recodefn(x) = replace(x, 5 => 3, 
                  4 => 3, 
                  3 => 2, 
                  2 => 1, 
                  1 => 1, 
)

# load with the coarse recoding function 
df_c = load_recode_data("data/mturk_coded_1k_final.csv", coarse_recodefn)
@rput df_c

# Agreement on coarsened data
R"irr::agree(dplyr::select(df_c, l1, l2))$value" 
R"irr::agree(dplyr::select(df_c, l1, l3))$value"
R"irr::agree(dplyr::select(df_c, l2, l3))$value"

# Kappa on coarsened data
R"irr::kappa2(dplyr::select(df_c, l1, l3))$value" 
R"irr::kappa2(dplyr::select(df_c, l2, l3))$value"
R"irr::kappa2(dplyr::select(df_c, l1, l2))$value"

#############################
#END
#############################

## join the official gcs scores with the mturk data

offd = CSV.read("/home/swojcik/github/facing_voters/data/data_for_regression.csv", DataFrame, silencewarnings=true) |> 
    (data -> rightjoin(data, df, on=[:imglink => :image_url])) |> 
    (data -> DataFrames.select(data, :imglink, :fem_body, :fem_face, :l1, :l2, :l3))

# Take the average of l1, l2, l3
offd.turk_gcs = (offd.l1 .+ offd.l2 .+ offd.l3) ./ 3

# correlations
cor(offd.turk_gcs, offd.fem_body)

# PLOT 
using Gadfly, Cairo, Fontconfig

l1 = Gadfly.layer(offd, x=:turk_gcs, y=:fem_body, Geom.point, alpha=[.5], 
        Theme(highlight_width=0mm));
l2 = layer(offd, x=:turk_gcs, y=:fem_body, color=[colorant"lightsalmon"], Geom.smooth, Theme(line_width=2mm));
mturk_plot = Gadfly.plot(l1, l2, 
        Guide.xlabel("MTurk Annotated Expression Scores (higher = more feminine)"), 
        Guide.ylabel("GCS (higher = more feminine)"), Guide.title("GCS vs Annotated Images"))
draw(PNG("images/mturk_plot.png", 20cm, 15cm), mturk_plot)
