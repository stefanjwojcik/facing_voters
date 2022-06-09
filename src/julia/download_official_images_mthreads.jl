# Purpose of this file is to download all image files from the official government record 
# for 2016 election in Brazil 

# TODO: 
# 1: create a directory in the github folder that will contain all the images for this study
# 2: deprecate all the old python code because it is no longer necessary - just use julia 
# 3: deprecate all the old csv's with the scraped data as they will no longer be used 

# USE HTTP 
using Images, HTTP, ImageMagick
using CSV, DataFrames, ProgressMeter
include("src/julia/utils.jl")

function get_official_images(row::DataFrameRow{DataFrame, DataFrames.Index})
    # getting race of candidates 
    # This pattern is: https://divulgacandcontas.tse.jus.br/candidaturas/oficial/2016/$SG_UF/2/$SG_UE/$SQ_CANDIDATO/$NR_TITULO_ELEITORAL_CANDIDATO.jpg
    # https://divulgacandcontas.tse.jus.br/candidaturas/oficial/2016/AC/01074/2/10000001123/006475182437.jpg
    # Limit to just mayor and city council elections 

    ## GET THE IMAGES 
    baselink = "https://divulgacandcontas.tse.jus.br/candidaturas/oficial/2016"
    local_path = ""
    # Join all information into the link 
    sg_uf = string(row.SG_UF)
    sg_ue = lpad(row.SG_UE, 5, "0")
    sq_ca = string(row.SQ_CANDIDATO)
    nr_tit = lpad(row.NR_TITULO_ELEITORAL_CANDIDATO, 12, "0" )
    link = join([baselink, sg_uf, sg_ue, "2", sq_ca, nr_tit], "/")
    if isfile(local_path*join([sg_uf, sg_ue, sq_ca, nr_tit], "-") * ".jpg")
        true
    else
        res = save_http_img(link * ".jpg", local_path*join([sg_uf, sg_ue, sq_ca, nr_tit], "-") * ".jpg")
        #append!(outdf, DataFrame(link = link * ".jpg", img_downloaded = (res == 200)))
        if res == 429
            println("Too many requests error, sleeping..")
            cntdown = 0
            while cntdown < 10 
                sleep(1)
                print("$cntdown sleeps...")
                cntdown += 1
            end
        end
    end
end


# Load directly into memory
#r = HTTP.get("https://divulgacandcontas.tse.jus.br/candidaturas/oficial/2016/AC/01040/2/10000001628/002001452445.jpg", hdr)
#buffer = IOBuffer(r.body)
#ImageMagick.load(buffer)

#r = HTTP.download("https://divulgacandcontas.tse.jus.br/candidaturas/oficial/2016/AC/01040/2/10000001628/002001452445.jpg", 
#    "/home/swojcik/Downloads/test.jpg", hdr)

# Source dataset 
df = CSV.read("/home/swojcik/github/facing_voters/data/official_local_election_data_2016.csv", DataFrame);
df = filter(r -> any(startswith.(["PREFEITO", "VEREADOR"], r.DS_CARGO)), df)

## USING ASYNC MAPPING!!! 
asyncmap(get_official_images, eachrow(df))

