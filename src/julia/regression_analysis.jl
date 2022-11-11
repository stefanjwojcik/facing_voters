# Produces the data for the regression analysis: "data_for_regression.csv"
# Draws on official election data, candidate data, and GCS scores 

using CSV, DataFrames, MixedModels, Effects, DataStructures 
include("src/julia/utils.jl")

# load the official biographical data 
# TODO: generate vote % by municipality 

function load_analysis_data()
    offd = CSV.read("data/official_local_election_data_2016.csv", DataFrame) |> 
        (data -> filter(:NR_TURNO => x -> x == 1, data)) |> 
        (data -> filter(:DS_ELEICAO => x -> x == "Eleições Municipais 2016", data))
    offd.imglink = map(generate_img_link, eachrow(offd))
    offd.downloaded = isfile.("images/official_images_data/" .* offd.imglink)
    offd = filter(row -> row.downloaded == 1, offd)

    # load the vote totals 
    votes = CSV.read("data/votacao_candidato_munzona_2016_BRASIL.csv", DataFrame) |> 
        (data -> filter(:NR_TURNO => x -> x == 1, data)) |> 
        (data -> filter(:NM_TIPO_ELEICAO => x -> contains(x, r"Ord"), data)) |>
        (data -> select(data, [:NR_CANDIDATO, :CD_MUNICIPIO, :SG_UF, :SG_UE, :SQ_CANDIDATO, :QT_VOTOS_NOMINAIS ])) |> 
        (data -> groupby(data, [:NR_CANDIDATO, :CD_MUNICIPIO, :SG_UF, :SG_UE, :SQ_CANDIDATO])) |> 
        (data -> combine(data, :QT_VOTOS_NOMINAIS => sum, renamecols=false)) |> #cand vote share
        (data -> groupby(data, :CD_MUNICIPIO)) |> 
        (data -> combine(data, :, :QT_VOTOS_NOMINAIS => sum => :QT_TODO_VOTOS))
    # joining votes and biographical data
    od = leftjoin(offd, votes, on = [:NR_CANDIDATO => :NR_CANDIDATO, 
                                    :SQ_CANDIDATO => :SQ_CANDIDATO, 
                                    :SG_UF => :SG_UF, 
                                    :SG_UE => :SG_UE])

    # load the expression data 
    expd = CSV.read("data/fem_body_officialV2.csv", DataFrame) |> 
        (data -> leftjoin(data, CSV.read("data/fem_face_officialV2.csv", DataFrame), on = [:imglink => :imglink]))

    # join the two datasets together 
    df = leftjoin(expd, od, on = [:imglink => :imglink]) |> 
        (data -> combine(data, :, :fem_face => x -> Float64.(x), renamecols=false)) |> 
        (data -> select(data, Not(:Column1)))
    # OUTCOME VARIABLE 
    df.logpct = log10.( (df.QT_VOTOS_NOMINAIS ./ df.QT_TODO_VOTOS) .+ 1e-5)
    df.str_CD_MUNICIPIO = string.(df.CD_MUNICIPIO)
    df.Gender = @. ifelse(df.DS_GENERO=="FEMININO", "Women", "Men")

    # Modeling 
    #fm = @formula(logpct ~ masc_body * DS_GENERO + DS_COR_RACA + NR_IDADE_DATA_POSSE + SG_PARTIDO + ST_REELEICAO + (1|str_CD_MUNICIPIO))
    #fm1 = fit(MixedModel, fm, df)
    return(df)
end

df = load_analysis_data()

# CREATE SCALED SCORE 
df.scaled_fem_body = @. ((df.DS_GENERO=="FEMININO")*df.fem_body + (df.DS_GENERO=="MASCULINO")*(1-df.fem_body))
df.scaled_fem_face = @. ((df.DS_GENERO=="FEMININO")*df.fem_face + (df.DS_GENERO=="MASCULINO")*(1-df.fem_face))

# PLOTTING HISTOGRAM OF THE GCS SCORE 
#Scale.discrete_color_manual("blue","purple"),
Gadfly.plot(df[rand(1:nrow(df), 20000), :], x=:scaled_fem_body, color=:Gender, Guide.xlabel("Conformity Score"), Geom.histogram)

############## MODELING 

#mod_vereador_face = R"lmer(log_pct ~ face_masc*gender + age + Partido + (1|`Município`), data = subset(df, race=="City Council"))"

fm = @formula(logpct ~ scaled_fem_body * DS_GENERO + DS_COR_RACA + NR_IDADE_DATA_POSSE + SG_PARTIDO + ST_REELEICAO + (1|str_CD_MUNICIPIO))
fm1 = fit(MixedModel, fm, df)
refgrid = similar(df[1:1000, :]) |> (data -> select(data, [:scaled_fem_body, :DS_GENERO, :DS_COR_RACA, :NR_IDADE_DATA_POSSE, :SG_PARTIDO, :ST_REELEICAO]))
# Make reference grid for prediction 
refgrid.masc_body .= [collect(0:.002:1)[1:end-1]; collect(0:.002:1)[1:end-1]]
refgrid.DS_GENERO .= repeat(unique(df.DS_GENERO), inner=500)
refgrid.DS_COR_RACA .= "BRANCA"
refgrid.NR_IDADE_DATA_POSSE .= Int64(median(df.NR_IDADE_DATA_POSSE))
partycount = counter(df.SG_PARTIDO)
refgrid.SG_PARTIDO .= collect(keys(partycount))[argmax(collect(values(partycount)))]
refgrid.ST_REELEICAO .= "N"

effects!(refgrid, fm1)

refgrid[!, :lower] = @. exp(refgrid.logpct - 1.96 * refgrid.err)
refgrid[!, :upper] = @. exp(refgrid.logpct + 1.96 * refgrid.err)
refgrid[!, :pct_vote] = @. exp(refgrid.logpct)
sort!(refgrid, [:masc_body])

p = plot(refgrid, x=:masc_body, y=:logpct, ymin=:lower, ymax=:upper, color=:DS_GENERO, Geom.line, Geom.ribbon)