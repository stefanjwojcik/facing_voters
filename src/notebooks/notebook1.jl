### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 21acdce2-ef98-11ed-3503-6982893a7b42
begin
using RCall, DataFrames, PlutoUI, HypertextLiteral
cd("/home/ubuntu/facing_voters/src/R")
end

# ╔═╡ ee87aa69-a95a-46e2-a635-2d3ce56870f9
# Function to show R Plots in line 
function Base.show(io::IO, ::MIME"image/png", p::RObject{VecSxp})
    (path, _) = mktemp()
    R"ggsave($path, plot=$p, device = 'png')"
    im = read(path)
    rm(path)
    write(io, im)
end

# ╔═╡ 63c2c72b-8a0b-42c0-a149-95fd7ff8d2a4
md"""
# Replication Notebook 1: Facing Voters (Wojcik and Mullenax)

This is Notebook 1 of 2. It contains the code to replicate the results in the paper.

The lines below load R libraries required for the analysis.
"""

# ╔═╡ a650399d-3f25-491e-973d-7560eb3bdd1e
# ╠═╡ show_logs = false
# Load R libraries 
R"""
packrat::packrat_mode()
library(dplyr)
library(ggplot2)
library(lme4)
library(readr)
library(stargazer)
library(effects)

library(readr)
setwd('/home/ubuntu/facing_voters')
offd = readr::read_csv('../data/data_for_regression.csv')
offd$race = factor(offd$DS_COR_RACA, labels = unique(offd$DS_COR_RACA)[c(2, 1, 3, 4, 5)])

print("DONE LOADING LIBRARIES AND DATA.")
"""

# ╔═╡ 457a77fa-200b-4f79-a5c8-8528e551dbc4
PlutoUI.TableOfContents(title="Table of Contents", indent=true)

# ╔═╡ b2c4f226-82a2-4593-9806-ff3a70c69ec8
md"""
## R Analyses
"""

# ╔═╡ 23177860-cb6e-4d52-a927-b4de84d0937f
md"""
### Figure 1
"""

# ╔═╡ fc5e206f-b935-460f-b212-f8f77c6dc44d
md"""
We load the official data, assign factor labels, and generate figure 1 from the paper. This figure shows the distribution of the estimated conformity scores. 
"""

# ╔═╡ 188a93d9-7a1b-4e69-be04-d14bd8ee1a57
# ╠═╡ show_logs = false
R"""

# HISTOGRAM 
p = ggplot(offd, aes(x=scaled_fem_body, fill=Gender)) + geom_histogram(aes(color=Gender), bins=100, alpha=0.5) + 
  geom_vline(aes(xintercept = .5), linetype = "dashed", alpha = .5) +
  facet_grid(~Gender) + 
  ylab("Density") + 
  xlab("Conformity Score") + theme_minimal()

"""

# ╔═╡ 43ce4bdc-5ceb-4c66-be2a-cdcb11c7c7ae
md"""
#### Interpretation of model accuracy
"""

# ╔═╡ bc929488-947f-482d-8148-f3cd8aa26383
# ╠═╡ show_logs = false
R"""
body_acc = offd %>% mutate(pred_gender = fem_body <= .5) %>% summarise( sum(pred_gender == (Gender=="Men"), na.rm=T)/n())
face_acc = offd %>% mutate(pred_gender = fem_face <= .5) %>% summarise( sum(pred_gender == (Gender=="Men"), na.rm=T)/n())
print("DONE")
"""

# ╔═╡ 9f0a1332-4eb8-4aa5-ab37-9c8351823cac
md"""
Accuracy of body conformity model is $(rcopy(R"round(body_acc[[1]], 2)")). 
Accuracy of face conformity model is $(rcopy(R"round(face_acc[[1]], 2)")).
"""

# ╔═╡ 1995bad9-294d-44eb-b95a-98a7bc36507f
md"""
#### Running Low Information Models and Computing Figures 
"""

# ╔═╡ 74115c3e-e901-4171-9fa5-d0716d7307ee
R"""
# Fitting these at the top, because the results trickle down. 
# LOW Information Elections: VEREADOR MODEL
# face model
mod_vereador_face = lmer(logpct ~ scaled_fem_face*Gender + race + NR_IDADE_DATA_POSSE + SG_PARTIDO + ST_REELEICAO + (1|str_CD_MUNICIPIO), data = subset(offd, DS_CARGO=="VEREADOR"))
# BODY MODEL - this is for the supporting information
mod_vereador_body = lmer(logpct ~ scaled_fem_body*Gender + race + NR_IDADE_DATA_POSSE + SG_PARTIDO + ST_REELEICAO + (1|str_CD_MUNICIPIO), data = subset(offd, DS_CARGO=="VEREADOR"))
print("DONE RUNNING MODELS")
"""

# ╔═╡ 462e2c83-dc37-4a9e-9a30-64d91f0bfe1d
md"""
### Table 2

Results of from Multilevel linear model of city council elections. Gender conformity is the key predictor.
"""

# ╔═╡ 72552c5a-b2c1-4b96-bc20-00e903298f36
# ╠═╡ show_logs = false
R"""
stargazer(mod_vereador_face, mod_vereador_body, title = "City Council Election Results", omit = "^SG_PARTIDO", star.cutoffs = c(0.05, 0.01, 0.001), type="text")
"""

# ╔═╡ 4086553a-d9e9-4092-addd-f331fcd463c2
md"""
### Figure 3

This graph is the estimated effect of GCS in high information elections. 
"""

# ╔═╡ a3c3f48f-ade7-4865-bfcd-4140f19e5a00
R"""
#  EFFECT MATRIX
effdf = as.data.frame(effects::effect("scaled_fem_body*Gender", mod_vereador_body, 
                             xlevels = list(scaled_fem_body = seq(0, 1, by=.01), Gender = c("Women", "Men") )))
#  THE BODY PLOT
g = ggplot() 
g = g + geom_ribbon(data = effdf[effdf$Gender=="Women", ], aes(ymin=exp(lower), ymax=exp(upper), x=scaled_fem_body, fill = "band"), alpha = 0.3, fill = "#456A83")
g = g + geom_line(data = effdf[effdf$Gender=="Women", ], aes(y = exp(fit), x=scaled_fem_body), alpha = 0.3, col = "#456A83")
g = g + geom_ribbon(data = effdf[effdf$Gender=="Men", ], aes(ymin=exp(lower), ymax=exp(upper), x=scaled_fem_body, fill = "band"), alpha = 0.3, fill = "#BF3B27")
g = g + geom_line(data = effdf[effdf$Gender=="Men", ], aes(y = exp(fit), x=scaled_fem_body), alpha = 0.3, col = "#BF3B27")
g = g + geom_rug(data = subset(offd, DS_CARGO=="VEREADOR"), aes(x = scaled_fem_body, y=exp(logpct), col= Gender)) 
g = g + xlab("Conformity Score (low to high)") + ylab("Est. Proportion of Vote") + ggtitle("Effect of GCS on Vote Share - City Council Elections") + ylim(0, .2)
"""

# ╔═╡ 2df11d09-76de-4a73-99a8-4b9feac97795
md"""
#### Section 5.3 - Low Information Model Interpretation
"""

# ╔═╡ cb764d09-7e3b-4985-ab04-0474dbe2f11a
begin 
	R"""
	# INTERPRETATION: VOTE CHANGE For Women
	# effect from middle to end
	amb_fem = exp(effdf$fit[which(effdf$scaled_fem_body==.5 & effdf$Gender=="Women")])
	fem_fem = exp(effdf$fit[which(effdf$scaled_fem_body==1 & effdf$Gender=="Women")])
	abschange1 = (fem_fem - amb_fem)
	relchange1 = ((fem_fem - amb_fem)/amb_fem)*100
	
	# effect from end to end
	amb_fem = exp(effdf$fit[which(effdf$scaled_fem_body==0 & effdf$Gender=="Women")])
	fem_fem = exp(effdf$fit[which(effdf$scaled_fem_body==1 & effdf$Gender=="Women")])
	abschange2 = (fem_fem - amb_fem)
	relchange2 = ((fem_fem - amb_fem)/amb_fem)*100
	
	# INTERPRETATION: VOTE CHANGE For Men
	# effect from middle to end
	amb_mas = exp(effdf$fit[which(effdf$scaled_fem_body==.5 & effdf$Gender=="Men")])
	mas_mas = exp(effdf$fit[which(effdf$scaled_fem_body==1 & effdf$Gender=="Men")])
	abschange3 = (mas_mas - amb_mas)
	relchange3 = ((mas_mas - amb_mas)/amb_mas)*100
	# effect from end to end
	amb_mas = exp(effdf$fit[which(effdf$scaled_fem_body==0 & effdf$Gender=="Men")])
	mas_mas = exp(effdf$fit[which(effdf$scaled_fem_body==1 & effdf$Gender=="Men")])
	abschange4 = (mas_mas - amb_mas)
	relchange4 = ((mas_mas - amb_mas)/amb_mas)*100
	
	print("DONE")
	"""
	abschange1 = rcopy(R"abschange1")
	relchange1 = rcopy(R"relchange1")
	abschange2 = rcopy(R"abschange2")
	relchange2 = rcopy(R"relchange2")
	abschange3 = rcopy(R"abschange3")
	relchange3 = rcopy(R"relchange3")
	abschange4 = rcopy(R"abschange4")
	relchange4 = rcopy(R"relchange4")
	print("DONE")
end

# ╔═╡ e6a1a5b9-dd8a-4960-b013-d1056597fb79
md"""

[Note: the paper reports end-to-end effects of 1.1 for women and 1.2 for men, but with proper rounding the effects are slightly higher at 1.2 for women and 1.1 for men.](#)

Women: \
Low information middle to end absolute effect is $(round(abschange1*100, digits=1)) %, relative effect is $(round(relchange1)) %.

Low information end to end absolute effect is $(round(abschange2*100, digits=1)) %, relative effect is not reported.

Men:\
Low information middle to end absolute effect is $(round(abschange3*100, digits=1)) %, relative effect is $(round(relchange3)) %.

Low information end to end absolute effect is $(round(abschange4*100, digits=1)), relative effect is not reported.


"""

# ╔═╡ 851fc399-845c-4ae0-951c-3113647655a4
md"""
#### Running High Information Models
"""

# ╔═╡ 68eb5b7b-2841-4c88-9f7a-014677dd9f60
md"""
### Table 3
"""

# ╔═╡ ed605c18-5ae7-4521-ae33-e570febe4f5a
# ╠═╡ show_logs = false
R"""
# HIGH Information Elections: MAYORAL MODEL
# face model
# face model
mod_pref_face = lmer(logpct ~ scaled_fem_face*Gender + race + NR_IDADE_DATA_POSSE + SG_PARTIDO + ST_REELEICAO + (1|str_CD_MUNICIPIO), data = subset(offd, DS_CARGO=="PREFEITO"))
# BODY MODEL - this is for the supporting information
mod_pref_body = lmer(logpct ~ scaled_fem_body*Gender + race + NR_IDADE_DATA_POSSE + SG_PARTIDO + ST_REELEICAO + (1|str_CD_MUNICIPIO), data = subset(offd, DS_CARGO=="PREFEITO"))
# TABLE - for main text
class(mod_pref_body) <- "lmerMod"
class(mod_pref_face) <- "lmerMod"
stargazer(mod_pref_face, mod_pref_body, title = "Mayoral Election Results", omit = "^SG_PARTIDO", star.cutoffs = c(0.05, 0.01, 0.001), type="text")
"""

# ╔═╡ 4639b021-e1f8-4318-bf07-ad548a735831
md"""
### Figure 4
"""

# ╔═╡ 53164a71-688c-479c-886e-bd7ed82f395a
R"""
effdf = as.data.frame(effects::effect("scaled_fem_body*Gender", mod_pref_body, 
                             xlevels = list(scaled_fem_body = seq(0, 1, by=.01), Gender = c("Men", "Women") ) ))
# CREATE THE ACTUAL PLOT
g = ggplot() 
g = g + geom_ribbon(data = effdf[effdf$Gender=="Women", ], aes(ymin=exp(lower), ymax=exp(upper), x=scaled_fem_body, fill = "band"), alpha = 0.3, fill = "#456A83")
g = g + geom_line(data = effdf[effdf$Gender=="Women", ], aes(y = exp(fit), x=scaled_fem_body), alpha = 0.3, col = "#456A83")
g = g + geom_ribbon(data = effdf[effdf$Gender=="Men", ], aes(ymin=exp(lower), ymax=exp(upper), x=scaled_fem_body, fill = "band"), alpha = 0.3, fill = "#BF3B27")
g = g + geom_line(data = effdf[effdf$Gender=="Men", ], aes(y = exp(fit), x=scaled_fem_body), alpha = 0.3, col = "#BF3B27")
g = g + geom_rug(data = offd, aes(x = scaled_fem_body, y=exp(logpct), col= Gender)) #+ ylim(0, .75)
g = g + xlab("Conformity Score (low to high)") + ylab("Est. Proportion of Vote") + ggtitle("Effect of GCS on Vote Share - Mayoral Elections")

"""


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
RCall = "6f49c342-dc21-5d91-9882-a32aef131414"

[compat]
DataFrames = "~1.5.0"
HypertextLiteral = "~0.9.4"
PlutoUI = "~0.7.51"
RCall = "~0.13.15"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "71442fb20c21099422578e9fd59a5c955a79ad2e"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "f84967c4497e0e1955f9a582c232b02847c5f589"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.7"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "e32a90da027ca45d84678b826fffd3110bb3fc90"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.8.0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "84204eae2dd237500835990bcade263e27674a93"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.16"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "6667aadd1cdee2c6cd068128b3d226ebc4fb0c67"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.9"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7302075e5e06da7d000d9bfa055013e3e85578ca"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.9"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "259e206946c293698122f63e2b513a7c99a244e8"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "213579618ec1f42dea7dd637a42785a608b1ea9c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.RCall]]
deps = ["CategoricalArrays", "Conda", "DataFrames", "DataStructures", "Dates", "Libdl", "Missings", "REPL", "Random", "Requires", "StatsModels", "WinReg"]
git-tree-sha1 = "d441bdeea943f8e8f293e0e3a78fe2d7c3aa24e6"
uuid = "6f49c342-dc21-5d91-9882-a32aef131414"
version = "0.13.15"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "77d3c4726515dca71f6d80fbb5e251088defe305"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.18"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "8cc7a5385ecaa420f0b3426f9b0135d0df0638ed"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.2"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.WinReg]]
git-tree-sha1 = "cd910906b099402bcc50b3eafa9634244e5ec83b"
uuid = "1b915085-20d7-51cf-bf83-8f477d6f5128"
version = "1.0.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═ee87aa69-a95a-46e2-a635-2d3ce56870f9
# ╠═21acdce2-ef98-11ed-3503-6982893a7b42
# ╟─63c2c72b-8a0b-42c0-a149-95fd7ff8d2a4
# ╠═a650399d-3f25-491e-973d-7560eb3bdd1e
# ╟─457a77fa-200b-4f79-a5c8-8528e551dbc4
# ╟─b2c4f226-82a2-4593-9806-ff3a70c69ec8
# ╟─23177860-cb6e-4d52-a927-b4de84d0937f
# ╟─fc5e206f-b935-460f-b212-f8f77c6dc44d
# ╠═188a93d9-7a1b-4e69-be04-d14bd8ee1a57
# ╠═43ce4bdc-5ceb-4c66-be2a-cdcb11c7c7ae
# ╠═bc929488-947f-482d-8148-f3cd8aa26383
# ╠═9f0a1332-4eb8-4aa5-ab37-9c8351823cac
# ╠═1995bad9-294d-44eb-b95a-98a7bc36507f
# ╠═74115c3e-e901-4171-9fa5-d0716d7307ee
# ╠═462e2c83-dc37-4a9e-9a30-64d91f0bfe1d
# ╠═72552c5a-b2c1-4b96-bc20-00e903298f36
# ╠═4086553a-d9e9-4092-addd-f331fcd463c2
# ╠═a3c3f48f-ade7-4865-bfcd-4140f19e5a00
# ╠═2df11d09-76de-4a73-99a8-4b9feac97795
# ╠═cb764d09-7e3b-4985-ab04-0474dbe2f11a
# ╠═e6a1a5b9-dd8a-4960-b013-d1056597fb79
# ╠═851fc399-845c-4ae0-951c-3113647655a4
# ╟─68eb5b7b-2841-4c88-9f7a-014677dd9f60
# ╠═ed605c18-5ae7-4521-ae33-e570febe4f5a
# ╟─4639b021-e1f8-4318-bf07-ad548a735831
# ╠═53164a71-688c-479c-886e-bd7ed82f395a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
