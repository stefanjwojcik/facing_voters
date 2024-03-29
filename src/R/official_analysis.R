# OFFICIAL Analysis: 6/2/2022

library(dplyr)
library(ggplot2)
library(lme4)
library(readr)
library(stargazer)
library(effects)
#library(JuliaCall)
#julia <- julia_setup()

# load the official data 
offd = read_csv("../../data/data_for_regression.csv")
offd$race = factor(offd$DS_COR_RACA, labels = unique(offd$DS_COR_RACA)[c(2, 1, 3, 4, 5)])

offd %>% mutate(pred_gender = fem_body <= .5) %>% summarise( sum(pred_gender == (Gender=="Men"), na.rm=T)/n())
offd %>% mutate(pred_gender = fem_face <= .5) %>% summarise( sum(pred_gender == (Gender=="Men"), na.rm=T)/n())


# HISTOGRAM 
p = ggplot(offd, aes(x=scaled_fem_body, fill=Gender)) + geom_histogram(aes(color=Gender), bins=100, alpha=0.5) + 
  geom_vline(aes(xintercept = .5), linetype = "dashed", alpha = .5) +
  facet_grid(~Gender) + 
  ylab("Density") + 
  xlab("Conformity Score") + theme_minimal()

# LOW Information Elections: VEREADOR MODEL
# face model
mod_vereador_face = lmer(logpct ~ scaled_fem_face*Gender + race + NR_IDADE_DATA_POSSE + SG_PARTIDO + ST_REELEICAO + (1|str_CD_MUNICIPIO), data = subset(offd, DS_CARGO=="VEREADOR"))
# BODY MODEL - this is for the supporting information
mod_vereador_body = lmer(logpct ~ scaled_fem_body*Gender + race + NR_IDADE_DATA_POSSE + SG_PARTIDO + ST_REELEICAO + (1|str_CD_MUNICIPIO), data = subset(offd, DS_CARGO=="VEREADOR"))
# TABLE OF BODY MODEL - FOR SUPPORTING INFORMATION
stargazer(mod_vereador_face, mod_vereador_body, title = "City Council Election Results", omit = "^SG_PARTIDO", star.cutoffs = c(0.05, 0.01, 0.001))
# predictions
#  EFFECT MATRIX
effdf = as.data.frame(effect("scaled_fem_body*Gender", mod_vereador_body, 
                             xlevels = list(scaled_fem_body = seq(0, 1, by=.01), Gender = c("Women", "Men") )))
#  THE BODY PLOT
g = ggplot() 
g = g + geom_ribbon(data = effdf[effdf$Gender=="Women", ], aes(ymin=exp(lower), ymax=exp(upper), x=scaled_fem_body, fill = "band"), alpha = 0.3, fill = "#456A83")
g = g + geom_line(data = effdf[effdf$Gender=="Women", ], aes(y = exp(fit), x=scaled_fem_body), alpha = 0.3, col = "#456A83")
g = g + geom_ribbon(data = effdf[effdf$Gender=="Men", ], aes(ymin=exp(lower), ymax=exp(upper), x=scaled_fem_body, fill = "band"), alpha = 0.3, fill = "#BF3B27")
g = g + geom_line(data = effdf[effdf$Gender=="Men", ], aes(y = exp(fit), x=scaled_fem_body), alpha = 0.3, col = "#BF3B27")
g = g + geom_rug(data = subset(offd, DS_CARGO=="VEREADOR"), aes(x = scaled_fem_body, y=exp(logpct), col= Gender)) 
g = g + xlab("Conformity Score (low to high)") + ylab("Est. Proportion of Vote") #+ ggtitle("Effect of GCS on Vote Share - City Council Elections") 
g = g + ylim(0, .2)
g + theme_minimal()

# INTERPRETATION: VOTE CHANGE For Women
# effect from middle to end
amb_fem = exp(effdf$fit[which(effdf$scaled_fem_body==.5 & effdf$Gender=="Women")])
fem_fem = exp(effdf$fit[which(effdf$scaled_fem_body==1 & effdf$Gender=="Women")])
#(fem_fem - amb_fem)
((fem_fem - amb_fem)/amb_fem)*100
# effect from end to end
amb_fem = exp(effdf$fit[which(effdf$scaled_fem_body==0 & effdf$Gender=="Women")])
fem_fem = exp(effdf$fit[which(effdf$scaled_fem_body==1 & effdf$Gender=="Women")])
#(fem_fem - amb_fem)
((fem_fem - amb_fem)/amb_fem)*100

# INTERPRETATION: VOTE CHANGE For Men
# effect from middle to end
amb_mas = exp(effdf$fit[which(effdf$scaled_fem_body==.5 & effdf$Gender=="Men")])
mas_mas = exp(effdf$fit[which(effdf$scaled_fem_body==1 & effdf$Gender=="Men")])
#(mas_mas - amb_mas)
((mas_mas - amb_mas)/amb_mas)*100
# effect from end to end
amb_mas = exp(effdf$fit[which(effdf$scaled_fem_body==0 & effdf$Gender=="Men")])
mas_mas = exp(effdf$fit[which(effdf$scaled_fem_body==1 & effdf$Gender=="Men")])
#(mas_mas - amb_mas)
((mas_mas - amb_mas)/amb_mas)*100

## HIGH-INFORMATION MAIN MODELS

# HIGH Information Elections: MAYORAL MODEL
# face model
# face model
mod_pref_face = lmer(logpct ~ scaled_fem_face*Gender + race + NR_IDADE_DATA_POSSE + SG_PARTIDO + ST_REELEICAO + (1|str_CD_MUNICIPIO), data = subset(offd, DS_CARGO=="PREFEITO"))
# BODY MODEL - this is for the supporting information
mod_pref_body = lmer(logpct ~ scaled_fem_body*Gender + race + NR_IDADE_DATA_POSSE + SG_PARTIDO + ST_REELEICAO + (1|str_CD_MUNICIPIO), data = subset(offd, DS_CARGO=="PREFEITO"))
# TABLE - for main text
class(mod_pref_body) <- "lmerMod"
class(mod_pref_face) <- "lmerMod"
stargazer(mod_pref_face, mod_pref_body, title = "Mayoral Election Results", omit = "^SG_PARTIDO", star.cutoffs = c(0.05, 0.01, 0.001))
# predictions
# CREATE EFFECT MATRIX
effdf = as.data.frame(effect("scaled_fem_body*Gender", mod_pref_body, 
                             xlevels = list(scaled_fem_body = seq(0, 1, by=.01), Gender = c("Men", "Women") ) ))
# CREATE THE ACTUAL PLOT
g = ggplot() 
g = g + geom_ribbon(data = effdf[effdf$Gender=="Women", ], aes(ymin=exp(lower), ymax=exp(upper), x=scaled_fem_body, fill = "band"), alpha = 0.3, fill = "#456A83")
g = g + geom_line(data = effdf[effdf$Gender=="Women", ], aes(y = exp(fit), x=scaled_fem_body), alpha = 0.3, col = "#456A83")
g = g + geom_ribbon(data = effdf[effdf$Gender=="Men", ], aes(ymin=exp(lower), ymax=exp(upper), x=scaled_fem_body, fill = "band"), alpha = 0.3, fill = "#BF3B27")
g = g + geom_line(data = effdf[effdf$Gender=="Men", ], aes(y = exp(fit), x=scaled_fem_body), alpha = 0.3, col = "#BF3B27")
g = g + geom_rug(data = offd, aes(x = scaled_fem_body, y=exp(logpct), col= Gender)) #+ ylim(0, .75)
g = g + xlab("Conformity Score (low to high)") + ylab("Est. Proportion of Vote") #+ ggtitle("Effect of GCS on Vote Share - Mayoral Elections")
g + theme_minimal()

# INTERPRETATION: VOTE CHANGE For Women
# effect from middle to end
amb_fem = exp(effdf$fit[which(effdf$scaled_fem_body==.5 & effdf$Gender=="Women")])
fem_fem = exp(effdf$fit[which(effdf$scaled_fem_body==1 & effdf$Gender=="Women")])
#(fem_fem - amb_fem)
((fem_fem - amb_fem)/amb_fem)*100
# effect from end to end
amb_fem = exp(effdf$fit[which(effdf$scaled_fem_body==0 & effdf$Gender=="Women")])
fem_fem = exp(effdf$fit[which(effdf$scaled_fem_body==1 & effdf$Gender=="Women")])
#(fem_fem - amb_fem)
((fem_fem - amb_fem)/amb_fem)*100

# INTERPRETATION: VOTE CHANGE For Men
# effect from middle to end
amb_mas = exp(effdf$fit[which(effdf$scaled_fem_body==.5 & effdf$Gender=="Men")])
mas_mas = exp(effdf$fit[which(effdf$scaled_fem_body==1 & effdf$Gender=="Men")])
#(mas_mas - amb_mas)
((mas_mas - amb_mas)/amb_mas)*100
# effect from end to end
amb_mas = exp(effdf$fit[which(effdf$scaled_fem_body==0 & effdf$Gender=="Men")])
mas_mas = exp(effdf$fit[which(effdf$scaled_fem_body==1 & effdf$Gender=="Men")])
#(mas_mas - amb_mas)
((mas_mas - amb_mas)/amb_mas)*100

################ 
# MODELS WITH EDUCATION INCLUDED
############################

mod_vereador_body_edu = lmer(logpct ~ scaled_fem_body*Gender + race + NR_IDADE_DATA_POSSE + DS_GRAU_INSTRUCAO +  SG_PARTIDO + ST_REELEICAO + (1|str_CD_MUNICIPIO), data = subset(offd, DS_CARGO=="VEREADOR"))

mod_pref_body_edu = lmer(logpct ~ scaled_fem_body*Gender + race + NR_IDADE_DATA_POSSE + DS_GRAU_INSTRUCAO+ SG_PARTIDO + ST_REELEICAO + (1|str_CD_MUNICIPIO), data = subset(offd, DS_CARGO=="PREFEITO"))

stargazer(mod_vereador_body_edu, mod_pref_body_edu, title = "Education Models", omit = "^SG_PARTIDO", star.cutoffs = c(0.05, 0.01, 0.001))

###### TABLES OF GCS SCORES BY RACE 

race_refactor = function(x) (case_when( x == "BRANCA" ~ "White", 
                                        x == "PARDA" ~ "Mixed", 
                                        x == "PRETA" ~ "Black", 
                                        x == "INDÍGENA" ~ "Indigenous", 
                                        x == "AMARELA" ~ "Asian"))
# GRAPH OF GCS BY RACE AND GENDER
offd %>%
  group_by(race, Gender) %>%
  summarise(GCS = mean(fem_body)) %>%
  ungroup() %>%
  mutate_at(c("race"), race_refactor) %>%
  mutate(race = fct_reorder(race, desc(GCS))) %>%
  ggplot(aes(x=race, y=GCS, col=Gender )) + geom_point(size=2) + geom_line(aes(group=race), col="dark grey") +
  theme_minimal() + ylim(0, 1) + xlab("Self-identified Candidate Race") + 
  geom_text(aes(label=round(GCS, digits=2)),hjust=0,vjust=0)


