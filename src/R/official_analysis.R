# OFFICIAL Analysis: 6/2/2022
library(dplyr)
library(ggplot2)
library(lme4)
# load the official data 
offd = read_csv("data/official_local_election_data_2016.csv")

# load the body masculinity estimates 
masc = read_csv("data/masc_body_official.csv")

# join them together 


# LOW Information Elections: VEREADOR MODEL
# face model
mod_vereador_face = lmer(log_pct ~ face_masc*gender + age + Partido + (1|`Município`), data = subset(df, race=="City Council"))
# BODY MODEL - this is for the supporting information
mod_vereador_body = lmer(log_pct ~ body_masc*gender + age + Partido + (1|`Município`), data = subset(df, race=="City Council"))
# TABLE OF BODY MODEL - FOR SUPPORTING INFORMATION
stargazer(mod_vereador_face, mod_vereador_body, title = "City Council Election Results", omit = "^Partido")
# predictions
#  EFFECT MATRIX
effdf = as.data.frame(effect("body_masc*gender", mod_vereador_body, 
                             xlevels = list(body_masc = seq(0, 1, by=.01), gender = "Female") ))
#  THE BODY PLOT
g = ggplot() 
g = g + geom_ribbon(data = effdf[effdf$gender=="Woman", ], aes(ymin=exp(lower), ymax=exp(upper), x=body_masc, fill = "band"), alpha = 0.3, fill = "#456A83")
g = g + geom_line(data = effdf[effdf$gender=="Woman", ], aes(y = exp(fit), x=body_masc), alpha = 0.3, col = "#456A83")
g = g + geom_ribbon(data = effdf[effdf$gender=="Man", ], aes(ymin=exp(lower), ymax=exp(upper), x=body_masc, fill = "band"), alpha = 0.3, fill = "#BF3B27")
g = g + geom_line(data = effdf[effdf$gender=="Man", ], aes(y = exp(fit), x=body_masc), alpha = 0.3, col = "#BF3B27")
g = g + geom_rug(data = subset(df, race=="City Council"), aes(x = body_masc, y=exp(log_pct), col= gender)) + ylim(0, 1)
g = g + xlab("GCS (body)") + ylab("Est. Percent of Vote") + ggtitle("Effect of GCS on Vote Share")
g

# INTERPRETATION: VOTE CHANGE For Women
# effect from middle to end
amb_fem = exp(effdf$fit[which(effdf$body_masc==.5 & effdf$gender=="Woman")])
fem_fem = exp(effdf$fit[which(effdf$body_masc==0 & effdf$gender=="Woman")])
#(fem_fem - amb_fem)
((fem_fem - amb_fem)/amb_fem)*100
# effect from end to end
amb_fem = exp(effdf$fit[which(effdf$body_masc==1 & effdf$gender=="Woman")])
fem_fem = exp(effdf$fit[which(effdf$body_masc==0 & effdf$gender=="Woman")])
#(fem_fem - amb_fem)
((fem_fem - amb_fem)/amb_fem)*100

# INTERPRETATION: VOTE CHANGE For Men
# effect from middle to end
amb_mas = exp(effdf$fit[which(effdf$body_masc==.5 & effdf$gender=="Man")])
mas_mas = exp(effdf$fit[which(effdf$body_masc==1 & effdf$gender=="Man")])
#(mas_mas - amb_mas)
((mas_mas - amb_mas)/amb_mas)*100
# effect from end to end
amb_mas = exp(effdf$fit[which(effdf$body_masc==0 & effdf$gender=="Man")])
mas_mas = exp(effdf$fit[which(effdf$body_masc==1 & effdf$gender=="Man")])
#(mas_mas - amb_mas)
((mas_mas - amb_mas)/amb_mas)*100

## HIGH-INFORMATION MAIN MODELS

# HIGH Information Elections: MAYORAL MODEL
# face model
mod_pref_face = lmer(log_pct ~ face_masc*gender + age + Partido + (1|`Município`), data = subset(df, race=="Mayor"))
mod_pref_body = lmer(log_pct ~ body_masc*gender + age + Partido + (1|`Município`), data = subset(df, race=="Mayor"))
# TABLE - for main text
class(mod_pref_body) <- "lmerMod"
class(mod_pref_face) <- "lmerMod"
stargazer(mod_pref_face, mod_pref_body, title = "Mayoral Election Results", omit = "^Partido")
# predictions
# CREATE EFFECT MATRIX
effdf = as.data.frame(effect("body_masc*gender", mod_pref_body, 
                             xlevels = list(body_masc = seq(0, 1, by=.01), gender = "Woman") ))
# CREATE THE ACTUAL PLOT
g = ggplot() 
g = g + geom_ribbon(data = effdf[effdf$gender=="Woman", ], aes(ymin=exp(lower), ymax=exp(upper), x=body_masc, fill = "band"), alpha = 0.3, fill = "#456A83")
g = g + geom_line(data = effdf[effdf$gender=="Woman", ], aes(y = exp(fit), x=body_masc), alpha = 0.3, col = "#456A83")
g = g + geom_ribbon(data = effdf[effdf$gender=="Man", ], aes(ymin=exp(lower), ymax=exp(upper), x=body_masc, fill = "band"), alpha = 0.3, fill = "#BF3B27")
g = g + geom_line(data = effdf[effdf$gender=="Man", ], aes(y = exp(fit), x=body_masc), alpha = 0.3, col = "#BF3B27")
g = g + geom_rug(data = df, aes(x = body_masc, y=exp(log_pct), col= gender)) #+ ylim(0, .75)
g = g + xlab("GCS (body)") + ylab("Est. Percent of Vote") + ggtitle("Effect of GCS on Vote Share")
g

# INTERPRETATION: VOTE CHANGE For Women
# effect from middle to end
amb_fem = exp(effdf$fit[which(effdf$body_masc==.5 & effdf$gender=="Woman")])
fem_fem = exp(effdf$fit[which(effdf$body_masc==0 & effdf$gender=="Woman")])
#(fem_fem - amb_fem)
((fem_fem - amb_fem)/amb_fem)*100
# effect from end to end
amb_fem = exp(effdf$fit[which(effdf$body_masc==1 & effdf$gender=="Woman")])
fem_fem = exp(effdf$fit[which(effdf$body_masc==0 & effdf$gender=="Woman")])
#(fem_fem - amb_fem)
((fem_fem - amb_fem)/amb_fem)*100

# INTERPRETATION: VOTE CHANGE For Men
# effect from middle to end
amb_mas = exp(effdf$fit[which(effdf$body_masc==.5 & effdf$gender=="Man")])
mas_mas = exp(effdf$fit[which(effdf$body_masc==1 & effdf$gender=="Man")])
#(mas_mas - amb_mas)
((mas_mas - amb_mas)/amb_mas)*100
# effect from end to end
amb_mas = exp(effdf$fit[which(effdf$body_masc==0 & effdf$gender=="Man")])
mas_mas = exp(effdf$fit[which(effdf$body_masc==1 & effdf$gender=="Man")])
#(mas_mas - amb_mas)
((mas_mas - amb_mas)/amb_mas)*100

