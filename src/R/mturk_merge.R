setwd("~/Documents/facing voters paper/") # This is where the data are currently living 
library(stringi)
library(caret)
library(lme4)
library(effects)
library(tidyverse)
library(ggplot2)
library(lubridate)

################:::
# Get official data:
#df$Nome.para.Urna = tolower(df$NM_URNA_CANDIDATO)

# Get full data:: PREPROCESSING
df_img = read.csv("data/Brazil 2016 Candidate Data/brazil_2016_candidate_data.csv", stringsAsFactors = F)
df_img = df_img %>%
  dplyr::select(-starts_with("Unnamed")) %>%
  mutate(name = basename(as.character(url)), Nome.para.Urna = tolower(Nome.para.Urna))

### THE ROUND 2 VOTES WERE SWAPPED WITH ROUND 1 VOTES DURING SCRAPING - FIXED HERE 
df_img = df_img %>%
  mutate(round_1_votes = as.numeric(gsub(",", "", round_1_votes))) %>%
  mutate(round_2_votes = as.numeric(gsub(",", "", round_2_votes))) %>%
  mutate(round_1_pct_rev = ifelse(is.na(round_2_pct), round_1_pct, round_2_pct), 
         round_1_votes_rev = ifelse(is.na(round_2_votes), round_1_votes, round_2_votes), 
         round_2_pct_rev = ifelse(is.na(round_2_pct), round_2_pct, round_1_pct), 
         round_2_votes_rev = ifelse(is.na(round_2_votes), round_2_votes, round_1_votes))

# currently not functioning as a merge w/ canonical file
#df_all = merge(df, df_img, by = "Nome.para.Urna", all.x=T)

# Load the face and full-body masculinity scores
df_face_only = read.csv("data/Brazil 2016 Candidate Data/df_face_only_masc_scores.csv")
df_body = read.csv("data/Brazil 2016 Candidate Data/df_full_body_masc_scores.csv")
df_measure = data.frame(face_masc = df_face_only$apparent_mas, body_masc = df_body$apparent_mas)

dfnames = read.csv("data/Brazil 2016 Candidate Data/image_names_full.csv")
df_measure = df_measure %>%
  mutate(gender = stri_extract(dfnames$names, regex = c("Female|Male"))) %>%
  mutate(gender = case_when(gender == "Male" ~ "Man", gender=="Female" ~ "Woman")) %>%
  mutate(name = as.character(dfnames$names)) %>%
  mutate(name = gsub("^age[0-9]{2}_.*_", "",  name)) %>%
  mutate(name = gsub("^age[0-9]{4}_.*_", "",  name)) %>%
  mutate(name = gsub("-d.jpg", "",  name)) %>%
  mutate(imglink = dfnames$name) %>%
  dplyr::select(face_masc, body_masc, gender, name, imglink)

# MERGE
df = merge(df_img, df_measure, by = "name")
df$gender = as.factor(df$gender)

# Code Vote Share, Age
df = df %>%
  mutate(log_pct = log10(round_1_pct)) %>%
  mutate(log_pct = ifelse(is.infinite(log_pct), NA, log_pct)) %>%
  mutate(birth_year = year(as.Date(Nascimento, format = "%d/%m/%Y"))) %>%
  mutate(birth_year = case_when(birth_year==970 ~ 1970, 
                                birth_year < 100 ~ birth_year + 1900,
                                birth_year > 1900 ~ birth_year)) %>%
  mutate(age = year(Sys.Date()) - birth_year) %>%
  mutate(race = case_when( Cargo.a.que.Concorre == "Prefeita" ~ "Mayor", 
                           Cargo.a.que.Concorre == "Prefeito" ~ "Mayor", 
                           Cargo.a.que.Concorre == "Vereador" ~ "City Council", 
                           Cargo.a.que.Concorre == "Vereadora" ~ "City Council")) %>%
  filter(!is.na(gender)) %>% # ONE missing gender variable
  mutate(Partido = factor(Partido)) %>%
  filter(Município != "") %>% 
  dplyr::select(log_pct, round_1_pct, face_masc, body_masc, gender, age, race, Partido, `Município`, round_1_result) %>%
  na.omit()

# select for key merging with other dataset 

dfout = df %>% select(imglink, name, Número)

write.csv(dfout, "~/github/facing_voters/data/df_name_map.csv")