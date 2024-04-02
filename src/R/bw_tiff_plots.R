
colorBlindBlack8  <- c("#000000", "#E69F00", "#56B4E9", "#009E73", 
                                "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
gender_colors <- c("#E69F00", "#000000")  # Use light and dark shades of gray

p <- ggplot(offd, aes(x = scaled_fem_body, fill = Gender)) +
  geom_histogram(aes(color = Gender), bins = 100, alpha = 0.5) +
  geom_vline(aes(xintercept = 0.5), linetype = "dashed", alpha = 0.5) +
  facet_grid(~Gender) +
  ylab("Density") +
  xlab("Conformity Score") +
  theme_minimal() +
  scale_color_manual(values = gender_colors) + # Set the fill colors
  scale_fill_manual(values = gender_colors)

#  THE BODY PLOT
g = ggplot() 
g = g + geom_ribbon(data = effdf[effdf$Gender=="Women", ], aes(ymin=exp(lower), ymax=exp(upper), x=scaled_fem_body, fill = "band"), alpha = 0.3, fill = "#000000")
g = g + geom_line(data = effdf[effdf$Gender=="Women", ], aes(y = exp(fit), x=scaled_fem_body), alpha = 0.3, col = "#000000")
g = g + geom_ribbon(data = effdf[effdf$Gender=="Men", ], aes(ymin=exp(lower), ymax=exp(upper), x=scaled_fem_body, fill = "band"), alpha = 0.3, fill = "#E69F00")
g = g + geom_line(data = effdf[effdf$Gender=="Men", ], aes(y = exp(fit), x=scaled_fem_body), alpha = 0.3, col = "#E69F00")
g = g + geom_rug(data = subset(offd, DS_CARGO=="VEREADOR"), aes(x = scaled_fem_body, y=exp(logpct), col= Gender)) 
g = g + xlab("Conformity Score (low to high)") + ylab("Est. Proportion of Vote") #+ ggtitle("Effect of GCS on Vote Share - City Council Elections") 
g = g + ylim(0, .2) 
g = g + scale_color_manual(values = gender_colors) 
g + theme_minimal()

## 
g = ggplot() 
g = g + geom_ribbon(data = effdf[effdf$Gender=="Women", ], aes(ymin=exp(lower), ymax=exp(upper), x=scaled_fem_body, fill = "#000000"), alpha = 0.3, fill = "#000000")
g = g + geom_line(data = effdf[effdf$Gender=="Women", ], aes(y = exp(fit), x=scaled_fem_body), alpha = 0.3, col = "#000000")
g = g + geom_ribbon(data = effdf[effdf$Gender=="Men", ], aes(ymin=exp(lower), ymax=exp(upper), x=scaled_fem_body, fill = "band"), alpha = 0.3, fill = "#E69F00")
g = g + geom_line(data = effdf[effdf$Gender=="Men", ], aes(y = exp(fit), x=scaled_fem_body), alpha = 0.3, col = "#E69F00")
g = g + geom_rug(data = offd, aes(x = scaled_fem_body, y=exp(logpct), col= Gender)) #+ ylim(0, .75)
g = g + xlab("Conformity Score (low to high)") + ylab("Est. Proportion of Vote") #+ ggtitle("Effect of GCS on Vote Share - Mayoral Elections")
g = g + scale_color_manual(values = gender_colors) 
g + theme_minimal()

##########
offd %>%
  group_by(race, Gender) %>%
  summarise(GCS = mean(fem_body)) %>%
  ungroup() %>%
  mutate_at(c("race"), race_refactor) %>%
  mutate(race = fct_reorder(race, desc(GCS))) %>%
  ggplot(aes(x=race, y=GCS, col=Gender )) + geom_point(size=2) + geom_line(aes(group=race), col="dark grey") +
  theme_minimal() + ylim(0, 1) + xlab("Self-identified Candidate Race") + 
  geom_text(aes(label=round(GCS, digits=2)),hjust=0,vjust=0) + 
  scale_color_manual(values = gender_colors) 

