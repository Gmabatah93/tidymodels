library(dplyr)
library(ggplot2)
theme_set(theme_minimal())
library(rsample)
library(recipes)

# Data
data(ames, package = "modeldata")

# EDA
ames %>% 
  ggplot(aes(Sale_Price)) +
  geom_histogram(bins = 50) +
  scale_x_log10()

ames <- ames %>% mutate(Sale_Price_Log10 = log10(Sale_Price))

# MODELING

# Preprocess
set.seed(123)
# - split: Simple Random Sample
ames_split_srs <- initial_split(ames, prop = 0.80)
# - split: Stratified Sample
ames_split_str <- initial_split(ames, prop = 0.8, strata = Sale_Price_Log10)
# - train
ames_train <- training(ames_split_str)
ames_test <- testing(ames_split_str)

# Feature Engineering
simple_ames <- 
  recipe(Sale_Price_Log10 ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_train) %>% 
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal()) %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>% 
  step_ns(Latitude, Longitude, deg_free = 20)

simple_ames <- prep(simple_ames, training = ames_train)

