library(forcats)
library(tidymodels)
theme_set(theme_minimal())
library(workflowsets)

# RESAMPLING: (rsamples) ----
# Ames Housing ====
# DATA
data(ames)

# EDA
ames %>% skimr::skim()
ames %>% select_if(is.numeric) %>% 
  cor() %>% 
  ggcorrplot::ggcorrplot(type = "lower")
# - Sales Price
ames %>% 
  ggplot(aes(Sale_Price)) +
  geom_histogram(bins = 100)
# - General Living Area
ames %>% 
  ggplot(aes(Gr_Liv_Area)) +
  geom_histogram(bins = 100)

ames %>% 
  ggplot(aes(Gr_Liv_Area, Sale_Price)) +
  geom_point(alpha = 0.2) +
  geom_smooth(aes(color = Bldg_Type),
              method = "lm", se = FALSE)
# - Neighborhood
ames %>% 
  count(Neighborhood) %>% 
  ggplot(aes(n, Neighborhood)) +
    geom_col()

# Split
ames_split <- ames %>% initial_split(prop = 0.8)
ames_train <- ames_split %>% training()
ames_test <- ames_split %>% testing()

# Resample
# - Validation
ames_val <- ames_train %>% validation_split(prop = 3/4)
# - 10 Fold
ames_folds <- ames_train %>% vfold_cv(v = 10)


#
# Cells ====
data(cells, package = "modeldata")
cells <- cells %>% select(-case)
# Split
set.seed(33)
cell_fold <- vfold_cv(cells)
#
# Concrete ====
data(concrete, package = "modeldata")
# Split
concrete_split <- initial_split(concrete, strata = compressive_strength)
concrete_train <- training(concrete_split)
concrete_test <- testing(concrete_split)
# - folds
concrete_folds <- vfold_cv(concrete_train, strata = compressive_strength, repeats = 5)

#
# Home Sales ====
# DATA
home_sales <- readRDS(file = "Data/home_sales.rds") %>% as_tibble()
# EDA
# - split
home_split <- initial_split(home_sales, 
                            prop = 0.7,
                            strata = selling_price)
home_train <- home_split %>% training()
home_test <- home_split %>% testing()

# Telecom ====

# Data
telecom <- readRDS("Data/telecom_df.rds") %>% as_tibble()
# EDA
telecom_train %>% skimr::skim()
telecom_train %>% select_if(is.double) %>% 
  cor() %>% 
  corrplot::corrplot(method = "number", type = "upper")

# - split
telecom_split <- initial_split(telecom,
                               prop = 0.75,
                               strata = canceled_service)

telecom_train <- telecom_split %>% training()
telecom_test <- telecom_split %>% testing()

#
# Loans ====
# Data
loans <- readRDS("Data/loan_df.rds")
# EDA
loans %>% skimr::skim()
loans %>% select_if(is.numeric) %>% 
  cor()
# - split
loans_split <- initial_split(loans,
                             strata = loan_default)

loans_train <- loans_split %>% training()
loans_test <- loans_split %>% testing()
# - resample
set.seed(123)
loans_fold <- loans_train %>% 
  vfold_cv(v = 5, strata = loan_default)
# Cocktails ====
cocktails <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-26/boston_cocktails.csv")
# Clean
cocktails_parsed <- cocktails %>%
  mutate(
    ingredient = str_to_lower(ingredient),
    ingredient = str_replace_all(ingredient, "-", " "),
    ingredient = str_remove(ingredient, " liqueur$"),
    ingredient = str_remove(ingredient, " (if desired)$"),
    ingredient = case_when(
      str_detect(ingredient, "bitters") ~ "bitters",
      str_detect(ingredient, "lemon") ~ "lemon juice",
      str_detect(ingredient, "lime") ~ "lime juice",
      str_detect(ingredient, "grapefruit") ~ "grapefruit juice",
      str_detect(ingredient, "orange") ~ "orange juice",
      TRUE ~ ingredient
    ),
    measure = case_when(
      str_detect(ingredient, "bitters") ~ str_replace(measure, "oz$", "dash"),
      TRUE ~ measure
    ),
    measure = str_replace(measure, " ?1/2", ".5"),
    measure = str_replace(measure, " ?3/4", ".75"),
    measure = str_replace(measure, " ?1/4", ".25"),
    measure_number = parse_number(measure),
    measure_number = if_else(str_detect(measure, "dash$"),
                             measure_number / 50,
                             measure_number
    )
  ) %>%
  add_count(ingredient) %>%
  filter(n > 15) %>%
  select(-n) %>%
  distinct(row_id, ingredient, .keep_all = TRUE) %>%
  na.omit()
# - pivot
cocktails_wide_df <- cocktails_parsed %>%
  select(-ingredient_number, -row_id, -measure) %>%
  pivot_wider(names_from = ingredient, values_from = measure_number, values_fill = 0) %>%
  janitor::clean_names() %>%
  na.omit()
#
# Hip Hop Songs ====

# Data
hiphop <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-14/rankings.csv")

# EDA
hiphop %>% 
  ggplot(aes(year, points, color = gender)) +
  geom_jitter(alpha = 0.5) +
  scale_y_log10()

# UN Voting ====
# Data
unvotes <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-03-23/unvotes.csv")
issues <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-03-23/issues.csv")
# Clean: Columns
unvotes_df <- unvotes %>% 
  select(country, rcid, vote) %>% 
  mutate(vote = factor(vote, levels = c("no","abstain","yes")),
         vote = as.numeric(vote),
         rcid = paste0("rcid_", rcid)) %>% 
  pivot_wider(names_from = "rcid", values_from = "vote", values_fill = 2)
# Hotel Bookings ====

# Data
hotels <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv")
# - clean: 
hotel_stays <- hotels %>%
  filter(is_canceled == 0) %>%
  mutate(
    children = case_when(
      children + babies > 0 ~ "children",
      TRUE ~ "none"
    ),
    required_car_parking_spaces = case_when(
      required_car_parking_spaces > 0 ~ "parking",
      TRUE ~ "none"
    )
  ) %>%
  select(-is_canceled, -reservation_status, -babies)

# EDA
hotel_stays %>% count(children)
hotel_stays %>% skimr::skim()
# - Visual: Monthly stays
hotel_stays %>%
  mutate(arrival_date_month = factor(arrival_date_month, levels = month.name)) %>% 
  count(hotel, arrival_date_month, children) %>% 
  group_by(hotel, children) %>% 
  mutate(prop = n / sum(n)) %>% 
  ggplot(aes(arrival_date_month, prop, fill = children)) +
  geom_col(position = "dodge") +
  facet_wrap(~ hotel, nrow = 2) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme(axis.text.x = element_text(hjust = 1, angle = 45))
# - Visual: Parking
hotel_stays %>%
  count(hotel, required_car_parking_spaces, children) %>% 
  group_by(hotel, children) %>% 
  mutate(prop = n / sum(n)) %>% 
  ggplot(aes(required_car_parking_spaces, prop, fill = children)) +
  geom_col(position = "dodge") +
  facet_wrap(~ hotel, nrow = 2) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme(axis.text.x = element_text(hjust = 1, angle = 45))
# - Visual: Corrplot
library(GGally)

hotel_stays %>% 
  select(children, adr, required_car_parking_spaces, total_of_special_requests) %>% 
  ggpairs(mapping = aes(color = children))

# Model Data
hotels_df <- hotel_stays %>% 
  select(children, hotel, arrival_date_month, meal, adr, adults, 
         required_car_parking_spaces, total_of_special_requests,
         stays_in_week_nights, stays_in_weekend_nights) %>% 
  mutate_if(is.character, factor)

# Split
set.seed(1234)

hotels_split <- initial_split(hotels_df)
hotels_train <- training(hotels_split)
hotels_test <- testing(hotels_split)
#
# The Office ====

# Data
ratings_raw <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-03-17/office_ratings.csv")
office <- schrute::theoffice

remove_regex <- "[:punct:]|[:digit:] |parts |part |the |and"

office_ratings <- ratings_raw %>%
  transmute(
    episode_name = str_to_lower(title),
    episode_name = str_remove_all(episode_name, remove_regex),
    episode_name = str_trim(episode_name),
    imdb_rating
  )

office_info <- office %>%
  mutate(
    season = as.numeric(season),
    episode = as.numeric(episode),
    episode_name = str_to_lower(episode_name),
    episode_name = str_remove_all(episode_name, remove_regex),
    episode_name = str_trim(episode_name)
  ) %>%
  select(season, episode, episode_name, director, writer, character)

office_characters <- office_info %>% 
  count(episode_name, character) %>% 
  add_count(character, wt = n, name = "character_count") %>% 
  filter(character_count > 800) %>% 
  select(-character_count) %>% 
  pivot_wider(names_from = character, 
              values_from = n,
              values_fill = list(n = 0))

office_creators <- office_info %>% 
  distinct(episode_name, director, writer) %>% 
  pivot_longer(director:writer, 
               names_to = "role",
               values_to = "person") %>% 
  separate_rows(person, sep = ";") %>% 
  add_count(person) %>% 
  filter(n > 10) %>% 
  distinct(episode_name, person) %>% 
  mutate(person_value = 1) %>% 
  pivot_wider(names_from = person,
              values_from = person_value,
              values_fill = list(person_value = 0))
  
office_model_data <- office_info %>% 
  distinct(season, episode, episode_name) %>% 
  inner_join(office_characters) %>% 
  inner_join(office_creators) %>% 
  inner_join(office_ratings) %>% 
  janitor::clean_names()

# EDA
office_model_data %>% 
  ggplot(aes(season, imdb_rating, fill = as.factor(season))) +
  geom_boxplot(show.legend = FALSE)

# Split
office_split <- initial_split(office_model_data, strata = season)
office_train <- training(office_split)
office_test <- testing(office_split)
# - bootstrap
office_boot <- bootstraps(office_train, strata = season)

#
# Food Consumption ====

# Data
food_consumption <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-18/food_consumption.csv")

# - clean
food_df <- food_consumption %>% 
  mutate(continent = countrycode::countrycode(country, 
                                              origin = "country.name",
                                              destination = "continent")) %>% 
  select(-co2_emmission) %>% 
  pivot_wider(names_from = food_category,
              values_from = consumption) %>% 
  janitor::clean_names() %>% 
  mutate(asia = case_when(continent == "Asia" ~ "Asia",
                          TRUE ~ "Other")) %>% 
  select(-country, -continent) %>% 
  mutate_if(is.character, factor)

# EDA
library(GGally)

food_df %>% 
  ggscatmat(columns = 1:11, color = "asia", alpha = 0.6)

# Split
set.seed(1234)

food_split <- initial_split(food_df, strata = asia)
food_train <- training(food_split)
food_test <- testing(food_split)
# - bootstrap
food_boot <- bootstraps(food_train, times = 30)

#
# FEATURE ENGINEERING: (recipes) ----
# Ames Housing ====
ames_recipe <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude,
         data = ames_train) %>% 
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = tune()) %>% 
  step_dummy(all_nominal()) %>% 
  step_interact(~Gr_Liv_Area:starts_with("Bldg_Type_")) %>% 
  step_ns(Latitude, deg_free = tune("longitude df")) %>% 
  step_ns(Longitude, deg_free = tune("latitude df"))

ames_recipe %>% parameters()  

ames_prep <- 
  ames_recipe %>% 
  prep(training = ames_train)

ames_prep %>% bake(new_data = NULL)

#
# Cells ====

# Multilayer Perceptron
cells_mlp_recipe <- 
  recipe(class ~ ., data = cells) %>% 
  step_YeoJohnson(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), num_comp = tune()) %>% 
  step_normalize(all_predictors())

# Support Vector Machine
cells_svm_recipe <- 
  recipe(class ~ ., data = cells) %>% 
  step_YeoJohnson(all_predictors()) %>% 
  step_normalize(all_predictors())
#
# Concrete ====
concrete_normalized_recipe <- 
  recipe(compressive_strength ~ ., data = concrete_train) %>% 
  step_normalize(all_predictors())

concrete_poly_recipe <- 
  concrete_normalized_recipe %>% 
  step_poly(all_predictors()) %>% 
  step_interact(~ all_predictors():all_predictors())

#
# Telcom ====

# Recipe
telecom_recipe <- 
  recipe(canceled_service ~ ., data = telecom_train) %>% 
  step_log(avg_call_mins, avg_intl_mins, base = 10) %>% 
  step_corr(all_numeric(), threshold = 0.8) %>% 
  step_normalize(all_numeric()) %>% 
  step_dummy(all_nominal(), -all_outcomes())
# - train
telecom_recipe_prep <- 
  telecom_recipe %>% 
  prep(training = telecom_train)
# - apply
telecom_train_prep <- telecom_recipe_prep %>% 
  bake(new_data = NULL)
telecom_test_prep <- telecom_recipe_prep %>% 
  bake(new_data = telecom_test)

#
# Loans ====
loans_recipe <- 
  recipe(loan_default ~ ., data = loans_train) %>% 
  step_corr(all_numeric(), threshold = 0.85) %>% 
  step_normalize(all_numeric()) %>% 
  step_dummy(all_nominal(), -all_outcomes())
# Cocktails: PCA ====

# - Preprocess
cocktail_pca_rec <- recipe(~., data = cocktails_wide_df) %>%
  update_role(name, category, new_role = "id") %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors())

cocktail_pca_prep <- prep(cocktail_pca_rec)
# - Visuals
tidy(cocktail_pca_prep, 2) %>% 
  filter(component %in% paste0("PC", 1:5)) %>%
  mutate(component = fct_inorder(component)) %>%
  ggplot(aes(value, terms, fill = terms)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~component, nrow = 1) +
  labs(y = NULL) +
  theme_bw()

tidy(cocktail_pca_prep, 2) %>%
  filter(component %in% paste0("PC", 1:4)) %>%
  group_by(component) %>%
  top_n(8, abs(value)) %>%
  ungroup() %>%
  ggplot(aes(abs(value), terms, fill = value > 0)) +
  geom_col() +
  facet_wrap(~component, scales = "free_y") +
  labs(
    x = "Absolute value of contribution",
    y = NULL, fill = "Positive?"
  )

# - projection: "U"SV
juice(cocktail_pca_prep) %>%
  ggplot(aes(PC1, PC2, label = name)) +
  geom_point(aes(color = category), alpha = 0.7, size = 2) +
  geom_text(check_overlap = TRUE, hjust = "inward", family = "IBMPlexSans") +
  labs(color = NULL)
#
# UN Voting: PCA ====
# - Preprocess
unvotes_pca_rec <- 
  recipe( ~ ., data = unvotes_df) %>% 
  update_role(country, new_role = "id") %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors())

unvotes_pca_prep <- prep(unvotes_pca_rec)
# - Visual
bake(unvotes_pca_prep, new_data = NULL) %>% 
  ggplot(aes(PC1, PC2, label = country)) +
  geom_point(color = "midnightblue", alpha = 0.7, size = 2) +
  geom_text(check_overlap = TRUE, hjust = "inward")

# - Issues
unvotes_pca_issues_comps <- tidy(unvotes_pca_prep, 2) %>% 
  filter(component %in% c("PC1","PC2","PC3","PC4")) %>% 
  left_join(issues %>% 
              mutate(terms = paste0("rcid_", rcid))) %>% 
  filter(!is.na(issue)) %>% 
  group_by(component) %>% 
  slice_max(abs(value), n = 8) %>% 
  ungroup()
# - visual
unvotes_pca_issues_comps %>% 
  mutate(value = abs(value)) %>% 
  ggplot(aes(value, terms, fill = issue)) +
  geom_col(position = "dodge") +
  facet_wrap(~ component, scales = "free_y") +
  labs(y = NULL, fill = NULL, x = "Abs Value of Contribution")
#
# Hotel Bookings ====

# Preprocess
hotels_rec <- 
  recipe(children ~ ., data = hotels_train) %>% 
  step_downsample(children) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_numeric()) %>% 
  step_normalize(all_numeric())
  
hotels_train_prep <- hotels_rec %>% prep() %>% juice()
hotels_train_prep %>% count(children)

hotel_test_prep <- bake(hotels_rec, new_data = hotels_test)


#
#
# The Office ====
office_rec <- 
  recipe(imdb_rating ~ ., data = office_train) %>% 
  update_role(episode_name, new_role = "ID") %>% 
  step_zv(all_numeric(), -all_outcomes()) %>% 
  step_normalize(all_numeric(), -all_outcomes())
office_prep <- prep(office_rec)

#

# Food Consumption ====
food_rec <- 
  recipe(asia ~ ., data = food_train)
#
# MODEL FITTING: (workflows | tune) ----
model_db
# Ames Housing: Linear Regression ====
# - spec
ames_lm_model <- 
  linear_reg() %>% 
  set_engine("lm")
# - fit
ames_lm_fit <- 
  ames_lm_model %>% 
  fit(Sale_Price ~ Longitude + Latitude, data = ames_train)

# Workflow
ames_lm_wflow <- 
  workflow() %>% 
  add_model(ames_lm_model) %>% 
  add_recipe(ames_recipe)

# Workflow Set
location <- list(
  longitude = Sale_Price ~ Longitude,
  latitude = Sale_Price ~ Latitude,
  coords = Sale_Price ~ Longitude + Latitude,
  neighborhood = Sale_Price ~ Neighborhood
)
# - fit
location_models <- workflow_set(preproc = location, models = list(lm = ames_lm_model))
location_models$info[[1]]
location_models %>% pull_workflow(id = "coords_lm")

location_models <-
  location_models %>%
  mutate(fit = map2(info, wflow_id, ~ fit(.x$workflow[[1]], ames_train)))#
location_models

#
# Ames Housing: Decision Tree ====
# - spec
ames_dt_model <- 
  decision_tree() %>% 
  set_engine("rpart") %>% 
  set_mode("regression")
# - fit
ames_dt_fit <- 
  ames_dt_model %>% 
  fit(Sale_Price ~ Longitude + Latitude, data = ames_train)

#
# Ames Housing: Random Forrest ====
# - spec
ames_rf_model <- 
  rand_forest(
    trees = 1000
  ) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

# Workflow
ames_rf_wflow <- 
  workflow() %>% 
  add_formula(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude) %>% 
  add_model(ames_rf_model)
# - fit
ames_rf_fit <- 
  ames_rf_wflow %>% 
  fit(data = ames_train)

# Resamples: Validation
rf_resample_val <- 
  ames_rf_wflow %>% 
  fit_resamples(resamples = ames_val)
# - metrics
rf_resample_val %>% collect_metrics()
# Resamples: 10 Fold
set.seed(123)
ames_keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

rf_resample_10Fold <- 
  ames_rf_wflow %>% 
  fit_resamples(resamples = ames_folds, control = ames_keep_pred)
# - metrics
rf_resample_10Fold %>% collect_metrics()
rf_resample_10Fold %>% collect_predictions() %>% 
  ggplot(aes(Sale_Price, .pred)) +
  geom_point(alpha = 0.5) + geom_abline(color = "red") +
  coord_obs_pred() +
  theme_bw()
#
# Ames Housing: EXTRA ====
# - Recipe
ames_basic_recipe <- 
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + 
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>% 
  step_other(Neighborhood, threshold = 0.01) %>% 
  step_dummy(all_nominal())

ames_interaction_recipe <- 
  ames_basic_recipe %>% 
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) 

ames_spline_recipe <- 
  ames_interaction_rec %>% 
  step_ns(Latitude, Longitude, deg_free = 50)
# - Preprocess
ames_preproc <- 
  list(basic = ames_basic_recipe, 
       interact = ames_interaction_recipe, 
       splines = ames_spline_recipe
  )
# - Fit
ames_lm_models <- workflow_set(ames_preproc, list(lm = ames_lm_model), cross = FALSE)
ames_lm_models <- 
  ames_lm_models %>% 
  workflow_map("fit_resamples",
               seed = 1101, verbose = TRUE,
               resamples = ames_folds, control = ames_keep_pred)
# - Eval
ames_lm_models %>% 
  collect_metrics() %>% 
  filter(.metric == "rmse")
# - Add Random Forrest
ames_four_models <- 
  as_workflow_set(random_forest = rf_resample) %>% 
  bind_rows(ames_lm_models)
# - Eval
ames_four_models %>% autoplot(metric = "rsq") + theme_bw()

#
# Cells: Multilayer Perceptron ====
# - Spec
cell_mlp_spec <- 
  mlp(hidden_units = tune(),
      penalty = tune(),
      epochs = tune()) %>% 
  set_engine("nnet", trace = 0) %>% 
  set_mode("classification")

# Workflow
cell_mlp_wflow <- 
  workflow() %>% 
  add_model(cell_mlp_spec) %>% 
  add_recipe(cells_mlp_recipe)

# Hyperparameters
cells_mlp_param <- 
  cell_mlp_wflow %>% 
  parameters() %>% 
  update(epochs = epochs(c(50,200)),
         num_comp = num_comp(c(0,40)))
# - metric
cell_roc_resample <- metric_set(roc_auc)
# - fit
set.seed(99)
celll_mlp_reg_tune <- 
  cell_mlp_wflow %>% 
  tune_grid(cell_fold, 
            grid = cells_mlp_param %>% grid_regular(levels = 3), 
            metrics = cell_roc_resample)
# - visual
celll_mlp_reg_tune %>% 
  autoplot() + 
  theme_bw()
celll_mlp_reg_tune %>% show_best()
# - final: workflow
cell_mlp_best <- celll_mlp_reg_tune %>% select_best(metric = "roc_auc")
cell_mlp_wflow_final <- 
  cell_mlp_wflow %>% 
  finalize_workflow(cell_mlp_best)
# - final: fit
cell_mlp_fit_final <- 
  cell_mlp_wflow_final %>% 
  fit(cells)
#
# Cells: Support Vector Machine ====
# Spec
cells_svm_spec <- 
  svm_rbf(cost = tune(),
          rbf_sigma = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

# Workflow
cells_svm_wflow <- 
  workflow() %>% 
  add_model(cells_svm_spec) %>% 
  add_recipe(cells_svm_recipe)

# Hyperparameters
cells_svm_param <- 
  cells_svm_wflow %>% 
  parameters() %>% 
  update(rbf_sigma = rbf_sigma(c(-7,-1)))
# - initial grid
cells_start_grid <- 
  cells_svm_param %>% 
  update(
    cost = cost(c(-6,1)),
    rbf_sigma = rbf_sigma(c(-6,-4))
  ) %>% 
  grid_regular(levels = 2)

cells_initial <- 
  cells_svm_wflow %>% 
  tune_grid(resamples = cell_fold,
            grid = cells_start_grid,
            metric = cell_roc_resample)

cells_initial %>% collect_metrics() %>% filter(.metric == "roc_auc")

# - bayesian optimization
cell_ctrl_svm_bayes <- control_bayes(verbose = TRUE)

set.seed(1234)
cell_svm_tune_bayes <- 
  cells_svm_wflow %>% 
  tune_bayes(
    resamples = cell_fold,
    metric = cell_roc_resample,
    initial = cells_initial,
    param_info = cells_svm_param,
    iter = 25, 
    control = cell_ctrl_svm_bayes
  )

# - simulated annealing

#
# Concrete: Multi
# - spec
concrete_lm_spec <- 
  linear_reg(penalty = tune(),
             mixture = tune()) %>% 
  set_engine("glmnet")

concrete_nnet_spec <- 
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>% 
  set_engine("nnet", MaxNWts = 2600) %>% 
  set_mode("regression")

concrete_mars_spec <- 
  mars(prod_degree = tune()) %>%  #<- use GCV to choose terms
  set_engine("earth") %>% 
  set_mode("regression")

concrete_svm_r_spec <- 
  svm_rbf(cost = tune(), rbf_sigma = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

concrete_svm_p_spec <- 
  svm_poly(cost = tune(), degree = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression")

concrete_knn_spec <- 
  nearest_neighbor(neighbors = tune(), dist_power = tune(), weight_func = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

concrete_cart_spec <- 
  decision_tree(cost_complexity = tune(), min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

concrete_rf_spec <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

concrete_xgb_spec <- 
  boost_tree(tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(), 
             min_n = tune(), sample_size = tune(), trees = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

library(rules)
concrete_cubist_spec <- 
  cubist_rules(committees = tune(), neighbors = tune()) %>% 
  set_engine("Cubist") 

# Hyperparameters
concrete_nnet_param <- 
  concrete_nnet_spec %>% 
  parameters() %>% 
  update(hidden_units = hidden_units(c(1,27)))
  
# Workflowset
concrete_models_normalized <-
  workflow_set(
    preproc = list(normalized = concreate_normalized_recipe),
    models = list(SVM_radial = concrete_svm_r_spec,
                  SVM_poly = concrete_svm_p_spec,
                  KNN = concrete_knn_spec,
                  NNET = concrete_nnet_spec)
  )
concrete_models_normalized %>% pull_workflow(id = "normalized_KNN")
# - add nnet parameter
concrete_models_normalized <- concrete_models_normalized %>% 
  option_add(param = concrete_nnet_param, id = "normalized_NNET")
concrete_models_normalized %>% pull_workflow(id = "normalized_NNET")
# - model variables
concrete_model_vars <- 
  workflow_variables(outcomes = compressive_strength, 
                     predictors = everything())
concrete_no_pre_proc <- 
  workflow_set(
    preproc = list(simple = concrete_model_vars), 
    models = list(MARS = concrete_mars_spec, CART = concrete_cart_spec,
                  RF = concrete_rf_spec, boosting = concrete_xgb_spec, Cubist = concrete_cubist_spec)
  )
concrete_with_features <- 
  workflow_set(
    preproc = list(full_quad = concrete_poly_recipe), 
    models = list(linear_reg = concrete_lm_spec, KNN = concrete_knn_spec)
  )

all_workflow <- 
  bind_rows(concrete_no_pre_proc, concrete_models_normalized, concrete_with_features) %>% 
  # Make the workflow ID's a little more simple: 
  mutate(wflow_id = gsub("(simple_)|(normalized_)", "", wflow_id))

# - grid
concrete_grid_control <- 
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
  )

concrete_grid_results <-
  all_workflow %>%
  workflow_map(
    seed = 1503,
    resamples = concrete_folds,
    grid = 25,
    control = concrete_grid_control
  )
#
# Telecom: Logistic Regression ====
# - spec
log_model <- 
  logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

log_fit <- 
  log_model %>% 
  fit(canceled_service ~ avg_call_mins + avg_intl_mins + monthly_charges,
      data = telecom_train)
log_fit %>% tidy()
# - final
log_fit_2 <- 
  log_model %>% 
  last_fit(canceled_service ~ avg_call_mins + avg_intl_mins + monthly_charges + months_with_company,
           split = telecom_split)


# Workflow
log_fit_prep_full <- 
  log_model %>% 
  fit(canceled_service ~ ., data = telecom_train_prep)

#
# Home Sales: Linear Regression  ====
# - spec
lm_mod <- 
  linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")
# - fit
lm_fit <- 
  lm_mod %>% 
  fit(selling_price ~ home_age + sqft_living, 
      data = home_train)
# - final
lm_final <- 
  lm_mod %>% 
  last_fit(selling_price ~ ., split = home_split)
#
# Loans: Decision Tree ====

# Spec
loan_dt_model <- 
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune(),
    min_n = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")
# Workflow
loans_dt_wflow <- 
  workflow() %>% 
  # model spec
  add_model(loan_dt_model) %>% 
  # recipe
  add_recipe(loans_recipe)

# Hyperparameter
set.seed(123)
# - grid
loans_dt_grid <- 
  grid_random(parameters(loan_dt_model),
              size = 5)

# Fit
loans_dt_fit <- 
  loans_dt_wflow %>% 
  tune_grid(resamples = loans_fold,
            grid = loans_dt_grid,
            metric = loans_metrics)
# - metric
loans_metrics <- metric_set(roc_auc, sens, spec)
# - refit
loans_dt_resample <- 
  loans_dt_wflow %>% 
  fit_resamples(resamples = loans_fold, 
                metrics = loans_metrics)
# FINAL
# - hyperparameter
loans_dt_best <- 
  loans_dt_fit %>% 
  select_best(metric = "roc_auc")
# - workflow
loans_dt_wflow_final <- 
  loans_dt_wflow %>% 
  finalize_workflow(loans_dt_best)
# - fit
loans_dt_final <-
  loans_dt_final_wflow %>% 
  last_fit(split = loans_split)
  

# Loans: Logistic Regression ====
# Spec
loans_log_model <- 
  logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

# Workflow 
loans_log_wflow <- 
  workflow() %>% 
  add_model(loans_log_model) %>% 
  add_recipe(loans_recipe)

# Fit
# - resamples
loans_log_resample <- 
  loans_log_wflow %>% 
  fit_resamples(resamples = loans_fold,
                metrics = loans_metrics)

# Hotel Bookings: KNN ====

# Validation Set
set.seed(1234)
hotels_validation_splits <- mc_cv(hotels_train, prop = 0.9, strata = children)

# Spec
hotels_knn_spec <- nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

# Workflow
hotels_knn_wflow <- 
  workflow() %>% 
  add_model(hotels_knn_spec) %>% 
  add_recipe(hotels_rec)

# Resample
hotel_knn_res <- 
  hotels_knn_wflow %>% 
  fit_resamples(
    hotels_validation_splits,
    control = control_resamples(save_pred = TRUE)
  )
# - eval
hotel_knn_res %>% collect_metrics()

#
# Hotel Bookings: Decision Tree ====

# Spec 
hotels_dt_spec <- decision_tree() %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

# Workflow
hotels_dt_wflow <-
  workflow() %>% 
  add_model(hotels_dt_spec) %>% 
  add_recipe(hotels_rec)

# Resample
hotels_dt_res <- 
  hotels_dt_wflow %>% 
  fit_resamples(
    hotels_validation_splits,
    control = control_resamples(save_pred = TRUE)
  )
# - eval
hotels_dt_res %>% collect_metrics()

#
# The Office: Lasso Regression ====

# Spec
office_spec <- 
  linear_reg(penalty = tune(),
             mixture = 1) %>% 
  set_engine("glmnet")

# Workflow
office_wflow <- 
  workflow() %>% 
  add_recipe(office_rec) %>% 
  add_model(office_spec)

# Resample
set.seed(1234)

office_lasso_grid <- 
  grid_regular(penalty(),
               levels = 50)
# - fit
office_lasso_resample <- 
  tune_grid(office_wflow,
            resamples = office_boot,
            grid = office_lasso_grid)
# - visual
office_lasso_resample %>% 
  collect_metrics() %>% 
  ggplot(aes(penalty, mean, color = .metric)) +
  geom_line(size = 1.5, show.legend = FALSE) +
  geom_ribbon(aes(ymin = mean - std_err,
                  ymax = mean + std_err),
                alpha = 0.1) +
  facet_wrap(~ .metric, scale = "free", nrow = 2) +
  scale_x_log10() + 
  theme_bw() + 
  theme(legend.position = "none")
# - best
office_lasso_best <- 
  office_lasso_resample %>% 
  select_best("rmse")
# - final: workflow
office_lasso_wflow_FINAL <- 
  finalize_workflow(office_wflow,
                    office_lasso_best)
# - final: fit
office_lasso_FINAL <- 
  office_lasso_wflow_FINAL %>% 
  fit(office_train)
# OR
last_fit(office_lasso_wflow_FINAL,
         office_split) %>% 
  collect_metrics()

office_lasso_FINAL %>% 
  pull_workflow_fit() %>% 
  vip::vi(lambda = office_lasso_best$penalty) %>% 
  mutate(Importance = abs(Importance),
         Variable = fct_reorder(Variable, Importance)) %>% 
  ggplot(aes(Importance, Variable, fill = Sign)) +
  geom_col()
#
# Food Consumption: Random Forrest ====

# Spec
food_rf_spec <- 
  rand_forest(
    mtry = tune(),
    trees = 1000,
    min_n = tune()
  ) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

# Workflow
food_rf_wflow <- 
  workflow() %>% 
  add_model(food_rf_spec) %>% 
  add_recipe(food_rec)

# Hyperparameters
food_fit <- 
  food_rf_wflow %>% 
  tune_grid(
    resamples = food_boot
  )
# - eval
food_fit %>% collect_metrics()
food_fit %>% select_best("roc_auc")
#
# MODEL EVALUATION: (yardstick) ----
# Home Sales ====
# - prediction
home_sales_results <- home_test %>% 
  select(selling_price, home_age, sqft_living) %>% 
  mutate(lm_pred = lm_fit %>% predict(home_test) %>% pull)

# - evaluation
home_sales_results %>% rmse(truth = selling_price, estimate = lm_pred)
home_sales_results %>% rsq(truth = selling_price, estimate = lm_pred)
# - visual
lm_test_results %>% 
  ggplot(aes(selling_price, lm_pred)) +
  geom_point(alpha = 0.5) + geom_abline(color = "purple", linetype = 2) +
  coord_obs_pred() +
  labs(
    title = "Evaluation",
    x = "Actual",
    y = "Predicted"
  ) +
  theme_bw()

# - FINAL evaluation
lm_final %>% 
  collect_predictions() %>% 
  ggplot(aes(selling_price, .pred)) +
  geom_point(alpha = 0.5) + geom_abline(color = "red", linetype = 2) +
  coord_obs_pred() +
  labs(
    title = "Final: Evaluation",
    x = "Actual",
    y = "Predicted"
  ) +
  theme_bw()




# Telecom ====
# - prediction
telecom_results <- telecom_test %>% 
  select(canceled_service) %>% 
  mutate(log_pred = log_fit %>% predict(telecom_test, type = "class") %>% pull,
         log_prob = log_fit %>% predict(telecom_test, type = "prob") %>% pull(.pred_yes))
# - confusion matrix
telecom_results %>% conf_mat(truth = canceled_service, estimate = log_pred) %>% summary()
telecom_results %>% conf_mat(truth = canceled_service, estimate = log_pred) %>% autoplot(type = "heatmap")
telecom_results %>% conf_mat(truth = canceled_service, estimate = log_pred) %>% autoplot(type = "mosaic")

telecom_results %>% accuracy(truth = canceled_service, estimate = log_pred)
telecom_results %>% sens(truth = canceled_service, estimate = log_pred)
telecom_results %>% spec(truth = canceled_service, estimate = log_pred)

telecom_metrics <- metric_set(accuracy, sens, spec, roc_auc)
telecom_results %>% telecom_metrics(truth = canceled_service, 
                                    estimate = log_pred, log_prob)
# - roc curve
telecom_results %>% 
  roc_curve(truth = canceled_service, estimate = log_prob) %>% 
  autoplot()
telecom_results %>% roc_auc(truth = canceled_service, estimate = log_prob)

# FINAL evaluation
telecom_fit_2_results <- log_fit_2 %>% collect_predictions()

telecom_fit_2_metrics <- metric_set(accuracy, sens, spec, roc_auc)
telecom_fit_2_metrics %>% telecom_final_metrics(truth = canceled_service, 
                                                estimate = .pred_class, .pred_yes)
telecom_fit_2_results %>% conf_mat(truth = canceled_service, estimate = .pred_class) %>% summary()

# Workflow Evaluation
# - prediction
telecom_prep_results <- telecom_test_prep %>% 
  select(canceled_service) %>% 
  mutate(pred = log_fit %>% predict(telecom_test, type = "class") %>% pull,
         prob = log_fit %>% predict(telecom_test, type = "prob") %>% pull(.pred_yes))
# - confusion matrix
telecom_prep_results %>% conf_mat(truth = canceled_service, estimate = pred) %>% summary()



# Loans ====

# Decision Tree
# - fit
loans_dt_fit %>% collect_metrics()
loans_dt_fit %>% show_best(metric = "roc_auc")
# - resample
loans_dt_resample %>% collect_metrics()
# - fianl
loans_dt_final %>% collect_metrics()

# Logistic Regression
# - resample
loans_log_resample %>% collect_metrics()

# Results
loans_dt_resample %>% 
  collect_metrics(summarize = FALSE) %>% 
  group_by(.metric) %>% 
  summarise(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate))

loans_log_resample %>% 
  collect_metrics(summarize = FALSE) %>% 
  group_by(.metric) %>% 
  summarise(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate))

# Ames Housing ====

# Predictions
# - Linear Regression
ames_results <- ames_test %>% 
  select(Sale_Price) %>% 
  mutate(lm_pred = ames_lm_fit %>% predict(new_data = ames_test) %>% pull)
# - Random Forrest
ames_results <- ames_results %>% 
  add_column(rf_pred = ames_rf_fit %>% predict(new_data = ames_test) %>% pull)

# Linear Regression
# - visual
ames_results %>% 
  ggplot(aes(Sale_Price, lm_pred)) +
  geom_abline(lty = 2) + geom_point(alpha = 0.5) +
  labs(
    title = "Linear Regression",
    x = "Sale Price", y = "Prediction"
  ) +
  coord_obs_pred()
# - evaluation
ames_metrics <- metric_set(rmse, rsq, mae)
ames_results %>% ames_metrics(truth = Sale_Price, estimate = lm_pred)

# Random Forrest
# - visual
ames_results %>% 
  ggplot(aes(Sale_Price, rf_pred)) +
  geom_abline(lty = 2) + geom_point(alpha = 0.5) +
  labs(
    title = "Random Forrest",
    x = "Sale Price", y = "Prediction"
  ) +
  coord_obs_pred()
# - evaulation
ames_results %>% ames_metrics(truth = Sale_Price, estimate = rf_pred)

# Hotel Bookings ====

# Resample: 
# - ROC
hotel_knn_res %>% 
  unnest(.predictions) %>% 
  mutate(model = "kNN") %>% 
  bind_rows(hotels_dt_res %>% 
              unnest(.predictions) %>% 
              mutate(model = "DT")) %>% 
  group_by(model) %>% 
  roc_curve(truth = children, .pred_children) %>% 
  autoplot()
# - CM
hotel_knn_res %>% 
  unnest(.predictions) %>% 
  conf_mat(children, .pred_class) %>% 
  autoplot(type = "heatmap")

# kNN

# Decision Tree