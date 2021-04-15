library(tidyr)
library(dplyr)
library(tidymodels)
theme_set(theme_minimal())

# RESAMPLING: (rsamples) ----
# Ames Housing ====
# DATA
data(ames)
# EDA
ames %>% skimr::skim()
ames %>% select_if(is.numeric) %>% 
  cor() %>% 
  ggcorrplot::ggcorrplot(type = "lower")
# Split
ames_split <- ames %>% initial_split(prop = 0.8)
ames_train <- ames_split %>% training()
ames_test <- ames_split %>% testing()
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
# FEATURE ENGINEERING: (recipes) ----
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
# MODEL FITTING: (workflows | tune) ----
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
