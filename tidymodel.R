library(tidyr)
library(dplyr)
library(tidymodels)
theme_set(theme_minimal())

# Data ----
home_sales <- readRDS(file = "Data/home_sales.rds") %>% as_tibble()

# Data Resampling: (rsamples) ----
# - split
home_split <- initial_split(home_sales, 
                            prop = 0.7,
                            strata = selling_price)
home_train <- home_split %>% training()
home_test <- home_split %>% testing()


#
# Feature Engineering ----
# Model Fitting: (parsnip) ----

# Linear Regression
# - fit
lm_mod <- 
  linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression")

lm_fit <- 
  lm_mod %>% 
  fit(selling_price ~ home_age + sqft_living, 
      data = home_train)
# - final
lm_final <- 
  lm_mod %>% 
  last_fit(selling_price ~ ., split = home_split)
# - prediction
lm_test_results <- home_test %>% 
  select(selling_price, home_age, sqft_living) %>% 
  mutate(lm_pred = lm_fit %>% predict(home_test) %>% pull)
#
# Model Tuning ----
# Model Evaluation: (yardstick) ----
# Linear Regression
lm_test_results
# - evaluation
lm_test_results %>% rmse(truth = selling_price, estimate = lm_pred)
lm_test_results %>% rsq(truth = selling_price, estimate = lm_pred)
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
# - final evaluation
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
