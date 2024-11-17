library(tidymodels)
library(readxl)

# read data set
# accessing all the sheets from dataset
sheet = excel_sheets("Project data set 1 (for reports 1 and 3) .xlsx")

# applying sheet names to dataframe names
data_frame = lapply(setNames(sheet[1:10], sheet[1:10]), 
                    function(x) read_excel("Project data set 1 (for reports 1 and 3) .xlsx", sheet=x))

# attaching all dataframes together
eeg_data_train = bind_rows(data_frame, .id="Sheet") %>% select(-Sheet)

# applying sheet names to dataframe names
data_frame = lapply(setNames(sheet[11:15], sheet[11:15]), 
                    function(x) read_excel("Project data set 1 (for reports 1 and 3) .xlsx", sheet=x))


eeg_data_test = bind_rows(data_frame, .id="Sheet") %>% select(-Sheet)
# =======================================================================================================
# eeg_data_train <- eeg_data_train %>% mutate(BIS = log(BIS))
# eeg_data_test <- eeg_data_test %>% mutate(BIS_orig = BIS, BIS = log(BIS))
# Data transformation
data_recipe <- recipe(BIS ~ ., data = eeg_data_train) %>%
  step_select(BIS, x2, x5, x6, x7, x8, skip = T) %>%
  step_log(x2, x6) %>% 
  step_normalize(all_numeric_predictors())

estimates <- prep(data_recipe, eeg_data_train) %>% bake(eeg_data_train)
moments::skewness((estimates$BIS)^(1/3))

DataExplorer::plot_correlation(estimates)

# Defining the models
# mars
mars_reg_spec <- 
  mars(prod_degree = tune(), prune_method = tune()) %>% 
  # This model can be used for classification or regression, so set mode
  set_mode("regression") %>% 
  set_engine("earth")

set.seed(345)
folds <- vfold_cv(eeg_data_train, v = 5)

mars_grid <- grid_regular(
  prod_degree(),
  prune_method(),
  levels = list(prod_degree = 2, prune_method = 6)
)

ctrl <- control_grid(verbose = FALSE, save_pred = TRUE)

library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

start <- Sys.time()
set.seed(0239)
# knn_tune <- tune_bayes(knn_mod_spec, data_recipe, folds, metrics = metric_set(rmse, rsq, mae, mape, smape),initial = 6, iter = 20)
mars_tune <- tune_grid(mars_reg_spec, data_recipe, folds, metrics = metric_set(rmse, rsq, mae, mape, smape), grid = 20, control = ctrl)
end <- Sys.time()
print(end - start)

show_best(mars_tune, metric = "rmse")

knn_best <- knn_tune %>% select_best(metric = "rmse")

mars_final_spec <-
  mars(prod_degree = 2, prune_method = "exhaustive") %>% 
  # This model can be used for classification or regression, so set mode
  set_mode("regression") %>% 
  set_engine("earth")

mars_final_wf <- 
  workflow() %>% 
  add_model(mars_final_spec) %>% 
  add_recipe(data_recipe)

start <- Sys.time()
set.seed(234)
mars_final_fit <- 
  mars_final_wf %>% 
  fit(data = eeg_data_train)
end <- Sys.time()
print(end - start)

pred <- predict(mars_final_fit, new_data = eeg_data_test)
pred_df <- bind_cols(eeg_data_test, pred)

rmse(pred_df, truth = BIS, estimate = .pred)

svr_metrics <- metric_set(rmse, rsq, mae)

svr_metrics(pred_df, truth = BIS, estimate = .pred)

ggplot(pred_df, aes(x = BIS, y = .pred)) + 
  # Create a diagonal line:
  geom_abline(lty = 2) + 
  geom_point(alpha = 0.5) + 
  labs(y = "Predicted BIS (log10)", x = "BIS (log10)") +
  # Scale and size the x- and y-axis uniformly:
  coord_obs_pred()

collect_predictions(mars_final_fit)

pred <- predict(mars_final_fit, eeg_data_test)
pred_df <- bind_cols(eeg_data_test, pred)

rmse(pred_df, truth = BIS, estimate = .pred)

svr_metrics <- metric_set(rmse, rsq, mae)

svr_metrics(pred_df, truth = BIS, estimate = .pred)