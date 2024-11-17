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
# Data transformation
data_recipe <- recipe(BIS ~ ., data = eeg_data_train) %>%
  step_rm(x1) %>%
  step_log(all_numeric(), -all_outcomes()) %>% 
  step_normalize(all_numeric(), -all_outcomes())

# Defining the models
# knn
knn_mod_spec <-
  nearest_neighbor(
    neighbors = tune(),
    weight_func = tune(),
    dist_power = tune()
  ) %>%
  # This model can be used for classification or regression, so set mode
  set_mode("regression") %>%
  set_engine("kknn")


set.seed(345)
folds <- vfold_cv(eeg_data_train, v = 10)

library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

start <- Sys.time()
set.seed(0239)
knn_tune <- tune_bayes(knn_mod_spec, data_recipe, folds, metrics = metric_set(rmse, rsq, mae, mape, smape),initial = 6, iter = 20)
end <- Sys.time()
print(end - start)

pred <- predict(mars_reg_fit, eeg_data_test)
pred_df <- bind_cols(eeg_data_test, pred)

rmse(pred_df, truth = BIS, estimate = .pred)

svr_metrics <- metric_set(rmse, rsq, mae)

svr_metrics(pred_df, truth = BIS, estimate = .pred)
