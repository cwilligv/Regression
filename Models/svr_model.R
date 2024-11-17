# Support vector regression
library(tidymodels)
library(readxl)

# read data set
# accessing all the sheets from dataset
sheet = excel_sheets("Project data set 1 (for reports 1 and 3) .xlsx")

# applying sheet names to dataframe names
data_frame = lapply(setNames(sheet[1:10], sheet[1:10]), 
                    function(x) read_excel("Project data set 1 (for reports 1 and 3) .xlsx", sheet=x))

# attaching all dataframes together
eeg_data_train = bind_rows(data_frame, .id="Sheet") %>% select(-Sheet, -x5)

# applying sheet names to dataframe names
data_frame = lapply(setNames(sheet[11:15], sheet[11:15]), 
                    function(x) read_excel("Project data set 1 (for reports 1 and 3) .xlsx", sheet=x))


eeg_data_test = bind_rows(data_frame, .id="Sheet") %>% select(-Sheet, -x5)
# =======================================================================================================
# Fitting the model
svr_mod <- 
  svm_linear(
    cost = 1,
    margin = 0.1
  ) %>% 
  set_engine("kernlab") %>% 
  set_mode("regression") 

start <- Sys.time()
set.seed(234)
svr_fit <- 
  svr_mod %>% 
  fit(BIS ~ ., data = eeg_data_train)
end <- Sys.time()
print(end - start)

svr_fit

pred <- predict(svr_fit, eeg_data_test)
pred_df <- bind_cols(eeg_data_test, pred)

ggplot(pred_df, aes(x = BIS, y = .pred)) + 
  # Create a diagonal line:
  geom_abline(lty = 2) + 
  geom_point(alpha = 0.5) + 
  labs(y = "Predicted Sale Price (log10)", x = "Sale Price (log10)") +
  # Scale and size the x- and y-axis uniformly:
  coord_obs_pred()

rmse(pred_df, truth = BIS, estimate = .pred)

svr_metrics <- metric_set(rmse, rsq, mae)

svr_metrics(pred_df, truth = BIS, estimate = .pred)

# Fitting with resampling (cross validation)
set.seed(345)
folds <- vfold_cv(eeg_data_train, v = 10)
folds

svr_wf <- 
  workflow() %>%
  add_model(svr_mod) %>%
  add_formula(BIS ~ .)

all_cores <- parallel::detectCores(logical = FALSE)

library(doParallel)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

set.seed(456)
svr_fit_rs <- 
  svr_wf %>% 
  fit_resamples(folds)

svr_fit_rs
collect_metrics(svr_fit_rs)
