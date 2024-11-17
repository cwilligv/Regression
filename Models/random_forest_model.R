# Random forest model
library(tidymodels)

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
# Fitting the model
rf_mod <- 
  rand_forest(trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

set.seed(234)
rf_fit <- 
  rf_mod %>% 
  fit(BIS ~ ., data = eeg_data_train)
rf_fit

# Fitting with resampling (cross validation)
set.seed(345)
folds <- vfold_cv(eeg_data_train, v = 10)
folds

rf_wf <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_formula(BIS ~ .)

set.seed(456)
rf_fit_rs <- 
  rf_wf %>% 
  fit_resamples(folds)

rf_fit_rs
collect_metrics(rf_fit_rs)




cores <- parallel::detectCores()
cores

rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger", num.threads = cores) %>% 
  set_mode("regression")