library(tidymodels)
library(readxl)
library(plyr)

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

# eeg_data_train <- eeg_data_train %>% mutate(BIS = log(BIS))
# eeg_data_test <- eeg_data_test %>% mutate(BIS = log(BIS))

# eeg_data_train <- eeg_data_train %>% rowwise() %>% mutate(entropy = DescTools::Entropy(x1,x2,x3,x4,x5,x6,x7,x8))

# Defining the pre-processing steps
data_recipe <- recipe(BIS ~ ., data = eeg_data_train) %>%
  # select columns that will be used
  step_select(BIS, x2, x5, x6, x7, x8, skip = T) %>% 
  # apply log transformation function to features
  step_log(x2, x6) %>%  
  # normalise all predictors
  step_normalize(all_numeric_predictors())

# estimates <- prep(data_recipe, eeg_data_train) %>% bake(eeg_data_train)

# sapply(eeg_data_train, function(x) c(sum=sum(x), var=var(scale(x, center = F)), sd=sd(x)))

# Defining the models for the screening
# Multi-adaptive Regression Splines 
mars_reg_spec <- 
  mars() %>% 
  set_mode("regression") %>% 
  set_engine("earth")

# knn
knn_mod_spec <-
  nearest_neighbor() %>%
  # This model can be used for classification or regression, so set mode
  set_mode("regression") %>%
  set_engine("kknn")

# SVR
svr_mod_spec <- 
  svm_linear() %>% 
  set_engine("kernlab") %>% 
  set_mode("regression") 

# ANN
nn_mod_spec <- 
  mlp() %>% 
  # This model can be used for classification or regression, so set mode
  set_mode("regression") %>% 
  set_engine("nnet")

# RF
rf_mod_spec <- 
  rand_forest() %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

# Define the cross validation strategy, 10 folds
set.seed(345)
folds <- vfold_cv(eeg_data_train, v = 10)

# Define the workflow to be applied to all models: preprocess inputs and train with 10-fold CV.
spot_check_wf <- 
  workflow_set(
    preproc = list(none = data_recipe),
    models = list(knn = knn_mod_spec, svr = svr_mod_spec, nn = nn_mod_spec, rf = rf_mod_spec, mars = mars_reg_spec)
  )

# To speed things up, let's run them in parallel using all my cores
library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

# Fitting process
start <- Sys.time()
set.seed(456)
spot_check_rs <- 
  spot_check_wf %>% 
  workflow_map("fit_resamples", resamples = folds, metrics = metric_set(rmse, rsq, mae, mape, smape))
end <- Sys.time()
print(end - start)

model_names <- c("knn", "svr", "ann", "rf", "mars")
screening_results <- data.frame(
  model = as.character(),
  mae = as.numeric(),
  mape = as.numeric(),
  rmse = as.numeric(),
  rsq = as.numeric(),
  smape = as.numeric()
)
temp <- head(screening_results, 0)

for (x in 1:length(model_names)) {
  temp <- ldply(spot_check_rs$result[[x]]$.metrics, data.frame) %>% 
    janitor::clean_names() %>%
    dplyr::group_by(metric) %>% 
    dplyr::summarise(mean = mean(estimate)) %>% 
    mutate(model = model_names[x]) %>% 
    pivot_wider(names_from = metric, values_from = mean)
  screening_results <- screening_results %>% bind_rows(temp)
}

spot_check_rs$info

autoplot(
  spot_check_rs,
  rank_metric = "rmse",
  metric = c("rmse"),
  select_best = T
) +
  geom_text(aes(y = mean - 1/2, label = model), angle = 0, hjust = 0) +
  xlab(label = "Models") +
  theme(legend.position = "none")
