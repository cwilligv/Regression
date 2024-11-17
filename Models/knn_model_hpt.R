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
  step_select(BIS, x2, x5, x6, x7, x8, skip = T) %>%
  step_log(x2, x6) %>% 
  step_normalize(all_numeric_predictors())

# estimates <- prep(data_recipe, eeg_data_train) %>% bake(eeg_data_train)
# moments::skewness((estimates$BIS)^(1/3))

# DataExplorer::plot_correlation(estimates)

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

knn_grid <- grid_regular(
  neighbors(),
  weight_func(),
  dist_power(),
  levels = list(neighbors = 4, weight_func = 4, dist_power = 1)
)
  
ctrl <- control_grid(verbose = FALSE, save_pred = TRUE)

library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

start <- Sys.time()
set.seed(0239)
# knn_tune <- tune_bayes(knn_mod_spec, data_recipe, folds, metrics = metric_set(rmse, rsq, mae, mape, smape),initial = 6, iter = 20)
knn_tune <- tune_grid(knn_mod_spec, data_recipe, folds, metrics = metric_set(rmse, rsq, mae, mape, smape), grid = knn_grid, control = ctrl)
end <- Sys.time()
print(end - start)

show_best(knn_tune, metric = "rmse")
 
# neighbors weight_func  dist_power .metric .estimator  mean     n std_err .config              
# <int> <chr>             <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                
# 1         7 biweight              1 rmse    standard    2.47    10  0.0431 Preprocessor1_Model03
# 2        10 biweight              1 rmse    standard    2.49    10  0.0424 Preprocessor1_Model04
# 3         7 triangular            1 rmse    standard    2.49    10  0.0419 Preprocessor1_Model15
# 4         4 triangular            1 rmse    standard    2.50    10  0.0440 Preprocessor1_Model14
# 5         4 epanechnikov          1 rmse    standard    2.51    10  0.0445 Preprocessor1_Model06

knn_best <- knn_tune %>% select_best(metric = "rmse")

# knn_final_wf <- 
#   workflow() %>% 
#   add_formula(BIS ~ .) %>% 
#   add_model(knn_mod_spec) %>% 
#   finalize_workflow(knn_best)

knn_final_spec <-
  nearest_neighbor(
    neighbors = 7,
    weight_func = "biweight",
    dist_power = 1
  ) %>%
  # This model can be used for classification or regression, so set mode
  set_mode("regression") %>%
  set_engine("kknn")

knn_final_wf <- 
  workflow() %>% 
  add_model(knn_final_spec) %>% 
  add_recipe(data_recipe)

start <- Sys.time()
set.seed(234)
knn_final_fit <- 
  knn_final_wf %>% 
  fit(data = eeg_data_train)
end <- Sys.time()
print(end - start)

# TRaining metrics
pred <- predict(knn_final_fit, eeg_data_train)
pred_df <- bind_cols(eeg_data_train, pred)

svr_metrics <- yardstick::metric_set(rmse, mae, mape, rsq)
# knn_rsq <- cor(pred_df$BIS, pred_df$.pred)^2

knn_training_metrics <- svr_metrics(pred_df, truth = BIS, estimate = .pred) %>% 
  select(.metric, .estimate) %>% 
  dplyr::rename(metric = .metric, knn.train = .estimate)

# Test metrics

pred <- predict(knn_final_fit, eeg_data_test)
pred_df <- bind_cols(eeg_data_test, pred)

svr_metrics <- metric_set(rmse, rsq, mae, mape)

knn_testing_metrics <- svr_metrics(pred_df, truth = BIS, estimate = .pred) %>% 
  select(.metric, .estimate) %>% 
  dplyr::rename(metric = .metric, knn.test = .estimate)

knn_final_metrics <- knn_training_metrics %>% 
  inner_join(knn_testing_metrics, by = "metric")

final_metrics <- knn_final_metrics %>% 
  inner_join(rf_final_metrics, by = "metric")

# Pearson coefficient
# applying sheet names to dataframe names
data_frame = lapply(setNames(sheet[11:15], sheet[11:15]), 
                    function(x) read_excel("Project data set 1 (for reports 1 and 3) .xlsx", sheet=x))


eeg_data_test_per_case = bind_rows(data_frame, .id="Sheet")

pred <- predict(knn_final_fit, eeg_data_test_per_case)
pred_df <- bind_cols(eeg_data_test_per_case, pred)

knn_pearson_per_case <- pred_df %>% 
  dplyr::rename(case = Sheet) %>% 
  dplyr::group_by(case) %>% 
  dplyr::summarise(knn.pearson = cor(BIS, .pred, method = "pearson"))

ggplot(knn_pearson_per_case, aes(x = case, y = knn.pearson, group = 1)) + 
  geom_point() +
  geom_line() + 
  labs(y = "Correlation Coafficient", x = "Case") + 
  ylim(0:1)

#create Bland-Altman plot

pred <- predict(knn_final_fit, eeg_data_test)
pred_knn <- bind_cols(eeg_data_test, pred) %>% 
  dplyr::rename(knn_pred=.pred) %>% 
  rowwise() %>%
  dplyr::mutate(avg = mean(c_across(c('BIS', 'knn_pred')), na.rm=TRUE),
                diff = BIS - knn_pred,
                perc = diff/avg)

(mean_diff <- mean(pred_knn$diff))
(lower <- mean_diff - 2*sd(pred_knn$diff))
(upper <- mean_diff + 2*sd(pred_knn$diff))

(agreement <- (sum(pred_knn$diff)/sum(pred_knn$avg))*100)
(agreement <- mean(pred_knn$diff/pred_knn$avg))


p1 <- ggplot(pred_knn, aes(x = avg, y = diff)) +
  geom_point(size=2) +
  geom_hline(yintercept = mean_diff) +
  geom_hline(yintercept = lower, color = "red", linetype="dashed", linewidth = 1) +
  geom_hline(yintercept = upper, color = "red", linetype="dashed", linewidth = 1) +
  annotate("text", label = paste0("upper = ",round(upper,2)), x = 5, y = 22) +
  annotate("text", label = paste0("bias = ",round(mean_diff,2)), x = 3, y = 1.5) +
  annotate("text", label = paste0("lower = ",round(lower,2)), x = 5, y = -18) +
  xlim(0,100) +
  ggtitle("Bland-Altman Plot (5 Test cases)") +
  ylab("Difference Between Measurements") +
  xlab("Average Measurement")
p1


library(ggExtra)
# print(ggMarginal(bland.altman.plot(pred_rf$BIS, pred_rf$rf_pred, graph.sys = "ggplot2"),
#                  type = "histogram", size=4))

ggMarginal(p1, type = "histogram", size=5)

ggplot(pred_knn, aes(x = BIS, y = knn_pred)) + 
  geom_point() + 
  geom_abline(intercept = 0,
              slope = 1,
              color = "red",
              size = 2) + 
  ggtitle("Predicted vs Actual")
