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

estimates <- prep(data_recipe, eeg_data_train) %>% bake(eeg_data_train)

# Defining the models
# knn
rf_mod_spec <-
  rand_forest(
    mtry = tune(),
    trees = 500,
    min_n = tune()
  ) %>%
  # This model can be used for classification or regression, so set mode
  set_mode("regression") %>%
  set_engine("ranger")

set.seed(345)
folds <- vfold_cv(eeg_data_train, v = 5)

rf_grid <- grid_regular(
  mtry(range = c(1, 5)),
  min_n(range = c(2, 8)),
  levels = 5
)

ctrl <- control_grid(verbose = FALSE, save_pred = TRUE)

library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

start <- Sys.time()
set.seed(0239)
# knn_tune <- tune_bayes(knn_mod_spec, data_recipe, folds, metrics = metric_set(rmse, rsq, mae, mape, smape),initial = 6, iter = 20)
rf_tune <- tune_grid(rf_mod_spec, data_recipe, folds, metrics = metric_set(rmse, rsq, mae, mape, smape), grid = 20, control = ctrl)
end <- Sys.time()
print(end - start)

show_best(rf_tune, metric = "rmse")

rf_best <- rf_tune %>% select_best(metric = "rmse")

# mtry min_n .metric .estimator  mean     n std_err .config              
# <int> <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>                
# 1     4     5 rmse    standard    2.70     5  0.0356 Preprocessor1_Model17
# 2     4     9 rmse    standard    2.76     5  0.0327 Preprocessor1_Model09
# 3     3    12 rmse    standard    2.82     5  0.0311 Preprocessor1_Model06
# 4     2     6 rmse    standard    2.83     5  0.0285 Preprocessor1_Model07
# 5     2    10 rmse    standard    2.91     5  0.0309 Preprocessor1_Model03

rf_final_spec <-
  rand_forest(
    mtry = 4,
    trees = 100,
    min_n = 5
  ) %>%
  # This model can be used for classification or regression, so set mode
  set_mode("regression") %>%
  set_engine("ranger")

rf_final_wf <- 
  workflow() %>% 
  add_model(rf_final_spec) %>% 
  add_recipe(data_recipe)

start <- Sys.time()
set.seed(234)
rf_final_fit <- 
  rf_final_wf %>% 
  fit(data = eeg_data_train)
end <- Sys.time()
print(end - start)

# TRaining metrics
pred <- predict(rf_final_fit, eeg_data_train)
pred_df <- bind_cols(eeg_data_train, pred)

svr_metrics <- yardstick::metric_set(rmse, mae, mape, rsq)
# knn_rsq <- cor(pred_df$BIS, pred_df$.pred)^2

rf_training_metrics <- svr_metrics(pred_df, truth = BIS, estimate = .pred) %>% 
  select(.metric, .estimate) %>% 
  dplyr::rename(metric = .metric, rf.train = .estimate)

# Test metrics

pred <- predict(rf_final_fit, eeg_data_test)
pred_df <- bind_cols(eeg_data_test, pred)

svr_metrics <- metric_set(rmse, rsq, mae, mape)

rf_testing_metrics <- svr_metrics(pred_df, truth = BIS, estimate = .pred) %>% 
  select(.metric, .estimate) %>% 
  dplyr::rename(metric = .metric, rf.test = .estimate)

rf_final_metrics <- rf_training_metrics %>% 
  inner_join(rf_testing_metrics, by = "metric")

final_metrics <- knn_final_metrics %>% 
  inner_join(rf_final_metrics, by = "metric")

# Pearson coefficient
# applying sheet names to dataframe names
data_frame = lapply(setNames(sheet[11:15], sheet[11:15]), 
                    function(x) read_excel("Project data set 1 (for reports 1 and 3) .xlsx", sheet=x))


eeg_data_test_per_case = bind_rows(data_frame, .id="Sheet")

pred <- predict(rf_final_fit, eeg_data_test_per_case)
pred_df <- bind_cols(eeg_data_test_per_case, pred)

rf_pearson_per_case <- pred_df %>% 
  dplyr::rename(case = Sheet) %>% 
  dplyr::group_by(case) %>% 
  dplyr::summarise(rf.pearson = cor(BIS, .pred, method = "pearson"))

ggplot(rf_pearson_per_case, aes(x = case, y = rf.pearson, group = 1)) + 
  geom_point() +
  geom_line() + 
  labs(y = "Correlation Coafficient", x = "Case") + 
  ylim(0:1)

knn_pearson_per_case %>% 
  inner_join(rf_pearson_per_case, by = "case") %>% 
  dplyr::rename(knn = knn.pearson, rf = rf.pearson) %>% 
  pivot_longer(cols = c(knn, rf), names_to = "model") %>%
  ggplot(aes(x = case, y = value, color = model, group = model)) + 
  geom_point() +
  geom_line() + 
  ylim(0,1)

# New Index chart
pred <- predict(rf_final_fit, eeg_data_test)
pred_rf <- bind_cols(eeg_data_test, pred) %>% 
  dplyr::rename(rf_pred=.pred)

pred <- predict(knn_final_fit, eeg_data_test) 
pred_knn <- bind_cols(eeg_data_test, pred) %>% 
  dplyr::rename(knn_pred = .pred)

test_new_index <- pred_knn %>% 
  bind_cols(pred_rf %>% select(rf_pred))

test_new_index %>% 
  select(BIS, knn_pred, rf_pred) %>% 
  dplyr::mutate(time = row_number()) %>%
  pivot_longer(cols = c(BIS, knn_pred, rf_pred), names_to = "index") %>%
  ggplot(aes(x=time, y=value, color = index)) + 
  geom_line() + 
  # geom_line(aes(y=knn_pred), color = 'lightblue') + 
  # geom_line(aes(y=rf_pred), color = 'orange') + 
  labs(x="Time, seconds", y = "index") +
  theme(legend.position = "bottom")

# new index chart by test case
pred <- predict(rf_final_fit, eeg_data_test_per_case)
pred_rf_bycase <- bind_cols(eeg_data_test_per_case, pred)

pred <- predict(knn_final_fit, eeg_data_test_per_case)
pred_knn_bycase <- bind_cols(eeg_data_test_per_case, pred)

test_new_index_bycase <- pred_rf_bycase %>% 
  dplyr::rename(rf_pred = .pred) %>% 
  bind_cols(pred_knn_bycase %>% select(.pred) %>% dplyr::rename(knn_pred = .pred)) %>% 
  select(Sheet, BIS, knn_pred, rf_pred) %>% 
  dplyr::group_by(Sheet) %>% 
  dplyr::mutate(time = row_number()) %>% 
  pivot_longer(cols = c(BIS, knn_pred, rf_pred), names_to = "index") %>% 
  ggplot(aes(x=time, y=value, color = index)) + 
  geom_line() + 
  # geom_line(aes(y=knn_pred), color = 'lightblue') + 
  # geom_line(aes(y=rf_pred), color = 'orange') + 
  labs(x="Time, seconds", y = "index") +
  theme(legend.position = "bottom") +
  facet_wrap(~Sheet, ncol = 1)

# errors by test case

  
#create Bland-Altman plot

pred <- predict(rf_final_fit, eeg_data_test)
pred_rf <- bind_cols(eeg_data_test, pred) %>% 
  dplyr::rename(rf_pred=.pred) %>% 
  rowwise() %>%
  dplyr::mutate(avg = mean(c_across(c('BIS', 'rf_pred')), na.rm=TRUE),
                diff = BIS - rf_pred,
                perc = diff/avg)

(mean_diff <- mean(pred_rf$diff))
(lower <- mean_diff - 2*sd(pred_rf$diff))
(upper <- mean_diff + 2*sd(pred_rf$diff))

(agreement <- (sum(pred_rf$diff)/sum(pred_rf$avg))*100)
(agreement <- mean(pred_rf$diff/pred_rf$avg))


p1 <- ggplot(pred_rf, aes(x = avg, y = diff)) +
  geom_point(size=2) +
  geom_hline(yintercept = mean_diff) +
  geom_hline(yintercept = lower, color = "red", linetype="dashed", linewidth = 1) +
  geom_hline(yintercept = upper, color = "red", linetype="dashed", linewidth = 1) +
  annotate("text", label = as.character(round(upper,2)), x = 4, y = 20.5) +
  annotate("text", label = as.character(round(mean_diff,2)), x = 4, y = 1.5) +
  annotate("text", label = as.character(round(lower,2)), x = 5, y = -16) +
  xlim(0,100) +
  ggtitle("RF Bland-Altman Plot (5 Test cases)") +
  ylab("Difference Between Measurements") +
  xlab("Average Measurement")

p2 <- ggplot(pred_knn, aes(x = avg, y = diff)) +
  geom_point(size=2) +
  geom_hline(yintercept = mean_diff) +
  geom_hline(yintercept = lower, color = "red", linetype="dashed", linewidth = 1) +
  geom_hline(yintercept = upper, color = "red", linetype="dashed", linewidth = 1) +
  annotate("text", label = paste0("upper = ",round(upper,2)), x = 5, y = 22) +
  annotate("text", label = paste0("bias = ",round(mean_diff,2)), x = 3, y = 1.5) +
  annotate("text", label = paste0("lower = ",round(lower,2)), x = 5, y = -18) +
  xlim(0,100) +
  ggtitle("KNN Bland-Altman Plot (5 Test cases)") +
  ylab("Difference Between Measurements") +
  xlab("Average Measurement")


library(ggExtra)
library(cowplot)
# print(ggMarginal(bland.altman.plot(pred_rf$BIS, pred_rf$rf_pred, graph.sys = "ggplot2"),
#                  type = "histogram", size=4))

p1.1 <- ggMarginal(p1, type = "histogram", size=5)
p1.2 <- ggMarginal(p2, type = "histogram", size=5)

plot_grid(p1.1, p1.2, labels=c("KNN", "RF"), ncol = 2, nrow = 1)

p3 <- ggplot(pred_knn, aes(x = BIS, y = knn_pred)) + 
  geom_point() + 
  geom_abline(intercept = 0,
              slope = 1,
              color = "red",
              size = 2)

p4 <- ggplot(pred_rf, aes(x = BIS, y = rf_pred)) + 
  geom_point() + 
  geom_abline(intercept = 0,
              slope = 1,
              color = "red",
              size = 2)

plot_grid(p3, p4, labels=c("KNN", "RF"), ncol = 2, nrow = 1)
