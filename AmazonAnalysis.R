library(tidymodels)
library(vroom)
library(embed)
library(ranger)


# Read in test and train data sets
am_train <- vroom("train.csv")
am_test <- vroom("test.csv")

am_train$ACTION <- as.factor(am_train$ACTION)

# RECIPE
am_recipe <- recipe(ACTION ~ ., data = am_train) |>
  step_mutate(across(all_predictors(), as.factor))|>
  step_novel(all_nominal_predictors()) |>
  step_other(all_nominal_predictors(), threshold = 0.001) |>
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |>
  step_zv(all_predictors())

prep <- prep(am_recipe)
baked <- bake(prep, new_data = am_train)
dim(baked)


# Logistic Regression
logRegmodel <- logistic_reg() |>
  set_engine('glm')

logReg_wf <- workflow() |>
  add_recipe(am_recipe) |>
  add_model(logRegmodel) |>
  fit(data = am_train)

amazon_predictions <- predict(logReg_wf,
                              new_data = am_test,
                              type = 'prob')

amazon_predictions <- logReg_wf |>
  predict(new_data = am_test, type = "prob") |>
  bind_cols(am_test) |>
  rename(ACTION=.pred_1) |>
  select(id, ACTION)
vroom_write(x = amazon_predictions, file = "./LogRegPreds.csv", delim = ",")


# Penalized Logistic Regression
penLogReg_model <- logistic_reg(mixture = tune(), penalty = tune()) |>
  set_engine('glmnet')

penLogReg_wf <- workflow() |>
  add_recipe(am_recipe) |>
  add_model(penLogReg_model)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 2)

folds <- vfold_cv(am_train, v = 2, repeats = 1)

CV_results_pen <- penLogReg_wf |>
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune_pen <- CV_results_pen |>
  select_best()

final_wf_pen <- penLogReg_wf |>
  finalize_workflow(bestTune_pen) |>
  fit(data = am_train)

final_wf_pen |>
  predict(new_data = am_test, type = 'prob')

penlog_predictions <- final_wf_pen |>
  predict(new_data = am_test, type = "prob") |>
  bind_cols(am_test) |>
  rename(ACTION=.pred_1) |>
  select(id, ACTION)
vroom_write(x = penlog_predictions, file = "./PenLogPreds.csv", delim = ",")

# Random Forests
rf_mod <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 500
) |>
  set_engine("ranger") |>
  set_mode("classification")

rf_wf <- workflow() |>
  add_recipe(am_recipe) |>
  add_model(rf_mod)

folds_rf <- vfold_cv(am_train, v = 5, strata = ACTION)

rf_grid <- grid_regular(mtry(range = c(1, 9)),
                              min_n(),
                              levels = 3)

CV_results_rf <- rf_wf |>
  tune_grid(resamples = folds_rf,
            grid = rf_grid,
            metrics = metric_set(roc_auc)) 

rf_best <- CV_results_rf |>
  select_best(metric = "roc_auc")

rf_final <- finalize_workflow(rf_wf, rf_best) |>
  fit(data = am_train)

rf_predictions <- rf_final |>
  predict(new_data = am_test, type = "prob") |>
  bind_cols(am_test) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)
 
vroom_write(rf_predictions, "./RFPreds.csv", delim = ",")


rf_mod_final <- rand_forest(
  mtry  = rf_best$mtry,
  min_n = rf_best$min_n,
  trees = 1500   # or 800–1500, whatever your machine tolerates
) |>
  set_engine("ranger") |>
  set_mode("classification")

rf_final <- workflow() |>
  add_recipe(am_recipe) |>   # your good dummy recipe
  add_model(rf_mod_final) |>
  fit(data = am_train)



# KNN
library(kknn)

knn_mod <- nearest_neighbor(neighbors = tune()) |>
  set_mode("classification") |>
  set_engine("kknn")

knn_wf <- workflow() |>
  add_recipe(am_recipe) |>
  add_model(knn_mod)

folds_knn <- vfold_cv(am_train, v = 5, strata = ACTION)

knn_grid <- grid_regular(neighbors(range=c(1L, 101L)),
                         levels = 25)

CV_results_knn <- knn_wf |>
  tune_grid(resamples = folds_knn,
            grid = knn_grid,
            metrics = metric_set(roc_auc))

knn_best <- CV_results_knn |>
  select_best(metric = "roc_auc")

knn_final <- finalize_workflow(knn_wf, knn_best) |>
  fit(data = am_train)

knn_predictions <- knn_final |>
  predict(new_data = am_test, type = "prob") |>
  bind_cols(am_test) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(knn_predictions, "./KNNPreds.csv", delim = ",")

# Naive Bayes
library(discrim)
library(naivebayes)
nb_mod <- naive_Bayes(Laplace = tune(),
                      smoothness = tune()) |>
  set_mode("classification") |>
  set_engine("naivebayes")

nb_wf <- workflow() |>
  add_recipe(am_recipe) |>
  add_model(nb_mod)

folds_nb <- vfold_cv(am_train, v = 5, strata = ACTION)

nb_grid <- grid_regular(Laplace(),
                        smoothness(),
                        levels = 3)

CV_results_nb <- nb_wf |>
  tune_grid(resamples = folds_nb,
            grid = nb_grid,
            metrics = metric_set(roc_auc))

nb_best <- CV_results_nb |>
  select_best(metric = "roc_auc")

nb_final <- finalize_workflow(nb_wf, nb_best) |>
  fit(data = am_train)

nb_predictions <- nb_final |>
  predict(new_data = am_test, type = "prob") |>
  bind_cols(am_test) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(nb_predictions, "./NBPreds.csv", delim = ",")

# Neural Networks

# Replace with YOUR path up to \cmd
git_path <- "C:/Program Files/Git/cmd"

Sys.setenv(PATH = paste(git_path, Sys.getenv("PATH"), sep = ";"))
system("git --version")   # should now print a version, not 127

library(remotes)
remotes::install_github("rstudio/tensorflow")
reticulate::install_python()
library(keras)
keras::install_keras()


# Imbalanced Data SMOTE
library(themis)
library(discrim)
library(naivebayes)

am_recipe_smote <- recipe(ACTION ~ ., data = am_train) |>
  step_mutate_at(all_predictors(), fn = factor) |>
  step_other(all_nominal_predictors(), threshold = 0.001) |>
  step_dummy(all_nominal_predictors()) |>
  step_smote(all_outcomes(), neighbors = 5) |>
  step_normalize(all_predictors())

prep_smote <- prep(am_recipe_smote)
baked_smote <- bake(prep, new_data = am_train)

nb_mod_smote <- naive_Bayes(Laplace = tune(),
                            smoothness = tune()) |>
  set_mode("classification") |>
  set_engine("naivebayes")

nb_wf_smote <- workflow() |>
  add_recipe(am_recipe_smote) |>
  add_model(nb_mod_smote)

folds_nb_smote <- vfold_cv(am_train, v = 5, strata = ACTION)

nb_grid_smote <- grid_regular(
  Laplace(),
  smoothness(),
  levels = 5
)


CV_results_nb_smote <- nb_wf_smote |>
  tune_grid(
    resamples = folds_nb_smote,
    grid = nb_grid_smote,
    metrics = metric_set(roc_auc)
  )


nb_best_smote <- CV_results_nb_smote |>
  select_best(metric = "roc_auc")

nb_final_smote <- nb_wf_smote |>
  finalize_workflow(nb_best_smote) |>
  fit(data = am_train)


nb_predictions_smote <- nb_final_smote |>
  predict(new_data = am_test, type = "prob") |>
  bind_cols(am_test) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(nb_predictions_smote, "./NBPreds_SMOTE.csv", delim = ",")
