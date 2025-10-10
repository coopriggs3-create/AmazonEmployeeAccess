library(tidymodels)
library(ggmosaic)
library(vroom)
library(ggplot2)
library(vcd)
library(embed)


# Read in test and train data sets
am_train <- vroom("train.csv")
am_test <- vroom("test.csv")


# Exploratory Data Analysis
head(am_train)
summary(am_train)
am_train$ACTION <- as.factor(am_train$ACTION)

# Mosaic Plot



ggplot(am_train, aes(x = ACTION, y = ROLE_TITLE, fill = ACTION)) +
  geom_boxplot() +
  labs(title = "Boxplot of RESOURCE by ACTION",
       x = "ACTION (0 = Denied, 1 = Approved)", y = "RESOURCE ID") +
  theme_minimal()


library(vcd)

mosaic(~ ROLE_TITLE + ACTION, data = am_train,
       shade = TRUE, legend = TRUE,
       main = "Mosaic Plot: ROLE_TITLE vs ACTION")

# RECIPE
am_recipe <- recipe(rFormula, data = am_train) |>
  step_mutate_at(vars_I_want_to_mutate, fn = factor) |>
  step_other(vars_I_want_other_cat_in, threshold = .001) |>
  step_dummy(vars_I_want_to_dummy) |>
  step_lencode_mixed(vars_I_want_to_target_encode, outcome = vars(target_var))

am_recipe <- recipe(ACTION ~ ., data = am_train) |>
  step_mutate_at(all_predictors(), fn = factor) |>
  step_other(all_nominal_predictors(), threshold = 0.001) |>
  step_dummy(all_nominal_predictors())

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
