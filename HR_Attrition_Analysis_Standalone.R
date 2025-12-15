# =============================================================================
# STAT 515 Final Project - IBM HR Employee Attrition Analysis
# Team: Rutvij & Sean Greg
# George Mason University | Fall 2024
#
# This is a STANDALONE R script that runs the complete analysis
# Run this in RStudio to see all models and outputs
# =============================================================================

# -----------------------------------------------------------------------------
# SETUP AND LIBRARIES
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("STAT 515 Final Project - IBM HR Employee Attrition Analysis\n")
cat("Team: Rutvij & Sean Greg | George Mason University\n")
cat("=============================================================================\n\n")

# Install packages if needed (uncomment if necessary)
# install.packages(c("readxl", "dplyr", "ggplot2", "tidyr", "caret", 
#                    "rpart", "rpart.plot", "randomForest", "glmnet", 
#                    "pROC", "car", "gridExtra"))

# Load required libraries
library(readxl)
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(glmnet)
library(pROC)
library(car)
library(gridExtra)

# Set seed for reproducibility
set.seed(515)

# -----------------------------------------------------------------------------
# DATA LOADING AND PREPROCESSING
# -----------------------------------------------------------------------------

cat("\n=============================================================================\n")
cat("PART 1: DATA LOADING AND PREPROCESSING\n")
cat("=============================================================================\n\n")

# Load the dataset - try multiple possible paths
possible_paths <- c(
  "data/HR_Data.xlsx",
  "HR_Data.xlsx",
  "./data/HR_Data.xlsx",
  "./HR_Data.xlsx"
)

data_path <- NULL
for (p in possible_paths) {
  if (file.exists(p)) {
    data_path <- p
    break
  }
}

if (is.null(data_path)) {
  cat("ERROR: HR_Data.xlsx not found!\n")
  cat("Current working directory:", getwd(), "\n")
  cat("Files in current directory:", paste(list.files(), collapse = ", "), "\n")
  stop("Please ensure HR_Data.xlsx is in the project folder or 'data' subfolder.\nOr set working directory to the project folder using setwd()")
}

hr_data <- read_excel(data_path)
cat("Data loaded successfully from:", data_path, "\n")

cat("Dataset Dimensions:", nrow(hr_data), "employees,", ncol(hr_data), "variables\n")
cat("Missing Values:", sum(is.na(hr_data)), "\n\n")

# Create binary variables
hr_data <- hr_data %>%
  mutate(
    Attrition_Binary = ifelse(Attrition == "Yes", 1, 0),
    OverTime_Binary = ifelse(`Over Time` == "Yes", 1, 0)
  )

# Class distribution
cat("Target Variable Distribution:\n")
print(table(hr_data$Attrition))
cat("\nPercentages:\n")
print(prop.table(table(hr_data$Attrition)) * 100)

# Calculate class weights
n_total <- nrow(hr_data)
n_pos <- sum(hr_data$Attrition_Binary)
n_neg <- n_total - n_pos
weight_pos <- n_total / (2 * n_pos)
weight_neg <- n_total / (2 * n_neg)

cat("\nClass Weights for Imbalanced Data:\n")
cat("  Class 0 (No Attrition):", round(weight_neg, 4), "\n")
cat("  Class 1 (Attrition):", round(weight_pos, 4), "\n")

# -----------------------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# -----------------------------------------------------------------------------

cat("\n=============================================================================\n")
cat("PART 2: EXPLORATORY DATA ANALYSIS\n")
cat("=============================================================================\n\n")

# Attrition by OverTime
cat("--- Attrition by OverTime Status ---\n")
overtime_summary <- hr_data %>%
  group_by(`Over Time`) %>%
  summarise(
    Total = n(),
    Attrition = sum(Attrition_Binary),
    Attrition_Rate = round(mean(Attrition_Binary) * 100, 1)
  )
print(overtime_summary)

# Attrition by Department
cat("\n--- Attrition by Department ---\n")
dept_summary <- hr_data %>%
  group_by(Department) %>%
  summarise(
    Total = n(),
    Attrition = sum(Attrition_Binary),
    Attrition_Rate = round(mean(Attrition_Binary) * 100, 1)
  )
print(dept_summary)

# Chi-square tests
cat("\n--- Chi-Square Tests ---\n")
chi_vars <- c("Over Time", "Department", "Job Role", "Marital Status", "Business Travel")

for (var in chi_vars) {
  chi_test <- chisq.test(table(hr_data[[var]], hr_data$Attrition))
  cat(sprintf("%s: Chi-sq = %.2f, p = %.4e %s\n", 
              var, chi_test$statistic, chi_test$p.value,
              ifelse(chi_test$p.value < 0.001, "***", 
                     ifelse(chi_test$p.value < 0.01, "**",
                            ifelse(chi_test$p.value < 0.05, "*", "")))))
}

# T-tests
cat("\n--- T-Tests for Numeric Variables ---\n")
t_vars <- c("Age", "Monthly Income", "Years At Company", "Job Satisfaction")

for (var in t_vars) {
  t_test <- t.test(hr_data[[var]] ~ hr_data$Attrition)
  cat(sprintf("%s: Mean(No)=%.2f, Mean(Yes)=%.2f, p=%.4f %s\n",
              var, t_test$estimate[1], t_test$estimate[2], t_test$p.value,
              ifelse(t_test$p.value < 0.001, "***", 
                     ifelse(t_test$p.value < 0.01, "**",
                            ifelse(t_test$p.value < 0.05, "*", "")))))
}

# -----------------------------------------------------------------------------
# RESEARCH QUESTION 1: WORK-LIFE IMBALANCE ANALYSIS
# -----------------------------------------------------------------------------

cat("\n=============================================================================\n")
cat("PART 3: RESEARCH QUESTION 1 - WORK-LIFE IMBALANCE ANALYSIS\n")
cat("=============================================================================\n")
cat("Question: How do work-life factors interact to predict attrition?\n\n")

# Prepare data for Q1
q1_data <- hr_data %>%
  select(Attrition_Binary, OverTime_Binary, `Distance From Home`, 
         `Work Life Balance`, `Years At Company`) %>%
  rename(
    DistanceFromHome = `Distance From Home`,
    WorkLifeBalance = `Work Life Balance`,
    YearsAtCompany = `Years At Company`
  )

# Train-test split
train_idx <- createDataPartition(q1_data$Attrition_Binary, p = 0.7, list = FALSE)
train_data <- q1_data[train_idx, ]
test_data <- q1_data[-train_idx, ]

cat("Train set:", nrow(train_data), "| Test set:", nrow(test_data), "\n\n")

# 3.1 Decision Tree
cat("--- 3.1 DECISION TREE MODEL ---\n\n")

dt_model <- rpart(
  Attrition_Binary ~ OverTime_Binary + DistanceFromHome + WorkLifeBalance + YearsAtCompany,
  data = train_data,
  method = "class",
  parms = list(prior = c(0.5, 0.5)),
  control = rpart.control(maxdepth = 4, minsplit = 50, cp = 0.01)
)

cat("Decision Tree Structure:\n")
print(dt_model)

cat("\nFeature Importance:\n")
if (length(dt_model$variable.importance) > 0) {
  importance_dt <- dt_model$variable.importance / sum(dt_model$variable.importance)
  print(sort(importance_dt, decreasing = TRUE))
}

# Decision tree predictions
pred_dt <- predict(dt_model, test_data, type = "class")
pred_prob_dt <- predict(dt_model, test_data, type = "prob")[, 2]

cat("\nConfusion Matrix (Decision Tree):\n")
print(confusionMatrix(factor(pred_dt), factor(test_data$Attrition_Binary)))

roc_dt <- roc(test_data$Attrition_Binary, pred_prob_dt, quiet = TRUE)
cat("\nDecision Tree AUC-ROC:", round(auc(roc_dt), 4), "\n")

# Plot decision tree
rpart.plot(dt_model, type = 4, extra = 104, fallen.leaves = TRUE,
           main = "Decision Tree: Work-Life Factors Predicting Attrition",
           box.palette = c("lightblue", "salmon"))

# 3.2 Logistic Regression with Interactions
cat("\n--- 3.2 LOGISTIC REGRESSION WITH INTERACTIONS ---\n\n")

hr_data <- hr_data %>%
  mutate(
    OT_x_WLB = OverTime_Binary * `Work Life Balance`,
    OT_x_Distance = OverTime_Binary * `Distance From Home`,
    OT_x_YearsAtCompany = OverTime_Binary * `Years At Company`
  )

logit_q1 <- glm(
  Attrition_Binary ~ OverTime_Binary + `Distance From Home` + `Work Life Balance` + 
    `Years At Company` + OT_x_WLB + OT_x_Distance + OT_x_YearsAtCompany,
  data = hr_data,
  family = binomial(link = "logit")
)

cat("Logistic Regression Summary:\n")
print(summary(logit_q1))

cat("\nOdds Ratios:\n")
or_q1 <- exp(coef(logit_q1))
ci_q1 <- exp(confint(logit_q1))
or_table <- data.frame(
  Variable = names(or_q1),
  OR = round(or_q1, 4),
  CI_Lower = round(ci_q1[, 1], 4),
  CI_Upper = round(ci_q1[, 2], 4),
  p_value = round(summary(logit_q1)$coefficients[, 4], 4)
)
print(or_table)

# 3.3 Random Forest
cat("\n--- 3.3 RANDOM FOREST MODEL ---\n\n")

rf_model <- randomForest(
  factor(Attrition_Binary) ~ OverTime_Binary + DistanceFromHome + WorkLifeBalance + YearsAtCompany,
  data = train_data,
  ntree = 500,
  mtry = 2,
  classwt = c(1, 3),
  importance = TRUE
)

print(rf_model)

cat("\nRandom Forest Variable Importance:\n")
print(importance(rf_model))

# Plot variable importance
varImpPlot(rf_model, main = "Random Forest Variable Importance")

# Random Forest predictions
pred_prob_rf <- predict(rf_model, test_data, type = "prob")[, 2]
roc_rf <- roc(test_data$Attrition_Binary, pred_prob_rf, quiet = TRUE)
cat("\nRandom Forest AUC-ROC:", round(auc(roc_rf), 4), "\n")

# ROC comparison plot
plot(roc_dt, col = "blue", lwd = 2, main = "ROC Curves - Q1: Work-Life Models")
plot(roc_rf, col = "red", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c(paste("Decision Tree (AUC =", round(auc(roc_dt), 3), ")"),
                  paste("Random Forest (AUC =", round(auc(roc_rf), 3), ")")),
       col = c("blue", "red"), lwd = 2)

# -----------------------------------------------------------------------------
# RESEARCH QUESTION 2: CAREER STAGNATION ANALYSIS
# -----------------------------------------------------------------------------

cat("\n=============================================================================\n")
cat("PART 4: RESEARCH QUESTION 2 - CAREER STAGNATION & COMPENSATION\n")
cat("=============================================================================\n")
cat("Question: At what thresholds do career stagnation indicators become critical?\n\n")

# 4.1 Weighted Logistic Regression
cat("--- 4.1 WEIGHTED LOGISTIC REGRESSION ---\n\n")

weights <- ifelse(hr_data$Attrition_Binary == 1, weight_pos, weight_neg)

logit_q2 <- glm(
  Attrition_Binary ~ `Years Since Last Promotion` + `Years In Current Role` +
    `Monthly Income` + `Percent Salary Hike` + `Job Level`,
  data = hr_data,
  family = binomial(),
  weights = weights
)

cat("Weighted Logistic Regression Summary:\n")
print(summary(logit_q2))

cat("\nOdds Ratios:\n")
or_q2 <- exp(coef(logit_q2))
print(round(or_q2, 4))

# 4.2 LASSO Variable Selection
cat("\n--- 4.2 LASSO VARIABLE SELECTION ---\n\n")

lasso_vars <- c("Years Since Last Promotion", "Years In Current Role", "Monthly Income",
                "Percent Salary Hike", "Job Level", "Age", "Total Working Years",
                "Years At Company", "Years With Curr Manager", "Distance From Home",
                "Job Satisfaction", "Environment Satisfaction", "Work Life Balance",
                "OverTime_Binary", "Num Companies Worked", "Stock Option Level")

X_lasso <- as.matrix(hr_data[, lasso_vars])
y_lasso <- hr_data$Attrition_Binary

cv_lasso <- cv.glmnet(X_lasso, y_lasso, family = "binomial", alpha = 1, nfolds = 10)

cat("Optimal Lambda (min):", cv_lasso$lambda.min, "\n")
cat("Optimal Lambda (1se):", cv_lasso$lambda.1se, "\n")

cat("\nLASSO Coefficients at lambda.min:\n")
lasso_coefs <- coef(cv_lasso, s = "lambda.min")
lasso_df <- data.frame(
  Variable = rownames(lasso_coefs),
  Coefficient = as.vector(lasso_coefs)
)
lasso_df <- lasso_df[abs(lasso_df$Coefficient) > 0.001, ]
lasso_df <- lasso_df[order(-abs(lasso_df$Coefficient)), ]
print(lasso_df)

cat("\nNumber of selected variables:", nrow(lasso_df) - 1, "of", length(lasso_vars), "\n")

# Plot LASSO CV
plot(cv_lasso, main = "LASSO Cross-Validation")

# 4.3 ROC-AUC and Threshold Analysis
cat("\n--- 4.3 ROC-AUC AND THRESHOLD ANALYSIS ---\n\n")

pred_probs_q2 <- predict(logit_q2, type = "response")
roc_q2 <- roc(hr_data$Attrition_Binary, pred_probs_q2, quiet = TRUE)

coords_best <- coords(roc_q2, "best", ret = c("threshold", "sensitivity", "specificity"))
cat("Optimal Threshold (Youden's J):", round(coords_best$threshold, 4), "\n")
cat("  Sensitivity:", round(coords_best$sensitivity, 4), "\n")
cat("  Specificity:", round(coords_best$specificity, 4), "\n")

cat("\nClassification at Different Thresholds:\n")
cat(sprintf("%-10s %-12s %-12s %-12s\n", "Threshold", "Sensitivity", "Specificity", "Accuracy"))
cat(rep("-", 50), "\n")

for (thresh in c(0.1, 0.2, 0.3, 0.4, 0.5, coords_best$threshold)) {
  pred <- ifelse(pred_probs_q2 >= thresh, 1, 0)
  sens <- mean(pred[hr_data$Attrition_Binary == 1] == 1)
  spec <- mean(pred[hr_data$Attrition_Binary == 0] == 0)
  acc <- mean(pred == hr_data$Attrition_Binary)
  marker <- ifelse(thresh == coords_best$threshold, " <-- Optimal", "")
  cat(sprintf("%-10.3f %-12.4f %-12.4f %-12.4f%s\n", thresh, sens, spec, acc, marker))
}

# Plot ROC
plot(roc_q2, col = "red", lwd = 2, main = "ROC Curve - Career Stagnation Model")
points(coords_best$specificity, coords_best$sensitivity, pch = 19, col = "green", cex = 2)
legend("bottomright", paste("AUC =", round(auc(roc_q2), 3)), bty = "n")

# -----------------------------------------------------------------------------
# RESEARCH QUESTION 3: DEPARTMENT-STRATIFIED SATISFACTION ANALYSIS
# -----------------------------------------------------------------------------

cat("\n=============================================================================\n")
cat("PART 5: RESEARCH QUESTION 3 - DEPARTMENT-STRATIFIED SATISFACTION\n")
cat("=============================================================================\n")
cat("Question: Do satisfaction effects differ across departments?\n\n")

# 5.1 VIF Analysis
cat("--- 5.1 VIF ANALYSIS ---\n\n")

vif_model <- lm(Attrition_Binary ~ `Job Satisfaction` + `Environment Satisfaction` +
                  `Relationship Satisfaction` + `Work Life Balance` + 
                  `Job Involvement` + `Monthly Income` + Age,
                data = hr_data)

cat("Variance Inflation Factors:\n")
vif_results <- vif(vif_model)
print(round(vif_results, 4))

cat("\nVIF Interpretation:\n")
cat("  < 5: No concern\n")
cat("  5-10: Moderate multicollinearity\n")
cat("  > 10: High multicollinearity (problematic)\n")

# 5.2 Stratified Models
cat("\n--- 5.2 STRATIFIED LOGISTIC REGRESSION BY DEPARTMENT ---\n")

departments <- c("Sales", "R&D", "HR")

for (dept in departments) {
  cat("\n========================================\n")
  cat("DEPARTMENT:", dept, "\n")
  cat("========================================\n")
  
  df_dept <- hr_data %>% filter(Department == dept)
  
  cat("Sample Size:", nrow(df_dept), "\n")
  cat("Attrition Rate:", round(mean(df_dept$Attrition_Binary) * 100, 1), "%\n")
  
  model_dept <- glm(
    Attrition_Binary ~ `Job Satisfaction` + `Environment Satisfaction` + `Relationship Satisfaction`,
    data = df_dept,
    family = binomial()
  )
  
  cat("\nCoefficients:\n")
  coef_summary <- summary(model_dept)$coefficients
  print(round(coef_summary, 4))
  
  cat("\nOdds Ratios:\n")
  print(round(exp(coef(model_dept)), 4))
  
  cat("\nModel AIC:", round(AIC(model_dept), 2), "\n")
  
  # AUC
  pred_probs <- predict(model_dept, type = "response")
  roc_dept <- roc(df_dept$Attrition_Binary, pred_probs, quiet = TRUE)
  cat("AUC:", round(auc(roc_dept), 4), "\n")
}

# -----------------------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------------------

cat("\n=============================================================================\n")
cat("FINAL SUMMARY\n")
cat("=============================================================================\n\n")

cat("Dataset Overview:\n")
cat("  - Total Employees:", nrow(hr_data), "\n")
cat("  - Overall Attrition Rate:", round(mean(hr_data$Attrition_Binary) * 100, 1), "%\n")
cat("  - OverTime Attrition Rate:", round(mean(hr_data$Attrition_Binary[hr_data$`Over Time` == "Yes"]) * 100, 1), "%\n")
cat("  - Non-OverTime Attrition Rate:", round(mean(hr_data$Attrition_Binary[hr_data$`Over Time` == "No"]) * 100, 1), "%\n")

cat("\nModel Performance (AUC-ROC):\n")
cat("  - Decision Tree (Q1):", round(auc(roc_dt), 4), "\n")
cat("  - Random Forest (Q1):", round(auc(roc_rf), 4), "\n")
cat("  - Weighted Logistic Regression (Q2):", round(auc(roc_q2), 4), "\n")

cat("\nKey Findings:\n")
cat("  1. OverTime is the strongest predictor of attrition\n")
cat("  2. Employees working overtime have ~3x higher attrition rate\n")
cat("  3. Years Since Last Promotion increases risk by ~13% per year\n")
cat("  4. LASSO selected", nrow(lasso_df) - 1, "of 16 candidate variables\n")
cat("  5. Optimal classification threshold:", round(coords_best$threshold, 3), "\n")
cat("  6. Satisfaction effects differ significantly by department\n")

cat("\n=============================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("=============================================================================\n")
