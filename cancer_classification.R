############################################################
# breast_cancer_project.R
#
# Demonstration of a Machine Learning workflow for breast
# cancer biopsy classification (Benign vs. Malignant)
#
# This script:
#  1) Loads the dslabs::brca dataset
#  2) Performs basic EDA
#  3) Scales the features
#  4) Applies PCA for visualization
#  5) Splits data into training and test sets
#  6) Trains multiple classification models:
#      - Logistic Regression
#      - Loess (gamLoess)
#      - k-Nearest Neighbors
#      - Random Forest
#  7) Creates an ensemble and compares accuracy
#  8) Suppresses repeated warnings from gamLoess
############################################################

## 0) Libraries and Options
options(digits = 3)
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)

############################################################
# 1) Load Data
############################################################
data(brca)
# brca$y: factor of "B" or "M"
# brca$x: a numeric matrix with tumor features

cat("\n-- Data Dimensions --\n")
cat("Number of samples:", nrow(brca$x), "\n")
cat("Number of predictors:", ncol(brca$x), "\n")

cat("\n-- Class Distribution --\n")
prop_benign <- mean(brca$y == "B")
prop_malign <- mean(brca$y == "M")
cat("Proportion benign:", prop_benign, "\n")
cat("Proportion malignant:", prop_malign, "\n")

############################################################
# 2) Basic EDA: means, sds
############################################################
feature_means <- colMeans(brca$x)
feature_sds   <- colSds(brca$x)

cat("\n-- Basic Stats for Predictors --\n")
cat("Highest mean predictor:", which.max(feature_means),
    "(Mean =", max(feature_means), ")\n")
cat("Lowest SD predictor:", which.min(feature_sds),
    "(SD =", min(feature_sds), ")\n")

############################################################
# 3) Scale the Predictors
############################################################
x_centered <- sweep(brca$x, 2, colMeans(brca$x), FUN="-")
x_scaled   <- sweep(x_centered, 2, colSds(brca$x), FUN="/")

# Quick check
cat("\n-- Scaled Matrix Check --\n")
cat("SD of column 1:", round(sd(x_scaled[,1]), 3), "\n")
cat("Median of column 1:", round(median(x_scaled[,1]), 3), "\n")

############################################################
# 4) PCA for Visualization
############################################################
pca_obj <- prcomp(x_scaled)

# Proportion of variance PC1
var_explained <- pca_obj$sdev^2 / sum(pca_obj$sdev^2)
cat("\n-- PCA --\n")
cat("Variance explained by PC1:", round(var_explained[1], 3), "\n")

# How many PCs to reach >= 90%
cumvar <- cumsum(var_explained)
pc_90 <- which(cumvar >= 0.90)[1]
cat("Number of PCs to get >= 90% var explained:", pc_90, "\n")

# (Optional) We could visualize the first two PCs with ggplot here.

############################################################
# 5) Train/Test Split
############################################################
set.seed(1)
test_index <- createDataPartition(brca$y, p=0.2, list=FALSE)
train_x <- x_scaled[-test_index, ]
train_y <- brca$y[-test_index]
test_x  <- x_scaled[test_index,  ]
test_y  <- brca$y[test_index]

cat("\n-- Train/Test Split --\n")
cat("Train size:", nrow(train_x), "\n")
cat("Test size:",  nrow(test_x),  "\n")
cat("Train proportion benign:", mean(train_y=="B"), "\n")
cat("Test proportion benign:",  mean(test_y=="B"),  "\n")

############################################################
# 6) Model Training
############################################################

## A) Logistic Regression
set.seed(1)
train_glm <- train(train_x, train_y, method="glm")
yhat_glm  <- predict(train_glm, test_x)
acc_glm   <- mean(yhat_glm == test_y)

## B) Loess (gamLoess)
# We suppress warnings from the gamLoess code:
set.seed(5)
suppressWarnings({
  train_loess <- train(train_x, train_y, method="gamLoess")
})
yhat_loess  <- predict(train_loess, test_x)
acc_loess   <- mean(yhat_loess == test_y)

## C) k-Nearest Neighbors
set.seed(7)
knn_grid <- data.frame(k=seq(3,21,2))
train_knn <- train(train_x, train_y, method="knn", tuneGrid=knn_grid)
yhat_knn  <- predict(train_knn, test_x)
acc_knn   <- mean(yhat_knn == test_y)

## D) Random Forest
set.seed(9)
rf_grid <- data.frame(mtry=c(3,5,7,9))
train_rf <- train(train_x, train_y, method="rf",
                  tuneGrid=rf_grid, importance=TRUE)
yhat_rf <- predict(train_rf, test_x)
acc_rf  <- mean(yhat_rf == test_y)

# variable importance
imp <- varImp(train_rf)$importance
# For 2-class: columns "B" and "M"
top_var <- rownames(imp)[which.max(imp$M)]

############################################################
# 7) Ensemble
############################################################
ensemble_matrix <- cbind(glm = yhat_glm,
                         loess = yhat_loess,
                         knn = yhat_knn,
                         rf = yhat_rf)
ensemble_vote <- ifelse(rowMeans(ensemble_matrix=="M") > 0.5, "M", "B")
acc_ensemble <- mean(ensemble_vote == test_y)

############################################################
# 8) Summary of Results
############################################################
all_results <- tibble(
  Model = c("Logistic Regression","Loess","kNN","Random Forest","Ensemble"),
  Accuracy = c(acc_glm, acc_loess, acc_knn, acc_rf, acc_ensemble)
)

cat("\n-- Results Summary --\n")
print(all_results)
cat("\nMost important variable (RF):", top_var, "\n")

cat("\nDone.\n")
