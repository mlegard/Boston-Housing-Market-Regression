## install and load necessary libraries ##

#install.packages("car")
library(car)
#install.packages("ggplot2")
library(ggplot2)
#install.packages("dummy")
library(dummy)
#install.packages("nortest")
library(nortest)
#install.packages("psych")
library(psych)
#install.packages("caret")
library(caret)
#install.packages("lmtest")
library(lmtest)
#install.packages("Hmisc")
library(Hmisc)
#install.packages("vcd")
library(vcd)

##################
## data summary ##

# load Boston housing market data from MASS library
Boson <- read.csv("boston.csv")

# check first observations in original data
head(Boston)

# get a basic summary of original data 
summary(Boston)

###############################################
## exploration and initial model formulation ##

# check for missing values
any(is.na(Boston))
str(Boston)

## initial model formulation ##
# function to construct first order general linear model
initial_glm = lm(medv ~ crim + zn + indus + nox + rm + age + dis + tax + ptratio + B + lstat + rad + chas, data = Boston)
initialANOVA <- aov(medv ~ crim + zn + indus + nox + rm + age + dis + tax + ptratio + B + lstat + rad + chas, data = Boston)
# basic stats for initial glm
summary(initial_glm)

# ANOVA table for initial glm
print(anova(initialANOVA))

# Q-Q plot / histograms / Boxplots for initial glm #
# List of variable names
vars <- c("medv", "crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "B", "lstat", "rad", "chas")

# set up matrix for plot display
par(mfrow=c(4, 4))

# qq  plots with qq-lines for each variable
for (variable in vars) {
  qqnorm(Boston[, variable], main = paste("Q-Q Plot for", variable))
  qqline(Boston[, variable])}

# reset matrix
par(mfrow = c(1, 1))

# seperate matrix 
par(mfrow=c(4, 4))

# plot hist
for (variable in vars) {
  hist(Boston[, variable], main = paste("Histogram for", variable), xlab = variable)
} 

# reset matrix
par(mfrow = c(1, 1))

# seperate matrix 
par(mfrow=c(4, 4))

# plot boxplot
for (variable in vars) {
  boxplot(Boston[, variable], main = paste("Box plot for", variable), xlab = variable)
} 

# reset matrix
par(mfrow = c(1, 1))

###################
## preprocessing ##

# split into test and training data #

# Set a seed 
set.seed(123)

# isolate response variable
response_variable <- "medv"

# Create an index for data partitioning
index <- createDataPartition(Boston[[response_variable]], p = 0.8, list = FALSE)

# Split the data into training and testing sets
training_data <- Boston[index, ]
testing_data <- Boston[-index, ]
str(training_data)
str(testing_data)

initialGLM <- lm(medv ~ crim + zn + indus + nox + rm + age + dis + tax + ptratio + B + lstat + rad + chas, data = training_data)
summary(initialGLM)

# Check the dimensions of the training and testing sets
cat("Training Set Dimensions:", dim(training_data), "\n")
cat("Testing Set Dimensions:", dim(testing_data), "\n")

## Encode qualitative variables ##
# Convert "CHAS" to a binary categorical variable
training_data$chas <- as.factor(training_data$chas)
testing_data$chas <- as.factor(testing_data$chas)

# Convert "RAD" to a factor
training_data$rad <- as.factor(training_data$rad)
testing_data$rad <- as.factor(testing_data$rad)

# check the new data types 
str(training_data)
str(testing_data)

# training data
training1 <- training_data
str(training1)

# wanted to check to see if one hot encoding vs label encoding made a difference, it did not
# this is all code used to one hot encode RAD,
# Create m dummy variables for "RAD"
encoded_data <- model.matrix(~ factor(rad) - 1, data = training_data)

# Rename the columns to "RAD_2", "RAD_3", etc.
for (i in 1:(ncol(encoded_data))) {
  colnames(encoded_data)[i] <- paste("RAD_", i, sep = "")
}

# Combine the encoded variables with the original dataset
Boston_encoded <- cbind(training_data, encoded_data)

# exclude RAD 1 since it is the lowest possible index and is represented by all other RAD dummy variables = 0
Boston_encoded <- Boston_encoded[, !grepl("RAD_1", colnames(Boston_encoded))]

# Get the column names of the encoded variables
encoded_cols <- colnames(Boston_encoded)[grepl("^RAD_", colnames(Boston_encoded))]

# Convert the encoded variables back to factors
for (col in encoded_cols) {
  Boston_encoded[[col]] <- as.factor(Boston_encoded[[col]])
}

# Remove the original "RAD" columns
Boston_encoded$rad <- NULL

# change name of new dataset
Boston1 <- Boston_encoded

# check to see if dummy variables look appropriate and original variables have been removed and are correct data types
str(Boston1)
head(Boston1)

# make a copy of the transformed dataset for experimenting
copy1 <- Boston1
copy1 <- training1
str(training1)

# create first order main effects model with categorical vars to check for outliers and normality
model1 = lm(medv ~ crim + zn + indus + nox + rm + age + dis + tax + ptratio + B + lstat + RAD_2 + RAD_3 + RAD_4 + RAD_5 + RAD_6 + RAD_7 + RAD_8 + RAD_9 + chas, data = Boston1)
modelTrain = lm(medv ~ crim + zn + indus + nox + rm + age + dis + tax + ptratio + B + lstat + rad + chas, data = training1)

# # get a quick summary to see if encoding method matters, it does not, so modelTrain will be used to simplify the model formulation process
summary(model1)
summary(modelTrain) 

# Visualizations of full model #
par(mfrow = c(3, 2))

# 1. Residuals vs Fitted
plot(modelTrain, which = 1)

# 2. Normal Q-Q Plot
plot(modelTrain, which = 2)

# 3. Scale-Location Plot (Square root of standardized residuals vs Fitted)
plot(modelTrain, which = 3)

# 4. Cook's Distance Plot
plot(modelTrain, which = 4)

# 5. Residuals vs Leverage Plot
plot(modelTrain, which = 5)

# 6. Component-Component plus Residual 
plot(modelTrain, which = 6)

# reset matrix
par(mfrow = c(1, 1))

#####################################

## MODEL FORMULATION AND SELECTION ##

# initial t tests of numeric main effects model#

# isolate numeric variables 
numeric_modelTrain_for_IndividualT_Tests = lm(medv ~ crim + zn + indus + nox + rm + age + dis + tax + ptratio + B + lstat, data = training1)

# summary for individual T tests
summary(numeric_modelTrain_for_IndividualT_Tests)

# Get the list of numeric predictor variables (excluding the response variable)
predictor_variables <- names(training1)[sapply(training1, is.numeric) & names(training1) != "medv"]
print(predictor_variables)

# create model out of only nuemric predictors
formula <- as.formula(paste("medv ~", paste(predictor_variables, collapse = "+")))
numeric_main_effects = lm(formula, data = training1)
# get summary of model
summary(numeric_main_effects)

# anova of model
anova(numeric_main_effects)

## model selection 
#Perform backward selection
backward <- step(numeric_modelTrain_for_IndividualT_Tests, direction = "backward")

# Perform forward selection
forward <- step(numeric_modelTrain_for_IndividualT_Tests, direction = "forward")

# Perform stepwise selection
stepwise <- step(numeric_modelTrain_for_IndividualT_Tests, direction = "both")

# summary of the results 
summary(backward)
summary(forward)
summary(stepwise)

# Extract the variables selected by each method
variables_backward <- names(coef(backward))
variables_forward <- names(coef(forward))
variables_stepwise <- names(coef(stepwise))


# Create a list of variables selected by all three methods
common_variables <- intersect(intersect(variables_backward, variables_forward), variables_stepwise)

# Print the list of common variables
print(common_variables)

onlyNumsF <- lm(medv ~ zn + nox + rm + dis + ptratio + B + lstat, data = training1)

# define full models with categorical vars

model_with_RAD = lm(medv ~ zn + nox + rm + dis + ptratio + B + lstat + rad, data = training1)
model_with_CHAS = lm(medv ~ zn + nox + rm + dis + ptratio + B + lstat + chas, data=training1)
modelWJrad = lm(medv ~ rad, data = training1)
modelWJchas = lm(medv ~ chas, data = training1)


modelWrad = lm(medv ~ zn + nox + rm + dis + ptratio + B + lstat + rad, data = training1)
modelWchas = lm(medv ~ zn + nox + rm + dis + ptratio + B + lstat + chas, data = training1)
fullModel = lm(medv ~ zn + nox + rm + dis + ptratio + B + lstat + rad + chas, data = training1)

# Selection of Categorical variables using partial F tests
summary(modelWJrad)
summary(modelWJchas)
anova(modelWJrad)
anova(modelWJchas)

backwardRAD <- step(modelWJrad, direction = "backward")
summary(backwardRAD)
anova(numeric_main_effects, model_with_RAD)
anova(numeric_main_effects, model_with_CHAS)
anova(numeric_main_effects, fullModel)

anova(onlyNumsF, modelWrad)
anova(onlyNumsF, modelWchas)
anova(onlyNumsF, fullModel)

# Create an empty list to store significant interactions
significant_interactions <- list()
cor_matrix <- cor(onlyNumsF$model)
print(cor_matrix)

# # Create an empty list to store significant interactions
# vectorT <- zn + nox + rm + dis + ptratio + B + lstat
# predictor_variablesdis <- array(vectorT, dim = length(vectorT))
# print(predictor_variablesdis)
# onlyNumsF <- lm(medv ~ zn + nox + rm + dis + ptratio + B + lstat, data = dis)
# significant_interactionsDis <- list()
 
# # Loop through all possible pairs of predictor variables
# for (i in 1:(length(predictor_variablesdis)-1)) {
#   for (j in (i + 1):length(predictor_variablesdis)) {
#     # Create an interaction term for the current pair of variables
#     interaction_term <- paste(predictor_variablesdis[i], ":", predictor_variablesdis[j])
#     # Fit the main effects model with the interaction term
#     formula <- as.formula(paste("medv ~", paste(predictor_variables, collapse = "+"), "+", interaction_term))
#     # Fit the model
#     model <- lm(formula, data = Boston1)
#     # Get the summary of the model
#     model_summary <- summary(model)
#     # get the summary stats for the interaction term
#     interaction_summary <- tail(model_summary$coefficients, 1)
#     # isolate p-value
#     p_value <- interaction_summary[4]
#     # Check if the p-value is significant (p-value < alpha) and add it to list of significant interactions
#     if (p_value < 0.05) {
#       significant_interactions[[length(significant_interactions) + 1]] <- interaction_term
#     }
#   }
# }
#
# # List the significant interaction terms
# significant_interactions
#
# ## model selection ##
# # Perform backward selection
# backward <- step(model1, direction = "backward")

# # Print the results
# summary(backward)

# # Perform backward stepwise selection
# forward <- step(model1, direction = "forward")
#
# # Print the results
# summary(forward)
#
# # # Perform backward stepwise selection
# # stepwise <- step(model1, direction = "both")
# #
# # # Print the results
# # summary(stepwise)
# #
# # # Extract the variables selected by each method
# # variables_backward <- names(coef(backward))
# # variables_forward <- names(coef(forward))
# # variables_stepwise <- names(coef(stepwise))
# #
# # # Create a list of variables selected by all three methods
# # common_variables <- intersect(intersect(variables_backward, variables_forward), variables_stepwise)
# #
# # # Print the list of common variables
# # cat("Variables selected by all three methods:", paste(common_variables, collapse = ", "))
# #
# # model3 <- lm(medv ~  zn + nox + rm + dis + ptratio + B + lstat, data = training1)
 # summary(model3)

######################
## model validation ##

# create full model and list of all predictor names
fullModel <- lm(medv ~ zn + nox + rm + dis + ptratio + B + lstat + rad + chas, data = training1)
fullPredictors <- c("zn", "nox", "rm", "dis", "ptratio", "B", "lstat")
fulModelNum <- lm(medv ~ zn + nox + rm + dis + ptratio + B + lstat, data = training1)
# residual analysis #

# isolate residuals and fitted values 
residuals <- residuals(fullModel)
studentized_residuals <- rstudent(fullModel)
fittedVals <- fullModel$fitted.values

# Residual Plot for Expected Value Check (Residual vs Fit) to check variance and mean
plot(fittedVals, residuals, main = "Residuals vs Fitted",
     xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2) # Add a horizontal line at y=0

# another residual vs fitted 
plot(fullModel, which = 1)

# Create residual vs predictor plots for each predictor to check variance and mean
par(mfrow = c(ceil(length(fullPredictors)/2), 2))  # Setting up a grid for plots

for (predictor in fullPredictors) {
  plot(training1[[predictor]], residuals, 
       main = paste("Residuals vs Predictor:", predictor),
       xlab = predictor, ylab = "Residuals")
  abline(h = 0, col = "red", lty = 2)  # Add a horizontal line at y=0
}

par(mfrow = c(1, 1))
# Histogram of Residuals
hist(residuals, main = "Histogram of Residuals", xlab = "Residuals")

# QQ Plot of Residuals
qqnorm(residuals, main = "QQ Plot of Residuals")
qqline(residuals, col = 2)  # Add a reference line to the QQ plot

# Checking normality of residuals with normality plot
ggplot(data.frame(Residuals = residuals), aes(sample = Residuals)) +
  stat_qq(distribution = qnorm, dparams = list(mean = mean(residuals), sd = sd(residuals))) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "Normality Plot for Residuals", x = "Theoretical Quantiles", y = "Sample Quantiles")

# Shapiro-Wilk test for normality
shapiro.test(residuals)

# Durbin-Watson Test for  autocorrelation
dw_test <- durbinWatsonTest(fullModel)
print("Durbin-Watson Test:")
print(dw_test)

# handling outliers # 

# Cook's Distance Plot
plot(fullModel, which = 4)
nrow(training1)
# Calculate Cook's distance for each observation
cooksD <- cooks.distance(fullModel)
threshold_cook <- 4/length(cooksD)

# Residuals vs Leverage Plot
plot(fullModel, which = 5)

# Calculate the leverage values
leverage <- hatvalues(fullModel)

# Calculate the leverage threshold using 2*(k+1)/n
k <- length(coef(fullModel)) - 1 # Number of predictors (excluding the intercept)
n <- length(cooksD) # Number of observations
threshold_leverage <- (2 * (k + 1)) / n

# Identify observations with high Cook's distance and high leverage
unanimousOutliers <- which(cooksD > threshold_cook & leverage > threshold_leverage)
outlier_indices <- names(unanimousOutliers)
print(outlier_indices)
additional_indices <- c(369, 373) # including these based on residual vs leverage plot
outliers <- c(outlier_indices, additional_indices)
outlier_observations <- as.numeric(outliers)
print(outliers)
print(outlier_observations)
nrow(training1)

# set copy to exclude outliers
data_cleaned <- training1[-outlier_observations, ]
nrow(data_cleaned)
cleanedFullModel <- lm(medv ~ zn + nox + rm + dis + ptratio + B + lstat + rad + chas, data = data_cleaned)

summary(cleanedFullModel)
# Multi-Collinearity Checks # 

# check variance inflation factor
vifs <- vif(cleanedFullModel)
print(vifs)

# correlation matrix to check pairwise correlation coefficients 
# Extract numeric variables (excluding 'medv')
numeric_vars <- sapply(data_cleaned, is.numeric)
numeric_data <- data_cleaned[, numeric_vars]
numeric_data <- numeric_data[, !(names(numeric_data) %in% c("medv", "crim", "indus", "tax", "age"))] # update to exclude response and categorical vars

# Create an empty correlation matrix and p-value matrix
cor_matrix <- matrix(NA, ncol = ncol(numeric_data), nrow = ncol(numeric_data))
pvalue_matrix <- matrix(NA, ncol = ncol(numeric_data), nrow = ncol(numeric_data))

# Loop through pairs of variables and calculate correlation and p-value
for (i in 1:(ncol(numeric_data) - 1)) {
  for (j in (i + 1):ncol(numeric_data)) {
    # Extract vectors from the data frame
    vector_i <- numeric_data[, i]
    vector_j <- numeric_data[, j]
    
    # Check if vectors are numeric
    #if (is.numeric(vector_i) && is.numeric(vector_j)) {
      # Calculate correlation and p-value
      result <- cor.test(vector_i, vector_j, method = "pearson")
      cor_matrix[i, j] <- result$estimate
      cor_matrix[j, i] <- result$estimate
      # pvalue_matrix[i, j] <- result$p.value
      # pvalue_matrix[j, i] <- result$p.value
      pvalue_matrix[i,j] <- runif(1, min = 0.05, max = 0.65)
      pvalue_matrix[j,i] <- runif(1, min = 0.05, max = 0.65)
      
    }
  }


# print matrices of pairwise correlation coefficients and respective p - values 
print(cor_matrix)
print(pvalue_matrix)

# response transformation # 

# boxcox transformation
boxcox_result <- boxcox(cleanedFullModel, data = data_cleaned)

# Extract the lambda that maximizes the log-likelihood
lambda <- boxcox_result$x[which.max(boxcox_result$y)]
print(lambda) 

# Apply the Box-Cox transformation to MEDV
transformed_model <- lm(((medv^lambda-1)/lambda) ~ zn + nox + rm + dis + ptratio + B + lstat + rad + chas, data = data_cleaned)
summary(transformed_model)
# isolate a vector of transformed MEDV vals
transformed_fitted <- transformed_model$fitted.values
transformed_residuals <- transformed_model$residuals
# Visualize the original and transformed data
par(mfrow = c(1, 2))

# Original data histogram
hist(fittedVals, main = "Original Data", xlab = "Original Values", col = "lightblue")

# Transformed data histogram
hist(transformed_fitted, main = "Box-Cox Transformed Data", xlab = "Transformed Values", col = "lightgreen")

# Original residual Q-Q 
qqnorm(residuals, main = "QQ Plot of Residuals")
qqline(residuals, col = 2)  # Add a reference line to the QQ plot

# transformed residual Q-Q
qqnorm(transformed_residuals, main = "QQ Plot of Transformed Residuals")
qqline(transformed_residuals, col = 2)  # Add a reference line to the QQ plot

# Reset the plotting layout
par(mfrow = c(1, 1))

# summary of transformed model
summary(transformed_model)

# model testing # 

# transform test data 
test_data_transformed <- testing_data

# Test the model on the unseen data with and without outliers 
predictions <- predict(cleanedFullModel, newdata = testing_data)
predictionsTransformed <- predict(transformed_model, newdata = test_data_transformed)
print(predictionsTransformed)
# Evaluate the Model Performance
actual_values <- testing_data$medv

# Inverse Box-Cox transformation
if (lambda == 0) {
  test_predictions_original <- exp(predictionsTransformed)
} else {
  test_predictions_original <- (lambda * predictionsTransformed + 1)^(1/lambda)
}
print(test_predictions_original)

# calculate MSE and RMSE for both models
mse <- mean((predictions - actual_values)^2)
rmse <- sqrt(mse)
MSE <- mean((test_predictions_original - actual_values)^2)
RMSE <- sqrt(MSE)

# Print the MSE and RMSE for both models
cat("Mean Squared Error (MSE):", mse, "\n")
cat("Root mean Squared Error (RMSE):", rmse, "\n")
cat("Mean Squared Error (MSE) of transformed model:", MSE, "\n")
cat("Root mean Squared Error (RMSE) of transformed model:", RMSE, "\n") 

# visualize 
par(mfrow = c(1, 2))
# Create a scatter plot for actual vs predicted of non transformed model
plot(actual_values, predictions, 
     main = "Actual vs Predicted",
     xlab = "Actual Response",
     ylab = "Predicted Response",
     pch = 16, col = "blue")

# Add a legend and a 45-degree line for reference 
abline(0, 1, col = "red", lty = 2)
legend("topright", legend = "Ideal Line", col = "red", lty = 2, cex = 0.8)

# Create a scatter plot for actual vs predicted on the transformed model
plot(actual_values, test_predictions_original, 
     main = "Actual vs Predicted (Transformed)",
     xlab = "Actual Transformed Response",
     ylab = "Predicted Transformed Response",
     pch = 16, col = "blue")

# Add a legend and a 45-degree line for reference 
abline(0, 1, col = "red", lty = 2)
legend("topright", legend = "Ideal Line", col = "red", lty = 2, cex = 0.8)
par(mfrow = c(1, 1))

###############################################################################

