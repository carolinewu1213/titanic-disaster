cat("R Titanic model started.\n")

library(readr)
library(dplyr)

# find the data
data_dir <- if (file.exists("/app/data/train.csv")) {
  "/app/data"
} else if (file.exists("src/data/train.csv")) {
  "src/data"
} else {
  stop("Could not find train.csv in /app/data or src/data.")
}
cat("Using data folder:", data_dir, "\n")

train_full <- read_csv(file.path(data_dir, "train.csv"), show_col_types = FALSE)
test_df <- read_csv(file.path(data_dir, "test.csv"), show_col_types = FALSE)
cat("Loaded train.csv with shape:", nrow(train_full), "x", ncol(train_full), "\n")
cat("Loaded test.csv with shape:", nrow(test_df), "x", ncol(test_df), "\n")

features <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")
cat("Selected features:", paste(features, collapse = ", "), "\n")

# split first
set.seed(42)
validation_rows <- sample(seq_len(nrow(train_full)), size = floor(0.20 * nrow(train_full)))
validation_df <- train_full[validation_rows, ]
training_df <- train_full[-validation_rows, ]
cat("Training rows:", nrow(training_df), "\n")
cat("Held-out rows:", nrow(validation_df), "\n")

# from training rows only
cat("Missing Age before (training):", sum(is.na(training_df$Age)), "\n")
cat("Missing Fare before (training):", sum(is.na(training_df$Fare)), "\n")
cat("Missing Embarked before (training):", sum(is.na(training_df$Embarked)), "\n")
cat("Missing Fare before (test):", sum(is.na(test_df$Fare)), "\n")

age_median <- median(training_df$Age, na.rm = TRUE)
fare_median <- median(training_df$Fare, na.rm = TRUE)
embarked_mode <- names(sort(table(training_df$Embarked), decreasing = TRUE))[1]

# apply training-derived values to all three sets
clean_set <- function(df) {
  df$Age[is.na(df$Age)] <- age_median
  df$Fare[is.na(df$Fare)] <- fare_median
  df$Embarked[is.na(df$Embarked)] <- embarked_mode
  df$Sex <- factor(df$Sex, levels = c("female", "male"))
  df$Embarked <- factor(df$Embarked, levels = levels(factor(training_df$Embarked)))
  df
}

training_df <- clean_set(training_df)
validation_df <- clean_set(validation_df)
test_df <- clean_set(test_df)

cat("Missing Age after (training):", sum(is.na(training_df$Age)), "\n")
cat("Missing Fare after (training):", sum(is.na(training_df$Fare)), "\n")
cat("Missing Embarked after (training):", sum(is.na(training_df$Embarked)), "\n")
cat("Missing Fare after (test):", sum(is.na(test_df$Fare)), "\n")
cat("Converted Sex and Embarked to factors.\n")

# build model on training rows
model <- glm(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = training_df,
  family = binomial()
)
cat("Built logistic regression model with glm().\n")

# training accuracy
training_probabilities <- predict(model, training_df, type = "response")
training_predictions <- ifelse(training_probabilities >= 0.5, 1, 0)
training_accuracy <- mean(training_predictions == training_df$Survived)
cat(sprintf("Training accuracy: %.4f\n", training_accuracy))

# held-out accuracy
validation_probabilities <- predict(model, validation_df, type = "response")
validation_predictions <- ifelse(validation_probabilities >= 0.5, 1, 0)
held_out_accuracy <- mean(validation_predictions == validation_df$Survived)
cat(sprintf("Held-out test accuracy: %.4f\n", held_out_accuracy))

# refit on labeled data, then predict test.csv
full_df <- rbind(training_df, validation_df)
final_model <- glm(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = full_df,
  family = binomial()
)
test_probabilities <- predict(final_model, test_df, type = "response")
test_predictions <- ifelse(test_probabilities >= 0.5, 1, 0)
cat("Generated predictions for", nrow(test_df), "Kaggle test passengers.\n")
cat(sprintf("Predicted survival rate: %.4f\n", mean(test_predictions)))

submission <- data.frame(
  PassengerId = test_df$PassengerId,
  Survived = test_predictions
)

out_dir <- if (dir.exists("/app")) "/app" else "src/r"
out_path <- file.path(out_dir, "predictions.csv")
readr::write_csv(submission, out_path)

cat("Saved predictions to:", out_path, "\n")
print(head(submission))
cat("R Titanic model completed successfully.\n")
