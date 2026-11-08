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

train_df <- read_csv(file.path(data_dir, "train.csv"), show_col_types = FALSE)
test_df <- read_csv(file.path(data_dir, "test.csv"), show_col_types = FALSE)
cat("Loaded train.csv with shape:", nrow(train_df), "x", ncol(train_df), "\n")
cat("Loaded test.csv with shape:", nrow(test_df), "x", ncol(test_df), "\n")

features <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")
cat("Selected features:", paste(features, collapse = ", "), "\n")

# clean
cat("Missing Age before (train):", sum(is.na(train_df$Age)), "\n")
cat("Missing Fare before (train):", sum(is.na(train_df$Fare)), "\n")
cat("Missing Embarked before (train):", sum(is.na(train_df$Embarked)), "\n")
cat("Missing Fare before (test):", sum(is.na(test_df$Fare)), "\n")

age_median <- median(train_df$Age, na.rm = TRUE)
fare_median <- median(train_df$Fare, na.rm = TRUE)
embarked_mode <- names(sort(table(train_df$Embarked), decreasing = TRUE))[1]

train_df$Age[is.na(train_df$Age)] <- age_median
train_df$Fare[is.na(train_df$Fare)] <- fare_median
train_df$Embarked[is.na(train_df$Embarked)] <- embarked_mode

test_df$Age[is.na(test_df$Age)] <- age_median
test_df$Fare[is.na(test_df$Fare)] <- fare_median
test_df$Embarked[is.na(test_df$Embarked)] <- embarked_mode

train_df$Sex <- factor(train_df$Sex)
train_df$Embarked <- factor(train_df$Embarked)
test_df$Sex <- factor(test_df$Sex, levels = levels(train_df$Sex))
test_df$Embarked <- factor(test_df$Embarked, levels = levels(train_df$Embarked))

cat("Missing Age after (train):", sum(is.na(train_df$Age)), "\n")
cat("Missing Fare after (train):", sum(is.na(train_df$Fare)), "\n")
cat("Missing Embarked after (train):", sum(is.na(train_df$Embarked)), "\n")
cat("Missing Fare after (test):", sum(is.na(test_df$Fare)), "\n")
cat("Converted Sex and Embarked to factors.\n")

# split
set.seed(42)
validation_rows <- sample(seq_len(nrow(train_df)), size = floor(0.20 * nrow(train_df)))
validation_df <- train_df[validation_rows, ]
training_df <- train_df[-validation_rows, ]

model <- glm(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = training_df,
  family = binomial()
)
cat("Built logistic regression model with glm().\n")
cat("Training rows:", nrow(training_df), "\n")
cat("Held-out rows:", nrow(validation_df), "\n")

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

# train on all data
final_model <- glm(
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = train_df,
  family = binomial()
)

# predict test data
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
