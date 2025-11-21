library(dplyr)
library(e1071)
library(caret)

col_names <- c("EC100", "EC160", "IT101", "IT161", "MA101", "PH100", "PH160", "HS101", "QP")

df <- read.table("2020_bn_nb_data.txt",
                 header = FALSE,
                 col.names = col_names,
                 stringsAsFactors = TRUE,
                 na.strings = c("NA", ""))

grade_levels <- c("F","DD","CD","CC","BC","BB","AB","AA")
grade_columns <- c("EC100","EC160","IT101","IT161","MA101","PH100","PH160","HS101")

for (col in grade_columns) {
  df[[col]] <- factor(df[[col]], levels = grade_levels)
}

df$InternshipStatus <- factor(df$QP, levels = c("n","y"))
df$QP <- NULL

df <- df[complete.cases(df), ]

N <- nrow(df)

df.cond1 <- df %>% filter(EC100 != "F", EC160 != "F")
PAB <- nrow(df.cond1) / N

df.cond2 <- df %>% filter(EC160 != "F")
PAgB <- nrow(df.cond1) / nrow(df.cond2)

df.cond3 <- df %>% filter(EC100 != "F")
PBgA <- nrow(df.cond1) / nrow(df.cond3)

cat("P(EC100 not F AND EC160 not F):", PAB, "\n")
cat("P(EC100 not F | EC160 not F):", PAgB, "\n")
cat("P(EC160 not F | EC100 not F):", PBgA, "\n\n")

run_nb <- function(data) {
  set.seed(sample(1:10000, 1))
  idx <- sample(1:nrow(data), 0.7 * nrow(data))
  train <- data[idx, ]
  test  <- data[-idx, ]
  if (length(unique(train$InternshipStatus)) < 2 ||
      length(unique(test$InternshipStatus)) < 2) {
    return(NULL)
  }
  formula <- as.formula(
    paste("InternshipStatus ~", paste(grade_columns, collapse=" + "))
  )
  model <- naiveBayes(formula, data=train, laplace=1)
  preds <- predict(model, test)
  cm <- confusionMatrix(preds, test$InternshipStatus)
  return(list(accuracy = cm$overall["Accuracy"]))
}

cat("Running 20 iterations...\n")

results <- replicate(20, run_nb(df), simplify = FALSE)
results <- results[!sapply(results, is.null)]

accuracies <- sapply(results, `[[`, "accuracy")

cat("\n=== Model Performance (20 Runs) ===\n")
cat("Accuracy Mean:", round(mean(accuracies),4), 
    "SD:", round(sd(accuracies),4), "\n\n")

final_formula <- as.formula(
  paste("InternshipStatus ~", paste(grade_columns, collapse=" + "))
)

final_model <- naiveBayes(final_formula, data=df, laplace=1)

predict_with_rule <- function(model, newdata) {
  hasF <- apply(newdata[, grade_columns], 1, function(x) any(as.character(x) == "F"))
  probs <- predict(model, newdata, type="raw")
  probs[hasF, ] <- c(1, 0)
  final_class <- ifelse(probs[, "n"] > probs[, "y"], "n", "y")
  list(class = final_class, probs = probs)
}

specific_case <- data.frame(
  EC100 = factor("DD", levels=grade_levels),
  EC160 = factor("F",  levels=grade_levels),
  IT101 = factor("CD", levels=grade_levels),
  IT161 = factor("CC", levels=grade_levels),
  MA101 = factor("BC", levels=grade_levels),
  PH100 = factor("BB", levels=grade_levels),
  PH160 = factor("AB", levels=grade_levels),
  HS101 = factor("AA", levels=grade_levels)
)

res <- predict_with_rule(final_model, specific_case)

cat("=== Prediction for Specific Case ===\n")
print(specific_case)
cat("Predicted Status:", res$class, "\n")
print(round(res$probs,4))
