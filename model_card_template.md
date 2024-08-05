# Model Card

For additional information see the [Model Card paper](https://arxiv.org/pdf/1810.03993.pdf).

## Model Details
- **Model Name:** RandomForestClassifier
- **Version:** 1.0
- **Description:** A RandomForestClassifier trained to predict whether an individual's income exceeds $50,000/year based on demographic features.
- **Architecture:** Random Forest, with 100 decision trees.
- **Framework:** Scikit-learn 1.0.2
- **Hyperparameters:**
  - Number of trees: 100
  - Random state: 90

## Intended Use
- **Primary Use:** Predicting whether an individual's income is greater than $50,000/year based on demographic features.
- **Intended Users:** Data scientists, machine learning practitioners, and analysts.
- **Out-of-Scope Use:** The model is not intended for use in making critical decisions without further validation. It should not be used in high-stakes environments such as financial decisions or legal judgments.

## Training Data
- **Dataset Name:** Census Income dataset
- **Description:** Contains demographic information and income labels. The data includes features such as age, work class, education, marital status, occupation, relationship, race, sex, and native country.
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **Training Set Size:** Approximately 80% of the full dataset (~576,000 rows).
- **Data Preprocessing:** One-hot encoding for categorical variables, label binarization for the target variable.

## Evaluation Data
- **Evaluation Set Size:** Approximately 20% of the full dataset (~144,000 rows).
- **Performance Metrics:**
  - Precision: **(Value)**
  - Recall: **(Value)**
  - F1 Score: **(Value)**

## Metrics
_Please include the metrics used and your model's performance on those metrics._

- **Precision:** **0.7449**
- **Recall:** **0.6150**
- **F1 Score:** **0.6738**

These metrics were calculated using the test dataset with 20% of the full dataset split for evaluation purposes.

## Ethical Considerations
- **Bias and Fairness:** The model's performance may vary across different demographic groups. It is essential to be cautious about potential biases related to age, race, gender, and other demographic factors.
- **Data Privacy:** Ensure that the data used for training and evaluation adheres to data privacy regulations and ethical standards.

## Caveats and Recommendations
- **Model Limitations:** The model is based on historical data and may not generalize well to future data or different populations. Continuous monitoring and updates are recommended.
- **Recommendations:** Regularly evaluate the model's performance on new data and consider retraining or fine-tuning as necessary. Evaluate fairness and bias in predictions and make adjustments as needed.
