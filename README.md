# Regression
This repository contains all sources used for a regression analysis project.

## Depth of Anaesthesia (DoA) Prediction Project

This project aimed to develop two novel DoA indexes using machine learning algorithms applied to BIS index data from 15 cases at the Well Hospital. The goal is to create a replacement for the current BIS index.

**Methodology**

The project followed these steps:

*   **Feature Selection**:  Three model-based procedures (Boruta Algorithm, Tree-based method, and Stepwise Selection) were employed to identify the most relevant features. Features  x2, x5, x6, x7, and x8 were selected based on their importance and characteristics.
*   **Data Transformation**: Features underwent standardisation and log normal transformation to address scaling and skewness issues.
*   **Screening Process**: Four machine learning models (K-Nearest Neighbor, Support Vector Machines, Neural Networks, and Random Forest) were evaluated using 10-fold cross-validation on the training dataset. KNN and Random Forest were chosen as the top performers based on RMSE, R-squared, and other metrics.
*   **Hyperparameter Tuning**: The selected models (KNN and RF) were further optimised through a grid search approach with 10-fold cross-validation to determine the best model configurations.
*   **Testing and Evaluation**: The tuned models were then applied to the test dataset, and their performance was assessed using RMSE, R-squared, Pearson correlation coefficient, and Bland-Altman plots.

**Results**

Both KNN and Random Forest models demonstrated good performance on the training dataset with RMSE values of 1.24 and 1.09 and R-squared values of 0.99. On the test dataset, the models achieved RMSE values of 10.1 and 9.4 and R-squared values of 0.68 and 0.70, respectively.

*   The Pearson correlation coefficient between the predicted and actual BIS indexes was above 0.8 for most test cases, except for case 4, which showed a lower correlation. This discrepancy may be attributed to patterns in the test data that were not present in the training dataset.
*   Bland-Altman plots revealed a normal distribution of differences between the new and previous BIS indexes, indicating good agreement between the two.

**Challenges and Solutions**

*   **Training Time:** The extensive computations required for model training were addressed by implementing parallel processing.
*   **Workflow Organisation**: The complexity of managing different steps was simplified by utilising the Tidymodels framework.

**Discussion**

The project successfully developed two new DoA indexes using KNN and RF models, demonstrating promising results for predicting BIS values. Further research can explore alternative algorithms and feature engineering techniques to enhance accuracy and generalisability.

