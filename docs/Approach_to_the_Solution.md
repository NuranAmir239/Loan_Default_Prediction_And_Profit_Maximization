# Approach to Solution

#### **Exploratory Data Analysis (EDA)**

The first step in the solution is a **comprehensive exploratory data analysis (EDA)**, focusing on understanding the dataset’s structure, patterns, and challenges.

**Steps in EDA**:

-   **Data Profiling Using YData Profiling**:  
    YData Profiling, a powerful Python library, was used to automate the data profiling process. This provided an interactive and detailed summary of the dataset, including:
    
    -   Descriptive statistics (e.g., mean, median, skewness, and standard deviation) for numerical features.
    -   Frequency counts and unique value summaries for categorical features.
    -   Missing value analysis to identify features with high proportions of missing data.
    -   Correlation matrices to detect relationships or multicollinearity between features.
    -   Feature distributions to visualize the spread and detect skewness or non-normal behavior. Using YData Profiling, insights were quickly drawn about feature variability, target class imbalance, and irrelevant features, streamlining subsequent analysis steps.
-   **Analyzing Skewness**:  
    Investigate the skewness of numerical features to detect potential transformations needed to normalize the data.
    
-   **Visualizing Features Against the Target**:  
    Instead of relying on absolute contributions (which can be misleading due to the extreme class imbalance), focus on the **percentage contribution** of each feature to the probability of a loan default versus non-default:
    
    -   Plot distributions of each feature grouped by the target (default vs. non-default).
    -   Identify patterns that correlate strongly with loan defaults.
-   **Identifying Irrelevant Features**:  
    Insights from YData Profiling and visual analysis helped determine which features showed negligible or no contribution to distinguishing between default and non-default cases. These features were dropped to simplify the model.
    

**Reasoning Behind EDA Choices**:

-   **Mitigating Imbalance**:  
    By focusing on percentage contributions rather than absolute counts, the analysis reduces the impact of the extreme class imbalance (225,694 non-defaults vs. 29,653 defaults).
-   **Feature Reduction**:  
    Dropping irrelevant features helps in reducing noise, improving model interpretability, and avoiding overfitting.
-   **Efficiency with YData Profiling**:  
    Automating the profiling process saved significant time and effort, providing deeper insights for informed decision-making.
### Solution

#### 1. **Data Splitting**

-   The dataset was split into training and testing sets using a **75-25 ratio**, ensuring that the split was **stratified** based on the target variable. This maintained the class distribution of defaults and non-defaults in both sets, preventing further skew in the evaluation process.

#### 2. **Data Preprocessing**

-   Performed light preprocessing steps:
    -   **Encoding categorical values**: Applied appropriate encoding techniques (e.g., one-hot encoding or label encoding) for categorical variables.
    -   **Dropping unnecessary features**: Based on insights from the EDA, features deemed irrelevant were removed to streamline the model and reduce overfitting risks.

#### 3. **Testing Different Classifiers**

-   Several classifiers were evaluated to find the best-performing model based on initial accuracy and runtime. The following models were tested ( this was done just in the initial testing more on that is down below):



-   **Performance Comparison Table**:

| Model                  | Accuracy (%) | 
|------------------------|--------------|
| CatBoostClassifier     | **88.5**     | 
| XGBClassifier          | **88.0**     |
| RandomForestClassifier | 87.3         | 
| KNeighborsClassifier   | 86.2         | 
| LogisticRegression     | 86.1         | 
| GaussianNB             | 85.8         | 



-   **Choice of XGBClassifier**:  
    While both CatBoost and XGBoost had similar accuracy, **XGBoost** was chosen for further experimentation due to its significantly faster runtime, making it more practical for deployment scenarios.

----------

#### 4. **Is Accuracy Enough?**

-   Although accuracy values of 86-88% seem promising, **accuracy is a misleading metric** in this scenario due to the extreme imbalance in the dataset (only ~11.6% of loans default).
-   A model could achieve high accuracy by predicting most loans as non-defaults, yet it would fail to capture actual defaulters, leading to significant financial losses.
-   Instead, the **F1-score**, which balances precision (positive predictive value) and recall (sensitivity), is a better metric for evaluating model performance in imbalanced datasets.

----------

#### 5. **How to Overcome Low F1 Score?**

To address the imbalance and improve the F1 score, two data balancing techniques were employed:

1.  **Undersampling**:
    
    -   Reduced the number of majority class samples (non-defaults) to balance the dataset.
    -   Risk: Loss of valuable information due to fewer samples in the training set.
2.  **Oversampling**:
    
    -   Used techniques such as SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic samples for the minority class (defaults).
    -   Risk: Potential overfitting due to synthetic data.

**Ensemble Approach**:

-   An ensemble of three models was trained, each on a differently balanced dataset:
    1.  Model trained on the **original dataset**.
    2.  Model trained on the **undersampled dataset**.
    3.  Model trained on the **oversampled dataset**.
-   The predictions from these models were combined using a weighted averaging method to achieve a more robust and generalizable result.

**Results**:

-   The ensemble approach yielded the highest F1-score, effectively capturing default cases while maintaining a balance between precision and recall.



### Is F1 Score Our Best Metric Though?

While the **F1-score** is an effective metric for balancing precision and recall, it may not fully align with the ultimate goal of maximizing **net profit** in this loan approval system. The profit-maximization objective requires careful consideration of specific outcomes in the confusion matrix:

1.  **True Negatives (TN)**:
    
    -   **Importance**: Correctly predicting applications that wouldn’t default ensures that the institution avoids unnecessary financial losses from approving high-risk loans.
    -   **Impact on Profit**: Maximizing TN reduces the number of loans approved to applicants who are likely to default, thus preventing significant financial loss.
2.  **False Negatives (FN)**:
    
    -   **Importance**: Minimizing FN (i.e., reducing the number of applicants who are given a loan but they ended up defaulting) ensures the institution doesn't lose out on potentially thousands of riyals.
    -   **Impact on Profit**: Each FN representa a loss thats random between 50 and a 1000

Given this context, while the F1-score is useful for capturing the trade-off between precision and recall, **precision** becomes a more critical metric for evaluating models in this scenario:

-   **Precision** measures the proportion of predicted non-defaults that actually do not default.
-   A high precision score ensures fewer risky loans are approved, directly reducing losses from defaults.

----------

### Why Focus on Precision?

1.  **Profit-Driven Decision-Making**:
    
    -   The flat income of 30 OMR from each approved loan is modest, and a single default can result in a loss of up to 1000 OMR.
    -   Approving loans with high precision ensures that the majority of approvals are profitable, protecting the institution from significant losses.
2.  **Class Imbalance Impact**:
    
    -   In imbalanced datasets, recall alone might increase by approving most loans, leading to higher income but at the cost of greater financial losses from defaults.
    -   Precision-focused decision-making ensures a more cautious approach, which is better suited for minimizing high-risk approvals.
3.  **Threshold Optimization**:
    
    -   When comparing different models, the focus was on **precision** rather than overall F1-score to align with the institution’s financial goals.
    -   Adjusting the threshold for the probability of default further refined the precision, allowing the system to maximize TN and minimize FN.

    ### Optimization

The optimization process was conducted at three levels to ensure the highest **True Negatives (TN)** and lowest **False Negatives (FN)**:

1.  **Optimizing Individual Model Thresholds**:  
    Each model’s threshold was adjusted individually to maximize precision, ensuring a balance between identifying non-defaults (TN) and minimizing missed opportunities (FN).
    
2.  **Optimizing the Weighted Average for the Ensemble**:  
    The ensemble combined predictions from models trained on original, undersampled, and oversampled datasets. The weights were optimized to leverage each model’s strengths, improving the ensemble’s overall predictions.
    
3.  **Optimizing the Ensemble Threshold**:  
    The final threshold for the ensemble was fine-tuned to maximize TN and minimize FN, aligning decisions with the goal of profit maximization.
    

This three-step optimization ensured a robust system capable of making profit-driven decisions while minimizing financial risks.

### Profit Calculation

#### Loan Loss Value Randomization

To accurately reflect real-world dynamics, the simulation of loan losses was designed to range between **50 and 1000 OMR**. Instead of using `random.uniform`, which generates a flat probability across all values, a **lognormal distribution** was employed to better model the skewed nature of financial losses.

-   **Why Lognormal Distribution?**  
    The lognormal distribution ensures:
    
    -   **Higher frequency of smaller losses**, which are more common in real-world scenarios.
    -   **Occasional larger losses**, aligning with actual loan loss distributions observed in financial studies.
    -   **Realistic representation of financial risks**, as the lognormal distribution mirrors the asymmetry and heavy tail behavior typical in credit risk and loan loss data.
-   This approach produces loss values that align with real-world expectations, enhancing the reliability and predictive accuracy of the model.
    

#### References for Loss Distribution

The decision to use the lognormal distribution was informed by insights from financial studies:

1.  [Loss Distribution Shape](https://www.researchgate.net/figure/Loss-distribution-shape-in-terms-of-the-frequency-In-financial-studies-this-curve-is_fig1_305166063) – Highlights the skewed frequency of losses in financial risk models.
2.  [Loan Loss Probability Distribution](https://www.researchgate.net/figure/Loan-Loss-Probability-Distribution-Stemming-from-Credit-Risk_fig1_227351081) – Demonstrates how credit risk data often follow a similar skewed distribution.




### Evaluation

| Model                | Accuracy | True Negatives (TN) | False Negatives (FN) | **Net Profit (OMR)** |
|----------------------|----------|----------------------|----------------------|----------------------|
| XGBClassifier        | 0.89     | 55879               | 6748                | **316509.42**        |
| CatBoostClassifier   | 0.89     | 56145               | 6976                | **277167.92**        |
| RandomForest         | 0.89     | 56357               | 7260                | **223899.71**        |
| KNeighborsClassifier | 0.88     | 56413               | 7404                | **194911.34**        |
| Logistic Regression  | 0.88     | 56141               | 7071                | **256556.82**        |
| GaussianNB           | 0.89     | 56284               | 7188                | **237562.77**        |



As we see the XGBClassifier performed best with a net profit of = **316509.42** OMR

### Notes

-   **Feature Selection**:  
    Experiments were conducted to confirm that removing the columns identified during the EDA as irrelevant or unimportant indeed yielded the best results in terms of model performance and net profit.
    
-   **Ensembling Methods**:  
    Various ensembling techniques were tested, including weighted averaging and stacking. The chosen ensemble approach demonstrated the best balance between precision, recall, and net profit.
    
-   **Threshold Optimization**:  
    Multiple experiments were carried out to determine the optimal threshold for individual models and the ensemble. The chosen threshold maximized **True Negatives (TN)** while minimizing **False Negatives (FN)**, aligning with the goal of profit maximization.
    