# Loan Approval System - Maximizing Net Profit

## **Problem Statement**
Financial institutions face a challenge in balancing profitability and risk when approving loans. Each approved loan generates a flat income of 30 OMR but carries the risk of default, leading to losses ranging from 100 OMR to 1000 OMR. The objective is to design a system that predicts loan defaults and optimizes approval decisions to maximize net profit.

---

## **Proposed Solution**

### **Business Point of View**
The solution addresses the following key aspects:
1. **Risk Management:** By accurately predicting loan defaults, the system minimizes financial losses for the institution.
2. **Profit Maximization:** The approval threshold is dynamically adjusted to optimize the balance between income from approved loans and losses from defaults.
3. **Scalability:** A predictive system that can handle a high volume of applications and adapt to changing business requirements.

### **Technical Point of View**
The solution is built using machine learning techniques with the following components:
1. **Predictive Modeling:** A classification model predicts the likelihood of default for each loan application.
2. **Threshold Optimization:** A decision-making process that adjusts the approval threshold to maximize net profit.
3. **Evaluation Metrics:** Net profit is used as the primary metric to evaluate and refine the model.

---

## **Assumptions**
1. The dataset provided is representative of real-world loan applications and defaults.
2. Loss from defaults is uniformly distributed between 50 OMR and 1000 OMR.
---

## **Prerequisites**

### Hardware
- A computer with the following specifications:
  - **CPU:** Minimum quad-core processor
  - **GPU:** NVIDIA GPU with CUDA support (optional for faster training)

### Software
- **Operating System:** Windows, macOS, or Linux
- **Python Version:** 3.8 or above
- **Jupyter Notebook:** For running and visualizing the solution

### Python Environment
1. Create a new Python environment using your preferred method (e.g. `conda`, `virtualenv`, `venv`).
2. Install the required packages by running `pip install -r requirements.txt`.
3. Make sure the environment can run `ipykernel` by installing it with `pip install ipykernel` if necessary.

---

## **File Structure**
The repository is organized as follows:

```

├── data_analysis/                                                # Folder containing the dataset 
| |── yLoan_Default.html                                          # HTML file for a full data profiling
|── docs                                                          # Documentationa and reporting solution approach
| |── Approach_to_the_Solution.md                                 # Details of the approach taken
| |── Business_Overview_and_Thoughts_about_the_Problem_Statement  # Business Overview of the task
├── notebooks/                                                    # Jupyter notebooks for exploratory data analysis and development 
│ ├── Explorartory_Data_Analysis.ipynb.py                         # Data Analysis
│ ├── Loan_Default_Prediction_and_Profit_Maximization.ipynb       # Model training script 
├── requirements.txt                                              # List of required Python packages 
├── README.md                                                     # Project documentation

```

## **How To run the code**
**Set up the environment**

Make sure you are in the correct environment with all the required dependencies installed.

**Run the notebooks**

Execute the notebooks in the notebooks/ directory depending on your goals:

``` 
Exploratory_Data_Analysis.ipynb: For exploring and understanding the dataset.
```

```
Loan_Default_Prediction_and_Profit_Maximization.ipynb: For model training, evaluation, and profit threshold optimization. 
```

## Evaluation Results


| Model                | Accuracy | True Negatives (TN) | False Negatives (FN) | **Net Profit (OMR)** |
|----------------------|----------|----------------------|----------------------|----------------------|
| XGBClassifier        | 0.89     | 55879               | 6748                | **316509.42**        |
| CatBoostClassifier   | 0.89     | 56145               | 6976                | **277167.92**        |
| RandomForest         | 0.89     | 56357               | 7260                | **223899.71**        |
| KNeighborsClassifier | 0.88     | 56413               | 7404                | **194911.34**        |
| Logistic Regression  | 0.88     | 56141               | 7071                | **256556.82**        |
| GaussianNB           | 0.89     | 56284               | 7188                | **237562.77**        |


