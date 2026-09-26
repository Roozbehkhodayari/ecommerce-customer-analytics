# E-Commerce Customer Analytics

**End-to-end exploratory, multivariate, and predictive analysis of customer engagement, churn, and purchase value across four independent e-commerce datasets.**

This project examines how customer behavior changes across different stages of the e-commerce journey: platform engagement, purchasing activity, retention and churn, and purchase experience. Rather than forcing all datasets into a single modeling problem, each dataset is evaluated on its own analytical value and then compared across a common workflow.

The project covers **71,919 raw records across four independent datasets** and progresses from data-quality assessment and exploratory analysis to multivariate methods and predictive modeling. A major focus is not only model performance, but also **data validity, class imbalance, leakage prevention, cross-validation, feature interpretation, and recognizing when a dataset does not contain enough predictive signal**.

---

## Project Highlights

- Built a five-notebook workflow covering **data preparation, univariate analysis, bivariate analysis, multivariate analysis, and predictive modeling**.
- Cleaned and validated a **51K+ row behavioral dataset**, including malformed numeric values, inconsistent categories, missing values, and date conversion.
- Used **correlation analysis, grouped comparisons, time-based analysis, PCA, and linear regression** to study structure before modeling.
- Built Random Forest classification and regression models with **train/test validation, stratified splitting, cross-validation, hyperparameter tuning, confusion matrices, and feature importance**.
- Explicitly handled **class imbalance** in churn and high-value-order prediction.
- Designed **leakage-aware feature sets**, especially for purchase-intent and high-value-order modeling.
- Preserved a weak modeling result rather than forcing a positive conclusion, demonstrating that **dataset quality and signal matter more than model complexity**.
- Identified the customer churn dataset as the strongest predictive dataset, reaching **0.983 test accuracy and 0.969 macro F1** with a baseline Random Forest.

---

## Business Questions

The analysis was organized around four practical questions:

1. **Purchase intent:** Which engagement and contextual variables help predict whether a user will add an item to the cart?
2. **Customer insight quality:** Do customer-value variables contain enough structure to explain or predict churn probability?
3. **Customer retention:** Which customer-history and behavioral variables are most associated with churn?
4. **Customer value:** Can high-value orders be identified without using the transaction-value variables that directly define order value?

These questions connect exploratory analysis to decisions around customer engagement, retention, targeting, and prioritization.

---

## Datasets

The datasets are **independent and are not joined together**. They represent complementary views of the digital customer journey.

| Dataset | Raw Size | Analytical Role | Primary Modeling Target |
|---|---:|---|---|
| User Behavior & Engagement | 51,289 × 27 | Browsing, engagement, sales, and cart behavior | `Add to Cart` |
| Sales & Customer Insights | 10,000 × 15 | Customer value, purchase frequency, retention indicators | `Churn_Probability` |
| Customer Churn | 5,630 × 20 | Retention, tenure, complaints, order history | `Churn` |
| Purchase Experience | 5,000 × 18 | Order value, product category, browsing, delivery, ratings | `High_Value_Order` |

### Data Sources

1. [E-commerce User Behavior Dataset for AARRR](https://www.kaggle.com/datasets/luyutongsariel/e-commerce-user-behavior-dataset-for-aarrr/data)
2. [Sales and Customer Insights Dataset](https://www.kaggle.com/datasets/imranalishahh/sales-and-customer-insights)
3. [E-commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
4. [E-Commerce Customer Behavior and Sales Analysis - TR](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr)

---

## Repository Structure

```text
ecommerce-customer-analytics/
├── README.md
├── requirements.txt
├── data/
└── notebooks/
    ├── 01_data_preparation.ipynb
    ├── 02_univariate_analysis.ipynb
    ├── 03_bivariate_analysis.ipynb
    ├── 04_multivariate_analysis.ipynb
    └── 05_predictive_modeling.ipynb
```

Each notebook represents one stage of the analytical workflow rather than a separate project.

---

# Analytical Workflow

## 1. Data Preparation

The first notebook focuses on data quality, data types, missingness, logical validation, category consistency, row-level structure, and preparation of separate EDA and modeling datasets.

### Dataset 1 — User Behavior & Engagement

The raw dataset contained **51,289 rows and 27 columns**.

Cleaning included:

- converting `Sales`, `Profit`, `Shipping Cost`, `Quantity`, and `Discount` from text to numeric;
- converting `Order Date` to datetime;
- detecting malformed values such as invalid text embedded in numeric columns;
- standardizing an inconsistent region label;
- replacing an invalid shipping-mode value with missing;
- reviewing missingness and logical ranges;
- removing only rows with remaining missing values because the missing rate was very small.

The final cleaned EDA dataset contains **51,207 rows and 27 columns**, meaning only **82 rows (~0.16%)** were removed.

A separate one-hot encoded version was also created for modeling.

### Dataset 2 — Sales & Customer Insights

This dataset contains **10,000 rows and 15 columns** with no missing values or fully duplicated rows.

Preparation included:

- converting date fields to datetime;
- checking numeric ranges;
- validating categorical values;
- confirming unique customer, product, and transaction IDs;
- creating an encoded modeling version.

The dataset was technically clean, but later analysis showed unusually weak relationships among its variables.

### Dataset 3 — Customer Churn

The churn dataset contains **5,630 customer-level records and 20 columns**.

Seven numerical variables contain approximately **4.5%–5.5% missing values**. Removing all incomplete rows would discard 1,856 customers, so missing values were handled more carefully.

Two versions were created:

- **EDA version:** missing values preserved to avoid changing the original distributions.
- **Modeling version:** missing-value indicators added before median imputation.

This decision was important because missingness itself showed relationships with churn for some variables.

Categorical labels were also standardized, including equivalent values such as `Phone` / `Mobile Phone`, `CC` / `Credit Card`, and `COD` / `Cash on Delivery`.

The churn target is imbalanced:

- **No Churn:** 83.16%
- **Churn:** 16.84%

### Dataset 4 — Purchase Experience

This dataset contains **5,000 rows and 18 columns** with no missing values or fully duplicated rows.

Preparation included:

- converting `Date` to datetime;
- validating monetary, quantity, engagement, delivery, and rating ranges;
- checking category consistency;
- confirming that both `Order_ID` and `Customer_ID` are unique;
- creating a separate one-hot encoded modeling dataset.

Each row represents one unique customer order and its associated purchase experience.

---

## 2. Univariate Analysis

Univariate analysis was used to understand distributions, skewness, outliers, category balance, and target structure before studying relationships between variables.

### User Behavior & Engagement

The data shows meaningful variation across sales, profit, browsing, and engagement variables. Several numerical distributions are non-normal or multi-modal, while category frequencies are uneven.

Product category explains some sales and profit differences, but browsing time alone does not clearly separate users who add products to the cart from those who do not.

### Sales & Customer Insights

The dataset is technically clean but unusually uniform. Numerical variables show limited variation in structure, categorical distributions are relatively even, and grouped comparisons show little separation.

This early result raised a question that became important later: **does the dataset actually contain enough signal to support useful prediction?**

### Customer Churn

The strongest univariate churn pattern appears in **customer tenure**. Churned customers are concentrated more heavily among shorter-tenure customers.

`WarehouseToHome`, `CashbackAmount`, and `DaySinceLastOrder` show weaker but potentially useful differences.

Because only **16.84%** of customers churned, later modeling uses metrics beyond accuracy.

### Purchase Experience

`Unit_Price`, `Discount_Amount`, and `Total_Amount` are strongly right-skewed, with a smaller number of high-value transactions extending the upper tail.

Product category shows meaningful differences in order value, while customer rating changes relatively little across delivery-time groups.

---

## 3. Bivariate Analysis

The third stage examines pairwise relationships using correlations, scatterplots, pairplots, grouped comparisons, and time-based analysis.

### User Behavior & Engagement

Financial variables such as `Sales`, `Profit`, and `Shipping Cost` are strongly related to one another.

However, those variables are not the most useful signals for purchase intent. Engagement actions such as **`Like` and `Share` show clearer relationships with `Add to Cart`**.

This distinction becomes important during predictive modeling: variables describing customer interaction are more informative for cart behavior than many transaction-level financial variables.

### Sales & Customer Insights

Numeric correlations are close to zero, scatterplots show little structure, and time-based patterns remain relatively stable.

The combination of highly even distributions and weak relationships suggests that the dataset may be highly simplified or synthetic-like, although that cannot be established from the analysis alone.

Rather than forcing a business conclusion, this dataset is retained as a **negative analytical result**.

### Customer Churn

The clearest bivariate churn relationships are:

- `Tenure` vs. `Churn`: approximately **−0.35**
- `Complain` vs. `Churn`: approximately **+0.25**
- `CouponUsed` vs. `OrderCount`: approximately **+0.75**

Longer-tenure customers are less likely to churn, while customers with complaints show higher churn rates.

Most individual variables do not perfectly separate churned and retained customers, supporting the need for multivariate models.

### Purchase Experience

The strongest pairwise relationship is between:

- `Unit_Price` and `Total_Amount`: approximately **0.79**

This relationship is expected because order value is mechanically connected to price and quantity.

Customer rating shows limited separation across product categories, and returning-customer status does not strongly separate average order value.

---

## 4. Multivariate Analysis

The multivariate stage evaluates whether combinations of variables reveal structure that is not visible in one- or two-variable analysis.

Methods include:

- correlation heatmaps;
- grouped heatmaps;
- bubble plots;
- Principal Component Analysis (PCA);
- standardized feature spaces;
- linear regression diagnostics.

### User Behavior & Engagement

PCA separates two broad directions:

- a **financial dimension**, driven by variables such as sales, profit, and shipping cost;
- an **engagement dimension**, associated with interaction behavior.

The first principal component explains about **32.5%** of variance, while the first two components explain about **49.7%**.

The first two PCs therefore summarize meaningful structure but do not capture enough information to cleanly separate add-to-cart behavior.

This finding supports using supervised models rather than relying on low-dimensional PCA alone.

### Sales & Customer Insights

Multivariate analysis confirms the earlier warning signs:

- correlations remain close to zero;
- grouped churn probabilities are nearly uniform;
- PCA variance is distributed rather than concentrated;
- tested regressions produce negative test R² values.

The dataset is therefore deprioritized for substantive predictive conclusions.

### Customer Churn

Multivariate analysis reinforces the importance of:

- `Tenure`
- `Complain`
- recent-order behavior
- cashback and order-history variables

PCA does not produce a clean churn/no-churn separation, which suggests that the class boundary is not easily represented in only two linear dimensions.

This dataset remains the strongest candidate for nonlinear classification.

### Purchase Experience

The first two principal components explain about **35.5%** of total variance.

Price-related features dominate the strongest correlations and the first PCA direction. Linear regression can produce strong apparent performance when direct components of `Total_Amount` are included, but this is not a useful predictive setup because `Unit_Price`, `Quantity`, and `Discount_Amount` directly determine transaction value.

That observation motivated a **leakage-aware modeling design** in the final notebook.

---

# Predictive Modeling

Random Forest models were used as nonlinear baselines and tuned models. Depending on the task, the workflow includes:

- train/test splitting;
- stratification for imbalanced classification;
- cross-validation;
- macro F1;
- precision and recall;
- ROC-AUC;
- RMSE and MAE;
- GridSearchCV hyperparameter tuning;
- confusion matrices;
- feature importance.

## Modeling Results

| Dataset / Task | Final Result | Interpretation |
|---|---|---|
| **Add-to-Cart Classification** | Accuracy **0.698**, Macro F1 **0.691** | Moderate, stable predictive signal |
| **Churn Probability Regression** | R² **−0.003**, RMSE **0.287**, MAE **0.248** | No meaningful predictive signal |
| **Customer Churn Classification** | Accuracy **0.983**, Macro F1 **0.969** | Strongest predictive dataset |
| **High-Value Order Classification** | Accuracy **0.806**, Recall **0.836**, F1 **0.683**, ROC-AUC **0.887** | Useful recall-oriented classifier |

---

## Model 1 — Predicting Add-to-Cart Behavior

### Target

`Add to Cart`

The positive class represents approximately **66.7%** of observations.

### Leakage-Aware Feature Selection

Identifiers and transaction-stage variables that could occur after or directly around the purchase event were excluded from the predictor set. The goal was to predict purchase intent using customer profile, product context, time context, and engagement behavior.

A stratified 80/20 train/test split was used.

### Baseline

The baseline Random Forest reached approximately:

- Accuracy: **0.690**
- Macro F1: **0.648**

Five-fold cross-validation produced a mean macro F1 of approximately **0.648** with a standard deviation of only **0.003**, indicating stable but moderate performance.

### Tuned Model

The tuned Random Forest reached:

- Accuracy: **0.698**
- Macro F1: **0.691**
- Best CV Macro F1: **0.690**

Best parameters:

```text
class_weight = balanced
max_depth = None
min_samples_leaf = 2
min_samples_split = 2
n_estimators = 300
```

### Most Important Features

| Feature | Importance |
|---|---:|
| `Share` | 0.194 |
| `Like` | 0.140 |
| `Browsing Time (min)` | 0.110 |
| `Gender_Male` | 0.082 |
| `Age` | 0.068 |

The strongest model features are engagement variables, supporting the earlier EDA finding that **customer interaction behavior provides more purchase-intent signal than most financial variables**.

---

## Model 2 — Predicting Churn Probability

Dataset 2 uses continuous `Churn_Probability` as a regression target.

The result is intentionally retained even though the model performs poorly.

### Cross-Validation

Mean five-fold results:

- CV R²: approximately **−0.045**
- CV RMSE: approximately **0.295**
- CV MAE: approximately **0.253**

### Tuned Test Performance

- R²: **−0.003**
- RMSE: **0.287**
- MAE: **0.248**

A negative R² means the model fails to outperform a simple mean-based prediction.

This result is analytically useful: **more model complexity cannot create signal that is not present in the data**. The dataset was therefore not used as primary evidence for the broader project.

---

## Model 3 — Customer Churn Classification

This is the strongest predictive task in the project.

### Target

`Churn`

Class distribution:

- No Churn: **83.16%**
- Churn: **16.84%**

Because the target is imbalanced, performance is evaluated using macro F1 in addition to accuracy.

### Baseline Random Forest

Holdout performance:

- Accuracy: **0.983**
- Macro F1: **0.969**

Five-fold cross-validation:

- Mean Accuracy: approximately **0.966**
- Mean Macro F1: approximately **0.935**

The tuned model did not improve the baseline holdout result, so the baseline model was retained for interpretation.

### Important Features

| Feature | Importance |
|---|---:|
| `Tenure` | 0.201 |
| `CashbackAmount` | 0.095 |
| `WarehouseToHome` | 0.075 |
| `Complain` | 0.064 |
| `NumberOfAddress` | 0.062 |
| `DaySinceLastOrder` | 0.058 |

`Tenure` is the dominant feature, consistent with the earlier EDA and correlation analysis.

Feature importance is interpreted as **model reliance, not causal effect**.

The strong results are also treated cautiously. External or time-based validation would be necessary before assuming similar performance on new customer populations.

---

## Model 4 — High-Value Order Classification

A new binary target was created:

`High_Value_Order = 1` for transactions at or above the 75th percentile of `Total_Amount`.

This creates a 25% positive class.

### Leakage Prevention

To avoid an obvious prediction, the following variables were removed:

- `Total_Amount`
- `Unit_Price`
- `Quantity`
- `Discount_Amount`

These variables either define the target directly or are mechanical components of transaction value.

The model therefore relies on indirect signals such as:

- product category;
- age;
- browsing behavior;
- returning-customer status;
- device type;
- payment method;
- calendar context.

### Baseline Random Forest

- Accuracy: **0.838**
- Precision: **0.720**
- Recall: **0.576**
- F1: **0.640**
- ROC-AUC: **0.891**

### Tuned Random Forest

- Accuracy: **0.806**
- Precision: **0.577**
- Recall: **0.836**
- F1: **0.683**
- ROC-AUC: **0.887**

Tuning shifts the model toward **higher recall**:

- Recall: **0.576 → 0.836**
- F1: **0.640 → 0.683**

This comes at the cost of lower precision and overall accuracy.

The trade-off is business-dependent. If failing to identify a potentially high-value order is more costly than reviewing additional false positives, the tuned model provides the more recall-oriented operating point.

### Most Important Signals

`Product_Category_Electronics` is the strongest feature in the tuned model, followed by other product-category indicators. Age, session duration, and pages viewed contribute smaller amounts of signal.

---

# Key Findings

### 1. Engagement behavior is more useful than transaction variables for purchase-intent prediction

`Share`, `Like`, and browsing time consistently emerge as the strongest predictors of add-to-cart behavior.

### 2. Customer tenure is the clearest churn signal

Across exploratory, bivariate, multivariate, and predictive analysis, `Tenure` repeatedly appears as the most informative churn-related variable.

### 3. Model quality depends on data quality, not only algorithms

Dataset 2 remains weak even after multivariate analysis and hyperparameter tuning. Keeping this negative result prevents overclaiming and demonstrates the importance of validating the information content of a dataset.

### 4. Leakage can make a model appear stronger than it really is

The purchase-experience analysis shows that direct components of order value can produce apparently strong relationships with `Total_Amount`. Removing these variables creates a harder but more meaningful prediction problem.

### 5. The best metric depends on the business objective

For high-value-order classification, tuning substantially improves recall while reducing precision and accuracy. Model selection should therefore depend on the operational cost of false negatives versus false positives.

---


# Business Implications & Recommended Actions

The analytical results suggest different business uses for each dataset. Importantly, not every dataset should lead to an operational model.

### 1. Use engagement signals to prioritize purchase-intent targeting

For Dataset 1, `Share`, `Like`, and `Browsing Time (min)` were the strongest predictors of add-to-cart behavior. This suggests that customer interaction signals may be more useful for purchase-intent targeting than relying only on transaction-level financial variables.

**Potential business use:**
- trigger recommendation or remarketing workflows for highly engaged users;
- prioritize users showing multiple engagement signals rather than using browsing time alone;
- use these signals as inputs to cart-conversion experiments or personalized messaging.

Because the model has only moderate predictive performance, these signals should support targeting decisions rather than act as a fully automated decision rule.

### 2. Prioritize newer customers and customers with complaints for retention review

For Dataset 3, `Tenure` was the strongest churn-related feature, and `Complain` also showed an important relationship with churn.

**Potential business use:**
- monitor newer customers more closely during the early customer lifecycle;
- flag customers with recent complaints for proactive support or retention outreach;
- combine tenure, complaint history, order activity, recency, and cashback behavior into a churn-risk workflow;
- evaluate retention campaigns using recall, precision, and business cost rather than accuracy alone.

The strong model performance suggests these variables are useful for prioritization, but the results should still be validated on future or external customer populations before production use.

### 3. Use recall-oriented models when missing high-value opportunities is costly

For Dataset 4, hyperparameter tuning increased recall for high-value orders from **0.576 to 0.836**, while precision and overall accuracy decreased.

**Potential business use:**
- use the tuned model when identifying as many potential high-value orders as possible is more important than avoiding false positives;
- support premium-service routing, targeted promotions, or prioritization workflows;
- choose the classification threshold based on the relative cost of false negatives versus false positives.

This result demonstrates that the "best" model depends on the business objective rather than a single performance metric.

### 4. Do not operationalize Dataset 2 without better signal

Dataset 2 is an important negative result. It is technically clean, but its features do not explain `Churn_Probability` well. Negative R² values across cross-validation folds confirm that the model does not outperform a simple mean-based prediction.

**Business implication:**
A company should **not** use this model to drive retention offers, customer prioritization, or churn interventions. Doing so could allocate marketing or retention budget to the wrong customers.

A better next step would be to collect or integrate stronger behavioral variables such as:

- recent customer activity;
- tenure;
- complaint history;
- engagement events;
- recency and frequency of orders;
- customer-service interactions.

The key lesson is that a clean dataset is not automatically a decision-ready dataset. Validating signal quality should come before operational deployment.

---

# Technical Skills Demonstrated

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- data cleaning and validation
- missing-value analysis
- categorical encoding
- exploratory data analysis
- correlation analysis
- time-based analysis
- feature engineering
- PCA and feature scaling
- linear regression
- Random Forest classification and regression
- class-imbalance evaluation
- stratified train/test splitting
- cross-validation
- hyperparameter tuning with GridSearchCV
- confusion-matrix analysis
- precision, recall, F1, macro F1, ROC-AUC
- RMSE, MAE, and R²
- feature-importance interpretation
- leakage-aware model design
- business-oriented analytical communication

---

# What I Would Improve for Production

This repository is an analytical portfolio project rather than a deployed production system. A production version would strengthen the workflow by:

- placing imputation, encoding, and other learned preprocessing steps inside scikit-learn pipelines;
- validating models on genuinely future or external customer data;
- adding probability calibration and threshold optimization where business costs are known;
- evaluating fairness and stability across relevant customer segments;
- tracking experiments and model versions;
- adding automated data-validation and model-monitoring checks.

---

# Reproducing the Analysis

Clone the repository and install the project dependencies:

```bash
pip install -r requirements.txt
```

Place the source datasets in the `data/` directory, then run the notebooks in order:

```text
01_data_preparation.ipynb
02_univariate_analysis.ipynb
03_bivariate_analysis.ipynb
04_multivariate_analysis.ipynb
05_predictive_modeling.ipynb
```

The notebooks are intentionally sequential: cleaned datasets created during preparation are reused in later stages.

---

## Final Takeaway

This project demonstrates an end-to-end data science workflow across four different e-commerce datasets rather than optimizing a single model in isolation.

The strongest evidence comes from the churn dataset, where customer history and behavioral variables produce both interpretable patterns and strong predictive performance. The engagement dataset provides moderate purchase-intent signal, while the purchase-experience dataset illustrates how leakage-aware feature design and business-specific metric trade-offs change model interpretation. The weak Sales & Customer Insights result is retained as evidence that responsible analysis also requires recognizing when a dataset does **not** support a useful predictive conclusion.

The central lesson is that effective data science requires more than fitting models: it requires understanding the data-generating context, validating assumptions, choosing metrics that match the business problem, preventing leakage, and communicating both successful and unsuccessful results clearly.
