# Forecasting Outperformers for S&P 500 Stocks Using Machine Learning
This repository contains the implementation of a project aimed at forecasting the relative performance of S&P 500 stocks within a one-week horizon using machine learning (ML). By framing this task as a binary classification problem, the project predicts whether a stock’s active return—defined as the difference between its individual return and the benchmark return—exceeds a predefined threshold.

## Project Overview
This project explores how machine learning models can help identify S&P 500 stocks likely to outperform in the short term. Predictions are tested through a simulated trading strategy to evaluate their real-world applicability. It demonstrates how data-driven approaches can support decision-making in financial markets while identifying opportunities for further refinement.

## Running the Code

To reproduce the results of this project, follow these steps:

1. Navigate to the `dslc_documentation/` directory.
2. Run the Jupyter notebooks in order, starting with `01_cleaning.ipynb` through to `04_cleaning.ipynb`. These notebooks process the raw data and generate the required findings.
3. The raw data used in these notebooks can be found in the `data/` directory, and any output plots (as well as interim preprocessed data files) are stored here. 

Each notebook builds on the previous one, so it is important to execute them sequentially to ensure the analysis proceeds correctly.

## Key ML models used:

- Logistic Regression
- Random Forest
- XGBoost
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Stacking Classifier (combining Logistic Regression, Random Forest, and XGBoost)
## Problem Formulation
The task of predicting stock outperformance is framed as a binary classification problem. The goal is to determine whether a given stock will outperform the S&P 500 index in the following period. The target variable is defined as a binary label, where:

Stocks achieving active returns above a predefined threshold N are considered outperformers.

Active Return Formula:

$$ar_{i,t} = r_{i,t} - bm_t$$

where $r_{i,t}$ represents the return of stock $i$ at time $t$, and $bm_t$ denotes the benchmark return.

Thresholded Target:

$$y_{i,t} = \mathbb I (\  a_{ri,t} > 0.01) $$
## Dataset
The dataset used in this project spans January 1, 2014 – September 30, 2024, with daily price data for S&P 500 constituent stocks and the S&P 500 index itself. The data was divided into:

Training Set: 60% (2015–2020)
Validation Set: 20% (2021–2022)
Test Set: 20% (2023–2024)
## Key preprocessing steps:

1. Standardized numeric formats and corrected non-numeric values.
2. Calculated lagged features for 12 weeks.
3. Addressed missing values and ensured continuity across holidays and non-trading days.
4. Computed active returns (difference between individual stock and benchmark returns).
## Methodology
The project applies a range of machine learning models for classification:

- Logistic Regression: A linear baseline model for simplicity and interpretability.
- K-Nearest Neighbors (KNN): Classifies stocks based on feature similarity.
- Random Forest: An ensemble method to capture non-linear feature interactions.
- Support Vector Classifier (SVC): Uses kernel methods for non-linear decision boundaries.
- XGBoost: An optimized gradient boosting algorithm for handling imbalanced datasets.
- Stacking Classifier: Combines Logistic Regression, Random Forest, and XGBoost to improve predictions through a meta-model.
## Trading Strategy
Predictions from the Stacking Classifier are used to simulate a trading strategy:
1. Portfolio Construction: Allocate capital evenly among stocks predicted to outperform each week.
2. Performance Metrics:
   -  Compare portfolio returns against the S&P 500 benchmark.
   -  Evaluate annualized returns and active returns.
### Key formula:

Annualized Return:

$$
R_{annualized} = \bigg(\frac{V_{end}}{V_{start}} \bigg) ^{\frac{252}T} - 1
$$
where $V_{start}$ and $V_{end}$ are portfolio values at the start and end of the period, and $T$ is the number of trading days over the period. 

Annualized Active Return:

$$
R_{ann,\ 
active} = R_{ann,\ portfolio} - R_{ann,\ 
benchmark}

$$
## Repository Structure

```yaml.
├── data/
│   ├── Raw Data 
│
├── dslc_documentation/
│   ├── functions/
│       ├── constants.py: Constants used across files
│       ├── helper_fns.py: Utility functions
│   ├── 01_cleaning.ipynb: Data cleaning
│   ├── 02_eda.ipynb: Exploratory Data Analysis
│   ├── 03_prediction.ipynb: Main prediction workflow
│   ├── 03_prediction_SVC.ipynb: SVC implementation
│   ├── 04_trading_strategy.ipynb: Simulated trading strategy
│
├── README.md: Main documentation
```
## Results
### Model Performance
Each of the machine learning models demonstrated unique strengths and challenges:

- Logistic Regression: This model provided a reliable baseline, but its linear nature limited its ability to capture complex relationships within the data.
- K-Nearest Neighbors (KNN): It performed adequately but faced challenges with scalability and imbalanced data.
- Random Forest: This model effectively captured non-linear relationships and interactions between features but required careful hyperparameter tuning to prevent overfitting.
- Support Vector Classifier (SVC): While this model was strong at handling non-linear data, it suffered from lower precision due to class imbalance.
- XGBoost: This gradient boosting algorithm excelled at handling imbalanced datasets and delivered robust performance, especially after tuning.
- Stacking Classifier: By combining Logistic Regression, Random Forest, and XGBoost, this ensemble model demonstrated the most balanced and consistent performance across various metrics, making it the best-performing model overall.
### Trading Strategy Results
The predictions from the Stacking Classifier were used to simulate a trading strategy. The strategy involved allocating equal capital to all stocks predicted to outperform the benchmark each week.

- Portfolio Performance: While the simulated portfolio generated notable returns over the test period, it slightly underperformed the S&P 500 benchmark due to market volatility and practical constraints.
- Insights: The strategy demonstrated the potential of machine learning for portfolio construction but revealed limitations, such as ignoring transaction costs, liquidity constraints, and other real-world factors.
## Future Work
Future work could:
- Explore advanced time-series models like Long Short-Term Memory (LSTM) networks or Transformers.
- Incorporate alternative data sources like sentiment analysis and macroeconomic indicators.
Address real-world trading constraints, such as transaction costs and liquidity. 
- Learn optimal portfolio alllocation enhancements rather than just selecting stocks expected out outperform. 
## Contributors
- Aashish Khubchandani (Cornell Tech)
- Abhijay Rane (Cornell Tech)
