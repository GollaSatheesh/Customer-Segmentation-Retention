# Customer Segmentation and Retention Prediction

## About the Project

This is my Data Science project based on an Online Retail dataset.

The main goal of this project is to understand customer purchasing behaviour, divide customers into different groups, and predict which customers may stop purchasing.

I used Python, Machine Learning, and Streamlit to build this project.

## Problem Statement

Businesses need to understand their customers to improve sales and maintain long-term relationships.

Some customers purchase regularly, while others may stop purchasing after some time.

In this project, I tried to answer three questions:

- Which customers are valuable to the business?
- Which customers may stop purchasing?
- What actions can a business take to retain those customers?

## Dataset

I used an Online Retail dataset containing customer transaction details.

The dataset includes:

- Invoice number
- Product details
- Quantity
- Invoice date
- Price
- Customer ID
- Country

After cleaning the data, I used it for customer analysis and model building.

## Project Workflow

### 1. Data Cleaning

- Checked missing values.
- Removed cancelled transactions.
- Handled invalid quantity and price values.
- Converted date columns into the correct format.
- Created a revenue column.

### 2. Exploratory Data Analysis (EDA)

I analysed customer purchasing behaviour and sales trends.

Some of the things I explored were:

- Monthly revenue.
- Number of repeat customers.
- One-time customers.
- Customer purchasing patterns.

### 3. RFM Analysis

I used RFM analysis to understand customer behaviour.

RFM stands for:

- Recency: How recently a customer purchased.
- Frequency: How many times a customer purchased.
- Monetary: How much money a customer spent.

Based on these values, I divided customers into groups such as Champions, Loyal Customers, At Risk, and Lost Customers.

### 4. Customer Segmentation Using K-Means

I applied the K-Means clustering algorithm to group customers based on their RFM values.

I used StandardScaler to scale the features before applying K-Means.

I selected four clusters and studied the purchasing behaviour of each group.

### 5. Customer Inactivity Prediction

I also built a Machine Learning model to predict whether a customer may become inactive.

For this task, I created a target variable using customer purchase history.

A customer was labelled inactive if they did not make another purchase during the next 90 days.

I used these features:

- Frequency
- Monetary
- Recency

I trained and compared two models:

- Logistic Regression
- Random Forest Classifier

I selected Logistic Regression as my main model for the retention prediction feature.

I also adjusted the prediction threshold to 0.3 because identifying more potentially inactive customers is important for retention.

### 6. Customer Retention Suggestions

Based on customer segments and predicted inactivity risk, I suggested different actions.

For example:

- Champions: Provide special offers and rewards.
- Loyal Customers: Encourage them with loyalty benefits.
- Potential Loyalists: Give personalised recommendations.
- Needs Attention: Send re-engagement offers.

These are suggestions based on the analysis, not guaranteed results.

## Streamlit Dashboard

I developed a Streamlit dashboard to display the project results.

The dashboard includes:

- Customer summary.
- RFM customer segments.
- Customer inactivity prediction results.
- Risk analysis by customer segment.
- A form to enter customer details and predict inactivity probability using the saved Logistic Regression model.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Project Folder Structure

```text
Customer-Segmentation-Retention/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
│
└── project_data/
    ├── retention_predictions.csv
    └── historical_rfm.csv
```

## How to Run the Project

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### Step 2: Open the project folder

```bash
cd Customer-Segmentation-Retention
```

### Step 3: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit application

```bash
streamlit run app.py
```

The dashboard will open in your browser.

## What I Learned

Through this project, I learned how to:

- Clean and analyse real-world transaction data.
- Perform customer segmentation using RFM analysis.
- Apply K-Means clustering.
- Build classification models using Scikit-learn.
- Understand model evaluation metrics.
- Adjust prediction thresholds based on business needs.
- Connect a trained Machine Learning model to a Streamlit application.

## Limitations

- The predictions are based on historical customer purchasing behaviour.
- A predicted inactive customer may still make another purchase.
- The model cannot guarantee that a customer will leave.
- The retention suggestions are recommendations and have not been tested through real business campaigns.

## Future Improvements

- Improve the prediction model using additional customer features.
- Deploy the dashboard online.
- Add more customer-level insights.
- Test different retention strategies using real campaign results.

## Conclusion

This project helped me understand how Data Science and Machine Learning can be used to study customer behaviour and support business decisions.

I built this project to practise the complete process, from data cleaning and analysis to Machine Learning and dashboard development.