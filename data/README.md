# Dataset

This project uses the **Bank Customer Churn Modelling** dataset (10,000 records, 14 columns).

**Download:** https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

1. Download `Churn_Modelling.csv`
2. Place it in this folder: `data/Churn_Modelling.csv`

The CSV is gitignored so the repo stays lightweight.

## Columns

| Column | Description |
|---|---|
| RowNumber, CustomerId, Surname | Identifiers (dropped before training) |
| CreditScore | Customer credit score |
| Geography | France / Germany / Spain |
| Gender | Female / Male |
| Age | Customer age |
| Tenure | Years as a customer |
| Balance | Account balance |
| NumOfProducts | Bank products held |
| HasCrCard | Holds a credit card (0/1) |
| IsActiveMember | Active member flag (0/1) |
| EstimatedSalary | Estimated annual salary |
| **Exited** | **Target** — 1 if the customer left the bank |
