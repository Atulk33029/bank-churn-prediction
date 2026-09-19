# Power BI Dashboard — build guide (~90 minutes)

`src/train_model.py` generates **`churn_predictions.csv`** in this folder. It contains every original
column plus four new ones the dashboard uses:

- `ChurnProbability` — model score between 0 and 1
- `PredictedChurn` — 0/1 at the 0.5 threshold
- `RiskBand` — Low / Medium / High
- `AgeGroup` — 18-30, 31-40, 41-50, 51-60, 60+

## Step 1 — Load the data
Power BI Desktop → **Get Data → Text/CSV** → select `churn_predictions.csv` → Load.

## Step 2 — Create these measures
In Modeling → New Measure:

```dax
Total Customers = COUNTROWS('churn_predictions')

Churned Customers = CALCULATE([Total Customers], 'churn_predictions'[Exited] = 1)

Churn Rate = DIVIDE([Churned Customers], [Total Customers])

At Risk Customers = CALCULATE([Total Customers], 'churn_predictions'[RiskBand] = "High")

Avg Balance = AVERAGE('churn_predictions'[Balance])

Revenue At Risk = CALCULATE([Avg Balance] * [At Risk Customers])
```

## Step 3 — Page 1: Executive Overview
| Visual | Field setup |
|---|---|
| 4 KPI cards | Total Customers, Churn Rate, At Risk Customers, Revenue At Risk |
| Donut chart | Legend = Exited, Values = Total Customers |
| Clustered column | Axis = Geography, Value = Churn Rate |
| Clustered column | Axis = AgeGroup, Value = Churn Rate |
| Line chart | Axis = Tenure, Value = Churn Rate |
| Slicers (left panel) | Geography, Gender, IsActiveMember, RiskBand |

## Step 4 — Page 2: Risk Segmentation
| Visual | Field setup |
|---|---|
| Stacked bar | Axis = NumOfProducts, Legend = RiskBand, Value = Total Customers |
| Matrix | Rows = Geography, Columns = AgeGroup, Values = Churn Rate (conditional formatting: red scale) |
| Scatter | X = Age, Y = ChurnProbability, Size = Balance, Legend = Exited |
| Table | CustomerId, Surname, Age, Geography, Balance, ChurnProbability, RiskBand — sorted descending by ChurnProbability, top 20 |

## Step 5 — Polish
- Apply a consistent theme (View → Themes → pick one dark or one branded palette).
- Title each page: "Customer Churn — Executive Overview" / "High-Risk Customer Targeting".
- Add one text box per page with a one-line insight, e.g. *"German customers churn at roughly twice the rate of French customers."*

## Step 6 — Export for the repo
1. **File → Export → Export to PDF** → save as `powerbi/dashboard.pdf`
2. Screenshot each page → save as `reports/figures/powerbi_page1.png` and `powerbi_page2.png`
3. Save the `.pbix` file here as `churn_dashboard.pbix` and commit it (usually well under GitHub's 100 MB limit)
4. Embed the screenshots in the main README

## Free alternative if you don't have Power BI Desktop
Power BI Desktop is Windows-only and free. On Mac, use **Tableau Public** (free) or **Looker Studio**
(free, browser-based — upload the CSV directly and build the same visuals). Update the README's tech
stack line accordingly so it stays accurate.
