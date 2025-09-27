## Group: Jana Zwein and Sleiman Mhanna

## IBM HR Analytics Attrition Prediction

This project predicts employee attrition using machine learning (XGBoost) on the IBM HR Analytics dataset. It provides a Streamlit web app for interactive predictions and a Jupyter notebook for data exploration, feature engineering, and model training.

## Problem Statement

Employee attrition is one of the most pressing challenges for modern organizations. High turnover disrupts operations, increases recruitment and training costs, and can negatively affect team morale and productivity. For HR teams, knowing which employees are at risk of leaving before it happens is crucial to take proactive measures—whether it’s targeted retention programs, career development plans, or workload adjustments.

This project tackles attrition from a business perspective: by leveraging employee data, we aim to build a predictive model that identifies potential flight risks. The goal is not just prediction—it’s providing HR managers with actionable insights that save money, retain top talent, and maintain organizational stability. With a reliable attrition risk model, companies can transform workforce management from reactive to strategic, turning employee retention into a competitive advantage.

## Features
- Data cleaning, feature engineering, and encoding
- Skewness correction and normalization
- Model training and evaluation (XGBoost)
- Model and scaler persistence
- Streamlit UI for real-time predictions
- Ready-to-use requirements.txt and .gitignore

## How to Run

1. Clone the repository
```sh
git clone https://github.com/SleimanMh/AI-Hackathon/edit/main/README.md
cd AI-Hackathon
git checkout /feature
```

2. Install dependencies
```sh
pip install -r requirements.txt
```

3. Download the dataset
This project uses the IBM HR Analytics Attrition dataset from Kaggle. The notebook will automatically download it using `kagglehub`.

4. Run the Jupyter notebook (for training and exploration)
```sh
jupyter notebook Untitled3.ipynb
```

5. Run the Streamlit app (for predictions)
```sh
streamlit run app.py
```

- Fill in the employee details in the web UI to get attrition predictions and probabilities.

File Structure
- `Untitled3.ipynb` — Data cleaning, feature engineering, model training, and evaluation
- `app.py` — Streamlit web app for predictions
- `requirements.txt` — Python dependencies
- `.gitignore` — Files and folders to exclude from git

Authors
- Sleiman Mhanna
- Jana Zwein

## License
This project is for educational and demonstration purposes.
