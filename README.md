Group: Jana Zwein and Sleiman Mhanna

IBM HR Analytics Attrition Prediction

This project predicts employee attrition using machine learning (XGBoost) on the IBM HR Analytics dataset. It provides a Streamlit web app for interactive predictions and a Jupyter notebook for data exploration, feature engineering, and model training.

Problem Statement
Employee attrition (turnover) is a major concern for organizations. Predicting which employees are likely to leave helps HR departments take proactive measures, reduce costs, and improve retention. This project aims to build a robust, interpretable model to predict attrition based on employee data.

Features
- Data cleaning, feature engineering, and encoding
- Skewness correction and normalization
- Model training and evaluation (XGBoost)
- Model and scaler persistence
- Streamlit UI for real-time predictions
- Ready-to-use requirements.txt and .gitignore

How to Run

1. Clone the repository
```sh
git clone https://github.com/SleimanMh/AI-Hackathon/edit/main/README.md
cd AI-Hackathon
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
