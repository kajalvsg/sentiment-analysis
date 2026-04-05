# Sentiment Analysis & NLP Text Classifier

End-to-end sentiment classification project using Python, NLTK, scikit-learn, and TF-IDF.

This repository includes:
- NLP preprocessing (tokenization-style cleaning, stopword removal, lemmatization)
- TF-IDF vectorization (unigrams + bigrams)
- Baseline models: Logistic Regression and Naive Bayes
- Evaluation: accuracy, precision, recall, F1-score
- Inference via CLI and Streamlit web app

## Project Structure

```text
sentiment-analysis/
├── app.py                         # Streamlit app
├── main.py                        # CLI prediction entrypoint
├── requirements.txt
├── README.md
├── data/
│   └── twitter_sentiment.csv
├── outputs/
│   └── model_results.csv          # generated after training
├── models/                        # generated after training
│   ├── best_model.joblib
│   ├── logisticregression_pipeline.joblib
│   └── naivebayes_pipeline.joblib
├── notebooks/
│   └── sentiment_analysis.ipynb
└── src/
    ├── __init__.py
    ├── preprocess.py
    ├── train.py
    └── predict.py
```

## Local Setup

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the Model

```bash
python -m src.train
```

This will:
- train Logistic Regression and Naive Bayes pipelines
- save model artifacts in `models/`
- save metrics in `outputs/model_results.csv`

## Run Predictions (CLI)

```bash
python main.py
```

Type text and get predicted sentiment + confidence.

## Run Web App (Streamlit)

```bash
streamlit run app.py
```

Open the shown local URL in your browser.

## Example Resume Bullets

- Built a text classification pipeline using NLP preprocessing (cleaning, stopword removal, lemmatization) and TF-IDF vectorization to classify sentiment in customer review data.
- Evaluated baseline models (Logistic Regression and Naive Bayes) using precision, recall, F1-score, and confusion-matrix compatible outputs.

## Push to GitHub

```bash
git init
git add .
git commit -m "Build end-to-end sentiment analysis NLP classifier with Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/sentiment-analysis.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Click **New app** and select your repo.
4. Set main file path to `app.py`.
5. Deploy.

After deployment, add the live app link to your GitHub README for portfolio visibility.
