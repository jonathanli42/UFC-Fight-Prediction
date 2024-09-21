# **UFC Fight Prediction Model**

This project is focused on predicting the outcomes of UFC fights using historical fighter data. The workflow consists of **scraping** data from UFCStats.com, **cleaning and feature engineering**, and building a **machine learning model** to predict the winner of a fight between two fighters.

---

## **Project Overview**

The goal of this project is to create a data pipeline that **scrapes**, **cleans**, and **processes UFC fight data** to make predictions on fight outcomes. Using scraped data from UFCStats.com, the project extracts fighter statistics and historical fight details, engineers relevant features, and trains a machine learning model to predict the result of future fights. **The user can input any two fighters, and the model will output the predicted winner and the probability of each fighter winning.**

---

## **Results**

The model was trained and tested on a dataset using **XGBClassifier** with **GridSearchCV** for hyperparameter tuning and cross-validation. The model achieved an **accuracy of 85%**, with the following performance metrics on the test set:
#### Classification Report

|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| **0**         | 0.85      | 0.85   | 0.85     | 459     |
| **1**         | 0.85      | 0.85   | 0.85     | 455     |
| **accuracy**  |           |        | 0.85     | 914     |
| **macro avg** | 0.85      | 0.85   | 0.85     | 914     |
| **weighted avg** | 0.85   | 0.85   | 0.85     | 914     |

#### Overall Performance
- **Test set accuracy**: 0.85
- **F1 Score**: 0.85

## **How It Works**

1. **Data Scraping**: The project scrapes fighter stats, fight history, and fight metrics from UFCStats.com.
2. **Data Cleaning**: The scraped data is cleaned, aggregated, and features are engineered to create a usable dataset.
3. **Modeling**: Using an **XGBoost** model, the project predicts fight outcomes based on fighter stats and performance history.

---

## **Data CSV Files**

- **combined_fighter_data.csv**: The initial raw data scraped from UFCStats, including various fighter statistics.
- **specific_fighter_data.csv**: The result of testing two specific fighters after data scraping. This file includes the fighter-specific statistics retrieved from the initial scrape.
- **fight_comp_data.csv**: Contains the paired data for two fighters, including calculated differences in performance metrics. This dataset is the result of merging and processing two fighters' data and is used right before feeding into the ML model.
- **cleaned_data_ml.csv**: The output generated after cleaning the raw data through the comp_fighter_clean notebook.
- **cleaned_data_ml_comp.csv**: This file contains the data that has been processed and formatted, ready for input into the machine learning model. It includes fighter statistics and comparison metrics between two fighters.

---

## **Notebooks**

- **comp_fighter_clean.ipynb**: This notebook is used for testing the initial data cleaning methods.
- **data_clean_round_data.ipynb**: Handles data cleaning, feature engineering, and aggregation of round-by-round fight data.
- **ml_model_eda.ipynb**: This notebook is responsible for performing exploratory data analysis (EDA) on the cleaned data.
- **ml_model.ipynb**: This notebook tests different machine learni ng models and hyperparameters.

---

## **Python Files**

### **Scraping Files**

- **scrape_basic_stats.py**: Scrapes basic fighter stats such as name, record, physical attributes, stance, and date of birth.
- **scrape_fight_dates.py**: Gathers data on fighter's fight history, including fight date, opponent, result (win/loss), and method of victory/defeat.
- **scrape_fight_round_details.py**: Scrapes detailed round-by-round fight metrics like striking accuracy, head/body/leg strike distribution, and takedown stats.
- **scrape_fight_urls.py**: Scrapes all fight URLs from UFCStats.com to identify and retrieve specific fight data.
- **scrape_fighters.py**: Scrapes all fighter URLs from UFCStats.com to create a list of fighters whose data will be pulled.
- **scrape_run.py**: Combines all the scraping scripts, extracting comprehensive data on each fighter’s stats and fight history, and saves it to `combined_fighter_data.csv`.

### **Cleaning and Model Logic**

- **clean_data_fighters.py**: Cleans and processes fighter statistics, preparing the data for the ML model by removing names and one-hot encoding categorical variables like stance and weight class.
- **fighter_comparison.py**: Compares two fighters by scraping and storing their data in `specific_fighter_data.csv`.
- **helper_clean_data_methods.py**: Provides helper functions for cleaning, feature engineering, and handling tasks like data imputation and one-hot encoding.
- **model_run.py**: The main script where users input fighters and get fight outcome predictions based on the trained ML model. The input dictionary of fighter pairs is customizable.
- **model_ufc_prediction.py**: Contains the prediction logic, using **GridSearchCV** for hyperparameter tuning and **XGBClassifier** for the model, optimized with **StratifiedKFold** cross-validation.

---

## **How to Run**

1. **Clone this repository.**

2. **Install the necessary Python libraries**:
    ```
    pip install -r requirements.txt
    ```

3. **Run the scraping files to collect the data**:
    ```
    python scrape_run.py
    ```

4. **Update the fighter pairs in `model_run.py`**:
    ```python
    fights_data = [
        {
            'fighter_1': 'Israel Adesanya',
            'fighter_2': 'Dricus Du Plessis',
            'is_title_fight': True,
            'weight_class': 'Middleweight',
            'is_male_fight': True
        }
    ]
    ```

5. **Run `model_run.py`** to get the fight prediction:
    ```
    python model_run.py
    ```

---

## **Disclaimer**

This project is intended for **educational and research purposes only**. The machine learning model is **not guaranteed** to predict real-world fight outcomes accurately and should **not be used for gambling, betting, or any other financial decisions**. Use the model responsibly and at your own risk.
