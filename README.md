# Cricket-Match-Outcome-Prediction-Using-Machine-Learning


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

## Overview

This repository implements **Cricket Match Winner Predictor**, an advanced machine learning-based web application designed to predict the outcomes of One Day International (ODI) cricket matches. The system focuses on binary classification: determining whether the chasing (batting second) team will win or lose. It leverages historical data from cricsheet.org spanning 2001–2024, incorporating factors like team performance, player statistics, match conditions, toss decisions, and venue details to provide accurate, data-driven predictions.

Developed as the final project for DATA 245 - Machine Learning Technology by Group 06, the project addresses the unpredictability of cricket by using ensemble models like Random Forest, which outperformed baselines. The predictions are delivered via an intuitive Flask-based web interface, enabling real-time input for teams, analysts, and fans. This tool not only forecasts match results but also offers insights into key influencing factors, enhancing strategic decision-making in sports analytics.

### Key Objectives
- Develop a robust ML model to predict ODI match outcomes (win/loss for the chasing team) using historical data.
- Analyze and quantify factors like player form, team dynamics, weather, pitch conditions, and toss impact.
- Create an accessible web interface for real-time predictions and visualizations.
- Advance cricket analytics by demonstrating the superiority of data-driven ML over traditional expert opinions.

### Why This Project?
Cricket's dynamic nature makes accurate predictions challenging, with traditional methods relying on subjective judgments or basic stats that miss complex interactions. Existing tools often lack real-time capabilities, interpretability, or integration of ball-by-ball data. Our solution stands out by:
- Using comprehensive feature engineering (e.g., runs left, balls left, run rates) to capture match pressure and dynamics.
- Achieving high accuracy (Random Forest: 91.60%) through ensemble methods that handle non-linear relationships.
- Providing transparent insights via feature importance and ROC curves.
- Deploying a user-friendly Flask app for practical use, bridging the gap between analytics and strategy.

## Features

- **Data Processing Pipeline**: Loads and merges match-level and ball-by-ball JSON data from cricsheet.org, handling over 20 years of ODI records.
- **Feature Engineering**:
  - Key features: BattingTeam, BowlingTeam, City, runs_left, balls_left, wickets_left, current_run_rate, required_run_rate, target.
  - Derived metrics: Pressure indices, run rates, and outcome encoding (1 for batting team win, 0 for loss).
  - Handles categorical variables (teams, cities) via one-hot encoding.
- **ML Models**:
  - Linear Regression (baseline: 84.69% accuracy).
  - Random Forest (best: 91.60% accuracy, handles non-linear patterns).
  - XGBoost (89.07% accuracy, robust for interactions).
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, ROC AUC (0.9786 for Random Forest), Confusion Matrix.
- **Web Interface (Flask)**: 
  - Input form for match details (teams, city, runs/balls/wickets left, rates, target).
  - Real-time prediction output with probability scores.
  - Visualizations: ROC curves, confusion matrices, performance charts.
- **Data Exploration**: Includes EDA visualizations like batsman strike rates, top teams/batsmen/bowlers, bowling economy.
- **Assumptions**: Clean dataset, binary classification (win/loss), encoded categoricals preserve importance.

## Architecture

The system follows a modular ML pipeline integrated with a web app:

### High-Level Components
1. **Data Layer**: JSON files from cricsheet.org → Pandas DataFrames → Preprocessing (cleaning, merging, feature engineering).
2. **ML Pipeline**:
   - Preprocessing: ColumnTransformer for one-hot encoding + passthrough for numerics.
   - Train-Test Split: 80-20.
   - Models: Pipeline with RandomForestClassifier (or alternatives).
   - Evaluation: Metrics computation, ROC, Confusion Matrix.
3. **Deployment Layer (Flask)**: 
   - Backend: Loads trained model, processes inputs, generates predictions.
   - Frontend: HTML/CSS forms for input, displays results and charts.
4. **Insights Module**: Feature importance from Random Forest, ROC analysis.

### Architecture Diagram
Below is a textual representation. For visuals, refer to `docs/architecture_diagram.png` (or generate via tools like Draw.io). UI screenshots are in `docs/images/` (e.g., main page, predictions).
<img width="1175" height="789" alt="image" src="https://github.com/user-attachments/assets/3931ae49-f0c3-499e-bd2c-b856471bac72" />


## Real-World Applications

- **Team Strategy**: Coaches use predictions for in-match decisions (e.g., batting/bowling choices).
- **Analytics**: Fans and analysts gain insights into factors like run rate impact.
- **Betting/Fantasy Sports**: Real-time odds based on model probabilities.
- **Broadcasting**: Enhances viewer engagement with predictive overlays.

## Installation

### Prerequisites
- Python 3.8+
- Git for cloning.
- Libraries: pandas, scikit-learn, flask, numpy, matplotlib (for EDA).
- Dataset: Download ODI JSON files from [cricsheet.org](https://cricsheet.org/) (place in `data/odis_male_json/`).


## Usage

1. Launch the Flask app.
2. On the main page: Enter match details (e.g., teams, current state).
3. Submit → View prediction (e.g., "98% chance of win") with probability breakdown.
4. Explore EDA: Run notebook for charts (e.g., top batsmen averages).
5. Predict via code: Load model, input DataFrame, call `predict_proba()`.

### Example Prediction
Input: India vs South Africa in Durban, 165 runs left, 120 balls, 1 wicket, CRR=4.5, RRR=8.25, Target=300.  
Output: [[0.02, 0.98]] (98% win probability for batting team).

For ambiguous/overfit issues (notebook shows 99.97% acc, PPT 91.60%), use stratified splits or cross-validation.

## Demo

Local demo at `http://localhost:5000`. <img width="945" height="474" alt="image" src="https://github.com/user-attachments/assets/686f2865-8f60-41b9-a0d4-d50519bd24b6" />

<img width="945" height="482" alt="image" src="https://github.com/user-attachments/assets/91781e40-75b5-4f81-a6a2-683ad89f7bf7" />

<img width="945" height="477" alt="image" src="https://github.com/user-attachments/assets/7e478866-8ec3-466a-ab58-fd121caa348e" />




## Performance and Results

- **Dataset**: ~380k rows post-processing (match + ball-by-ball).
- **Metrics Comparison**:
  | Model            | Accuracy | Precision | Recall | F1 Score | ROC AUC |
  |------------------|----------|-----------|--------|----------|---------|
  | Linear Regression| 84.69%  | 0.33     | 0.33  | 0.34    | 0.85   |
  | Random Forest   | 91.60%  | 0.92     | 0.90  | 0.91    | 0.9786 |
  | XGBoost         | 89.07%  | 0.89     | 0.89  | 0.89    | N/A    |

- **Key Insights**:
  - Random Forest excels due to non-linear handling and feature importance (e.g., required_run_rate most influential).
  - Confusion Matrix (RF): 93.1% TN, 90.5% TP; low FP/FN.
  - ROC: Steep curve indicates strong separation.

### Model Justifications
- **Random Forest**: Best for heterogeneous data, provides insights; robust to overfitting.
- **XGBoost**: Good for interactions but slightly lower accuracy here.
- **Linear Regression**: Baseline; assumes linearity, underperforms on complex dynamics.

<img width="540" height="473" alt="image" src="https://github.com/user-attachments/assets/42640ccb-1b81-4ad8-a144-e6a3a4b9869b" />


<img width="553" height="987" alt="image" src="https://github.com/user-attachments/assets/638ed018-9d8e-46fb-adc4-0ce146032e20" />

<img width="908" height="625" alt="image" src="https://github.com/user-attachments/assets/ad709946-5d77-49ae-b897-129bd65a1a35" />


![Uploading image.png…]()



## Future Work

- **Improvements**: Add real-time data feeds, weather integration, multi-class (e.g., draw predictions).
- **Enhancements**: Cloud deployment (Heroku/AWS), mobile app, RL for continuous learning.
- **Extensions**: T20/IPL support, player-specific predictions.
- **Challenges Addressed**: Data quality (merging sources), interpretability (SHAP for explanations).


## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Fork the repo, create a branch, submit PRs. Issues welcome on GitHub.

## Acknowledgments

- Dataset: cricsheet.org.
- Tools: scikit-learn, Flask, Pandas.
- Inspiration: Sports analytics research on ODI predictions.

Thank you! For queries, open an issue.
