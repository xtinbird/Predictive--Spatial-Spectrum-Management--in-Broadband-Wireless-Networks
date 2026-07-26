# Predictive--Spatial-Spectrum-Management--in-Broadband-Wireless-Networks
This is for UC Berkeley Professional Certificate in Machine Learning and Artificial intelligence: Required Capstone Assignment 20.1: Initial Report and Exploratory Data Analysis (EDA)

## Research Question
How can a machine learning pipeline that forecasts user mobility patterns and classifies spatial proximity enable proactive beam alignment and orthogonal spectrum allocation in dense urban broadband wireless networks to minimize co-channel interference and maximize throughput?

## Project Overview
As mobile users move through dense urban broadband wireless networks, directional antenna beams must dynamically track them to maintain high throughput and prevent co-channel interference. This project develops a machine learning pipeline that forecasts user mobility and signal characteristics, predicts optimal beam indices, and classifies high-interference-risk users so that the network can proactively allocate spectrum and adjust beams before performance degrades.

## Data Sources
I primarily generate synthetic telemetry data using Python (NumPy / Pandas) that simulates realistic user mobility, multi-cell interference, RSRP, RSRQ, SINR, UE coordinates, velocity, Doppler shift, beam indices, and resource block utilization.

For external validation, I use the public “Mobility Dataset from a 7.2 O-RAN deployment” available on Mendeley (real base-station measurements including RSRP, RSRQ, and SINR).

## Techniques Used
- **Time-series forecasting**: ARIMA models to predict future signal metrics (RSRP, RSRQ, SINR) and mobility-related features  
- **Regression**: Multiple Linear Regression to predict the optimal antenna beam index  
- **Classification**: K-Nearest Neighbors (KNN), Logistic Regression, Decision Tree, and Random Forest to identify high-interference-risk users

## Repository Structure
- `01_EDA_and_Baseline_Model.ipynb` → Exploratory Data Analysis + Baseline Models (Module 20)
- Additional notebooks will be added in later modules (ARIMA, model comparison, etc.)

## Results (Module 20 – EDA & Baseline Models)

### Dataset
Synthetic O-RAN style telemetry was generated with 20 UEs and 500 timesteps per UE.  
Key features include location (x, y), speed, heading, Doppler shift, SSB beam index, RSRP, RSRQ, SINR, and PRB utilization.

The real-world “Mobility Dataset from a 7.2 O-RAN deployment” (Mendeley) was also cleaned and prepared for external validation of the interference classification model.

### Exploratory Data Analysis Highlights
- SINR, RSRP, and PRB utilization show the strongest relationship with high-interference risk.
- Clear spatial patterns exist, with lower SINR near cell edges.
- Doppler shift and heading provide useful information about user mobility.

### Baseline Models
This project involves two complementary prediction tasks, so two baseline models were developed:

1. **Classification Task – High Interference Risk Detection**  
   - **Goal:** Identify users who are at high risk of co-channel interference.  
   - **Model:** Random Forest Classifier  
   - **Primary Metric:** F1-Score (chosen because of class imbalance and the higher cost of missing high-risk users)

2. **Regression Task – Next Beam Prediction**  
   - **Goal:** Predict the optimal future beam index so the network can proactively steer the antenna.  
   - **Model:** Multiple Linear Regression  
   - **Metrics:** MAE, RMSE, and R²

Feature importance analysis from the Random Forest model confirmed that SINR, RSRP, and PRB utilization are the most influential variables for interference prediction.

### Real-World Validation
The Random Forest classifier (trained on synthetic data using only common features: RSRP, RSRQ, and SINR) was evaluated on the cleaned real O-RAN dataset. This provides an initial measure of how well the interference detection model generalizes to real network measurements.

### Next Steps
In upcoming modules I will:
- Implement ARIMA forecasting
- Add KNN, Logistic Regression, and Decision Tree models
- Compare all classifiers
- Expand real-world validation

## How to Run
1. Clone this repository
2. Open `01_EDA_and_Baseline_Model.ipynb`
3. Run all cells in order


