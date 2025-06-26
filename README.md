

### 🔧 Project: **Traffic Signal Forecasting Using Probabilistic Decomposition Transformer**

**Description:**
Designed and implemented an intelligent traffic forecasting system that leverages a **Probabilistic Decomposition Transformer (PDT)** architecture to predict vehicle flow across multiple city junctions. The model captures **trend**, **seasonality**, and **residual components** of traffic patterns, enabling highly interpretable and data-driven traffic management.

**Key Contributions:**

* ✅ **Data Preparation & Preprocessing:**

  * Utilized hourly traffic data across 4 city junctions.
  * Cleaned and normalized data using Min-Max Scaling and constructed supervised learning sequences for temporal modeling.

* ✅ **Model Development:**

  * Built a custom Transformer-based time series model using **PyTorch**.
  * Enhanced interpretability by decomposing predictions into **trend** and **seasonality**.
  * Incorporated a confidence interval (95%) around forecasts to reflect uncertainty.

* ✅ **User Interface:**

  * Developed a **Streamlit web app** allowing users to:

    * Select forecast horizon (1 day, 1 week, 1 month)
    * Choose junction number and forecast start date
    * Visualize forecasts along with ground truth, trend, and seasonal components.

* ✅ **Performance Analysis:**

  * Evaluated model performance using **MAE**, **MSE**, and **R²-score**.
  * Detected and visualized peak traffic hours using rolling average analytics.

* ✅ **Deployment Ready:**

  * Containerized model and app files with GitHub version control.
  * Documented project thoroughly with a clear roadmap for future enhancements.

**Tech Stack:**
`Python`, `PyTorch`, `Streamlit`, `Pandas`, `NumPy`, `Matplotlib`, `Scikit-learn`, `Git/GitHub`

**Future Scope:**

* Automating real-time traffic signal control per junction.
* Integrating directional vehicle flow and adaptive green light timing.
* Expanding to city-wide intelligent traffic systems using live IoT sensor data.


