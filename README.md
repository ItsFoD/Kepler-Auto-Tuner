# 🚀 Space Classification with Multiple ML Models

This project applies and compares **17 different machine learning algorithms** to a classification problem using the dataset provided in `cumulative.csv`.  
It is implemented in a Jupyter Notebook (`space.ipynb`) and includes preprocessing, model training, hyperparameter optimization, evaluation, and visualization.

---

## 📂 Project Structure

- `space.ipynb` → Main Jupyter Notebook with code, experiments, and results.  
- `cumulative.csv` → Dataset used for training and testing.  
- `README.md` → Documentation (this file).  

---

## 🔎 What This Code Does

This project builds a **mini AutoML pipeline** that automatically:

1. **Runs RandomizedSearchCV**  
   - For each candidate model (Logistic Regression, SVM, Random Forest, XGBoost, etc.), it tests a wide range of hyperparameters.  
   - Hyperparameters are sampled from distributions (e.g., learning rate, tree depth, number of neighbors).  

2. **Evaluates Performance**  
   - Uses **cross-validation (CV)** to measure model robustness.  
   - Reports **ROC-AUC and accuracy** on a separate test set.  

3. **Tracks and Visualizes Results**  
   - Live plots show the performance of each model as training progresses.  
   - Tables summarize model rankings, best hyperparameters, and CV/test performance.  

4. **Explains Hyperparameter Effects**  
   - For each model, plots show how different hyperparameter values impact performance.  
   - Provides textual rationales for why the chosen hyperparameters were selected.  

5. **Selects the Best Model**  
   - The pipeline automatically identifies and saves the **top-performing model** based on ROC-AUC on the test set.  
   - Feature importances (if supported by the model) are also displayed.  


---

## 🧪 Models Compared

A total of **17 models** were evaluated:

| Model | Type | Key Idea |
|-------|------|----------|
| Logistic Regression | Linear | Learns linear decision boundary. |
| Support Vector Machine (SVC) | Linear / Nonlinear | Maximizes margin between classes. |
| K-Nearest Neighbors (KNN) | Instance-based | Classifies based on nearest neighbors. |
| Gaussian Naive Bayes | Probabilistic | Assumes Gaussian likelihoods and feature independence. |
| Linear Discriminant Analysis (LDA) | Linear | Projects data for maximum class separation. |
| Quadratic Discriminant Analysis (QDA) | Quadratic | Like LDA but allows quadratic boundaries. |
| Decision Tree | Tree-based | Splits data recursively into rules. |
| Random Forest | Ensemble (Bagging) | Many trees averaged to reduce variance. |
| Extra Trees | Ensemble (Randomized Trees) | Like Random Forest but splits are more random. |
| Gradient Boosting | Ensemble (Boosting) | Sequentially improves weak learners. |
| HistGradientBoosting | Ensemble (Boosting) | Efficient, histogram-based gradient boosting. |
| AdaBoost | Ensemble (Boosting) | Focuses on misclassified examples. |
| MLP (Neural Network) | Neural | Fully connected feedforward neural net. |
| SGD Classifier | Linear | Linear model optimized with stochastic gradient descent. |
| Calibrated SVC | Calibrated Linear/Kernel | SVC with probability calibration. |
| XGBoost | Gradient Boosting | Optimized boosting with regularization. |
| LightGBM | Gradient Boosting | Fast, leaf-wise boosting optimized for large datasets. |

---

## ⚙️ Techniques Used

- **Data Preprocessing**  
  - Handling missing values  
  - Feature scaling with `StandardScaler`  
  - Train/test split  

- **Model Training & Hyperparameter Tuning**  
  - Pipelines for preprocessing + model  
  - `RandomizedSearchCV` for optimization  
  - Distribution-based hyperparameter search (log-uniform, uniform, randint)  

- **Evaluation Metrics**  
  - ROC-AUC (main metric)  
  - Accuracy  
  - Recall  
  - Classification reports  

- **Visualization**  
  - ROC-AUC comparison plots  
  - CV vs Test performance scatter plots  
  - Hyperparameter effect plots  
  - Feature importance analysis  

---

## 📊 Results

- The notebook reports:  
  - Cross-validation (CV) scores  
  - Test ROC-AUC and accuracy  
  - Best hyperparameters for each model  
  - Ranking of all models by performance  

- The **best-performing model** is highlighted with detailed classification metrics and (if available) feature importances.  

---

## ▶️ How to Run

1. Clone this repository.  
2. Install dependencies:  

   ```bash
   pip install -r requirements.txt
