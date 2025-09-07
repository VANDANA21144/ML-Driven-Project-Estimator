# ML-Driven Project Estimator

This project explores machine learning models for ** ML-Driven Project Estimator**, 
focusing on cost and effort prediction using Random Forest and Analogous Estimation approaches.

## Project Overview
- Estimates project cost and required team size based on historical project data.
- Implements **Random Forest Classifier** for prediction.
- Includes **Analogous Estimation** method for cost calculation.
- Visualizes results with distribution plots and correlation matrices.
- Achieves around **90% accuracy** in prediction (based on report).

## Repository Contents
- `ML-Driven-Project-Estimator.py` → Python script extracted from project report.
- `ML-Driven-Project-Estimator.ipynb` → Jupyter Notebook version of the project.
- `data.csv` → Sample dataset for testing.
- `Mini_Project_Report.pdf` → Original project report.
- `README.md` → Project documentation.

## Tech Stack
- Python 3.6+
- pandas, numpy
- matplotlib
- scikit-learn

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ML-Driven-Project-Estimator.git
   cd ML-Driven-Project-Estimator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Python script:
   ```bash
   python ML-Driven-Project-Estimator.py
   ```

4. Or open the Jupyter Notebook:
   ```bash
   jupyter notebook ML-Driven-Project-Estimator.ipynb
   ```

##  Dataset Format
The dataset (`data.csv`) should contain:
- **Effort** → Project effort values
- **Team size** → Number of people working
- **LOC** → Lines of Code
- **Team selections** → Label for classification (Small, Medium, Large)

##  Future Improvements
- Implement additional ML models (SVM, XGBoost, Neural Networks).
- Integrate hyperparameter tuning for optimization.
- Expand dataset for better generalization.

---

