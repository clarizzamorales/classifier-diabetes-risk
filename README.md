
# Diabetes Risk Classifier using Decision Tree (Entropy-Based)

This project is a machine learning classifier built to predict whether a person is likely to have diabetes based on behavioral and clinical indicators from the 2015 BRFSS dataset. It uses a Decision Tree Classifier with entropy as the splitting criterion and includes entropy analysis for label distribution.

---

## Project Highlights

- **Model Type**: Decision Tree Classifier (using `entropy` criterion)
- **Data Source**: Behavioral Risk Factor Surveillance System (BRFSS) 2015
- **Evaluation Metric**: Average accuracy over 30 runs
- **Entropy Analysis**: Uses normalized entropy to understand label distribution
- **Visualization**: Exports the trained decision tree as a `.pdf` using Graphviz

---

## What We’ll Learn

- How to preprocess healthcare data
- How to apply entropy calculations to evaluate class balance
- How to build and train a decision tree using `scikit-learn`
- How to visualize the tree with Graphviz
- How to evaluate model performance through repeated train/test splits

---

## Files Included

| File | Description |
|------|-------------|
| `diabetes_risk_classifier.py` | Final script to run the full training + visualization pipeline |
| `FinalProject2.ipynb` | Notebook version for step-by-step walkthrough and visualization |
| `diabetes_binary_health_indicators_BRFSS2015.csv` | Dataset used for training the model |
| `decision_tree.pdf` | Visual representation of the trained decision tree |
| `requirements.txt` | List of Python dependencies to install before running the code |
| `README.md` | This documentation file |

---

## How to Run

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/diabetes-risk-classifier.git
   cd diabetes-risk-classifier
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the script**:
   ```bash
   python diabetes_risk_classifier.py
   ```

5. A `decision_tree.pdf` file will be generated with the trained tree.

---

## Example Output

```
Normalized entropy value: 0.679
Average accuracy over 30 runs: 0.768
```

---

## Notes

- The model removes several features to focus on a small subset of predictors.
- Graphviz must be installed on your system to export the tree visualization. Use `brew install graphviz` on macOS or `apt-get install graphviz` on Ubuntu.
- This project was originally developed as part of the Discover AI program.

---

## Author

Clarizza Morales  
Python · AI/ML · Healthcare Data Enthusiast

## Data Source

This project uses the **[Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)** provided by Alex Teboul on Kaggle.  
It includes health-related attributes derived from the 2015 BRFSS survey to help classify diabetes status in adults.
---

## License

This project is open-source and free to use.
