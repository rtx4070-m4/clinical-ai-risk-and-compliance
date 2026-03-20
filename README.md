
# Malpractice in the Age of Algorithmic Diagnostics
## Liability Analysis of a CKD Phenotyping Model Producing False Negatives

### Overview
End-to-end, production-grade project combining Machine Learning, Explainable AI, Risk Simulation, and Legal Analysis focused on **false negatives** in CKD prediction.

### Key Features
- Multi-model pipeline: Logistic Regression, Random Forest, XGBoost, Neural Network
- Strong emphasis on **Recall (Sensitivity)** and false negative analysis
- Explainability: SHAP + LIME (global & local)
- Risk simulation engine for missed CKD cases
- Streamlit app with **low-confidence warning**
- Full legal report (tort law, standard of care, liability allocation)
- Research paper & presentation content

### Quickstart
```bash
pip install -r requirements.txt
python -m src.train
streamlit run deployment/app.py
```

### Repo Structure
See folders: `data/`, `notebooks/`, `src/`, `models/`, `legal_analysis/`, `docs/`, `presentation/`, `deployment/`.

### Results (Typical)
- Recall prioritized models reduce false negatives but may increase FP
- XGBoost/RandomForest provide best FN trade-off after threshold tuning
- SHAP highlights creatinine, age, BP as dominant drivers

### Legal Insight
False negatives create **high liability exposure** due to delayed treatment. Explainability and clinical override are essential.

### License
MIT
