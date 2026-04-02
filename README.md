# 💳 Customer Segmentation Using Credit Card Clustering

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?style=flat&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=flat&logo=plotly&logoColor=white)

---

## 📌 Overview

This notebook implements an **end-to-end unsupervised machine learning pipeline** to segment credit card customers based on their behavioral patterns. The goal is not prediction accuracy — it is **discovery**: identifying meaningful customer groups that can inform business decisions around credit risk, marketing strategy, and customer retention.

The project applies **K-Means** and **Agglomerative (Hierarchical) Clustering** on a real-world credit card dataset, with **PCA** used for dimensionality reduction and visualization.

---

## 🎯 Goals & Objectives

| Goal | Details |
|---|---|
| **Primary** | Segment ~9,000 active credit card customers into behaviorally distinct groups |
| **Analytical** | Understand how preprocessing choices (scaling, transformation, PCA) impact clustering quality |
| **Business** | Derive actionable customer profiles: risk levels, spending styles, engagement tiers |
| **Educational** | Demonstrate the full unsupervised ML workflow — from raw data to business interpretation |

By the end of the notebook, you will understand:

- How clustering differs fundamentally from supervised learning
- Why preprocessing decisions (especially skewness correction and scaling) matter for distance-based algorithms
- How to evaluate clusters using inertia, the elbow method, and silhouette scores
- How to interpret cluster outputs from a **data science and business perspective**

---

## 📊 Dataset

### Source
**CC GENERAL.csv** — a widely-used public dataset of credit card customer behavior.  
Available on [Kaggle: Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).

### Shape
- **Rows:** ~8,950 customers (after dropping 1 null row)
- **Columns:** 18 features (17 behavioral + 1 ID column)

### Key Features

| Feature | Description |
|---|---|
| `CUST_ID` | Customer identifier (dropped before modeling) |
| `BALANCE` | Current balance on the account |
| `PURCHASES` | Total purchases made |
| `ONEOFF_PURCHASES` | Maximum single purchase amount |
| `INSTALLMENTS_PURCHASES` | Purchases made in installments |
| `CASH_ADVANCE` | Cash withdrawn against the card |
| `PURCHASES_FREQUENCY` | How often purchases are made (0–1) |
| `CASH_ADVANCE_FREQUENCY` | How often cash advances occur (0–1) |
| `CREDIT_LIMIT` | Customer's credit ceiling |
| `PAYMENTS` | Total payments made |
| `MINIMUM_PAYMENTS` | Minimum payments made (has missing values) |
| `PRC_FULL_PAYMENT` | % of months where full balance was paid |
| `TENURE` | Months of customer relationship |

### Data Quality Notes
- **Missing values:** 1 in `CREDIT_LIMIT` (dropped); 313 in `MINIMUM_PAYMENTS` (~3.5%) → median-imputed
- **No duplicates** detected
- **All features** are heavily right-skewed with significant outliers
- **High correlations** exist between `PURCHASES`, `ONEOFF_PURCHASES`, and `INSTALLMENTS_PURCHASES`

---

## 🔬 Methods

### Pipeline Summary

```
Raw CSV  →  Drop null CREDIT_LIMIT  →  Median Impute MINIMUM_PAYMENTS
         →  Power Transform (Yeo-Johnson)  →  PCA (9 components, ~96.5% variance)
         →  K-Means Clustering (k=5)
```

| Step | Tool/Method | Rationale |
|---|---|---|
| Missing value handling | `SimpleImputer(median)` | Robust to skewed distributions |
| Skewness correction | `PowerTransformer` (Yeo-Johnson) | All features are highly right-skewed |
| Standardization | Built into `PowerTransformer` | Ensures equal feature contribution |
| Dimensionality reduction | `PCA(n_components=9)` | Captures 96.5% of variance; aids visualization |
| Cluster number selection | Elbow Method + Silhouette Score | Silhouette score preferred due to unclear elbow |
| Primary clustering | `KMeans(k=5, init='k-means++')` | Best silhouette score at k=5 |
| Validation clustering | `AgglomerativeClustering(ward)` | Cross-validates K-Means results |
| Evaluation metric | Silhouette Score | Measures cohesion vs. separation |
| Visualization | `plotly` 3D scatter, 2D scatter | Interactive cluster exploration in PCA space |

---

## 💡 Key Findings

### The 5 Customer Segments

| Cluster | Label | Risk Level | Profile |
|---|---|---|---|
| **0** | 🔴 Financially Stressed | **High** | Rarely purchases; relies on cash advances; carries significant debt; almost never pays in full |
| **1** | 🟢 Premium Active Transactors | **Low** | Very frequent purchases; prefers card over cash; high credit limit; moderate full payment rate |
| **2** | 🟢 Disciplined Low-Balance Users | **Low** | Regular installment purchases; very low balances; highest full payment percentage; responsible spenders |
| **3** | 🟡 Passive Credit Users | **Medium** | Infrequent users; moderate balance; low engagement; sometimes carries balance |
| **4** | 🔴 High-Exposure Debt Carriers | **Very High** | Heavy mixed usage (purchases + installments + cash advance); very large balances; rarely pays in full; high interest revenue for the bank |

### Business Implications
- **Clusters 0 & 4** represent credit risk and should trigger early intervention or limit reviews
- **Cluster 1** represents VIP customers — ideal targets for premium loyalty rewards
- **Cluster 2** are reliable, low-risk users — good candidates for limit upgrades
- **Cluster 3** may benefit from re-engagement offers to increase card usage

---

## 🗂️ Notebook Structure

| Section | Cells | Description |
|---|---|---|
| **1. Import Libraries** | 4 | All dependencies imported |
| **2. Load Data** | 5–7 | Read CSV, inspect shape and first rows |
| **3. Data Exploration** | 8–30 | Data types, missing values, duplicates, cardinality, distributions, outliers, correlation heatmap |
| **4. Data Pipeline** | 31–48 | Preprocessing pipeline (imputation + power transform), PCA variance analysis, full pipeline assembly |
| **5. Experimentation** | 49–66 | Elbow method, silhouette scores, K-Means fitting (k=5), label assignment, 2D & 3D PCA visualizations |
| **6. Interpretation** | 67–75 | Cluster profiling by key features, business label assignment |
| **7. Hierarchical Clustering** | 76–83 | Dendrogram (200-sample), Agglomerative Clustering evaluation, silhouette comparison |

---

## ⚙️ Setup & Requirements

### Prerequisites
- Python **3.8+**
- Jupyter Notebook or JupyterLab

### Dependencies

| Library | Version | Purpose |
|---|---|---|
| ![pandas](https://img.shields.io/badge/-pandas-150458?logo=pandas&logoColor=white) `pandas` | ≥ 1.3 | Data loading and manipulation |
| ![numpy](https://img.shields.io/badge/-numpy-013243?logo=numpy&logoColor=white) `numpy` | ≥ 1.21 | Numerical operations |
| ![matplotlib](https://img.shields.io/badge/-matplotlib-11557C?logo=python&logoColor=white) `matplotlib` | ≥ 3.4 | Static plotting |
| ![seaborn](https://img.shields.io/badge/-seaborn-4C72B0?logo=python&logoColor=white) `seaborn` | ≥ 0.11 | Statistical visualizations |
| ![plotly](https://img.shields.io/badge/-plotly-3F4F75?logo=plotly&logoColor=white) `plotly` | ≥ 5.0 | Interactive 2D/3D scatter plots |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white) `scikit-learn` | ≥ 1.0 | ML pipeline, clustering, PCA, metrics |
| ![scipy](https://img.shields.io/badge/-scipy-8CAAE6?logo=scipy&logoColor=white) `scipy` | ≥ 1.7 | Hierarchical clustering / dendrogram |

### Installation

**1. Clone or download the project files:**
```bash
git clone https://github.com/your-username/credit-card-clustering.git
cd credit-card-clustering
```

**2. Create and activate a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install all dependencies:**
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy jupyter
```

Or using a `requirements.txt`:
```bash
pip install -r requirements.txt
```

**4. Download the dataset:**

Download `CC GENERAL.csv` from [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) and place it in the same directory as the notebook.

**5. Launch Jupyter:**
```bash
jupyter notebook Credit_Card_Clustering.ipynb
```

---

## ▶️ Usage Instructions

### Running the Full Pipeline

1. Open the notebook in Jupyter and select **Kernel → Restart & Run All** to execute all 83 cells in sequence.
2. Ensure `CC GENERAL.csv` is in the working directory before running Cell 6 (`df = pd.read_csv('CC GENERAL.csv')`).

### Step-by-Step Execution Guide

| Cells | Action | Expected Output |
|---|---|---|
| 4 | Run imports | No errors |
| 6–7 | Load data | DataFrame head + shape `(8950, 18)` |
| 9–27 | EDA | Distribution histograms, boxplots, correlation heatmap |
| 34–48 | Build pipeline | PCA cumulative variance curve; choose 9 PCs |
| 51–55 | Tune k | Elbow + silhouette plots; set `n_clusters=5` |
| 57–66 | Fit & visualize | 3D and 2D interactive Plotly scatter plots |
| 69–74 | Interpret | Cluster mean feature table, written profiles |
| 78–83 | Hierarchical | Dendrogram + silhouette score comparison |

### Reproducing Results

To ensure reproducibility, all stochastic steps use `random_state=42`:
```python
PCA(n_components=9, random_state=42)
KMeans(n_clusters=5, init='k-means++', random_state=42)
```

To experiment with a different number of clusters:
```python
full_pipeline.set_params(kmeans__n_clusters=4)  # Try 3, 4, 6, ...
full_pipeline.fit(df)
```

### Generated Outputs

| Output | Cell(s) | Description |
|---|---|---|
| Distribution histograms | 21 | Feature-level distribution overview |
| Boxplots (outliers) | 24 | Per-feature outlier visualization |
| Correlation heatmap | 27 | Feature inter-correlation via Seaborn |
| Cumulative PCA variance curve | 44 | Select number of principal components |
| Elbow curve | 52 | Inertia vs. number of clusters |
| Silhouette score chart | 54 | Cluster quality vs. k |
| 3D Plotly scatter | 65 | Interactive 3-PC cluster visualization |
| 2D Plotly scatter | 66 | Interactive 2-PC cluster overview |
| Cluster mean table | 71 | Key feature means per cluster |
| Dendrogram | 78 | Hierarchical clustering tree (200 samples) |

---

## ⚠️ Limitations

1. **No ground-truth labels** — clustering quality is inherently subjective; silhouette scores guide but do not define the "correct" solution.
2. **Ambiguous elbow** — the elbow method did not yield a clean knee, making cluster-count selection partially subjective.
3. **PCA loses interpretability** — while PCA improves clustering, the principal components are linear combinations of all features and cannot be directly interpreted.
4. **K-Means assumes spherical clusters** — this may distort results for non-globular natural groupings in the data.
5. **Static snapshot** — the dataset is a single-point-in-time view; customer behavior evolves over time.
6. **Missing data imputation** — median imputation for `MINIMUM_PAYMENTS` (~3.5% missing) may slightly bias cluster assignments.
7. **Dendrogram sampled** — only 200 random samples are used for the dendrogram due to computational constraints; full-data linkage is expensive.

---

## 🚀 Potential Extensions & Future Work

- **DBSCAN / HDBSCAN:** Explore density-based clustering to handle outliers natively without outlier-sensitive preprocessing.
- **Gaussian Mixture Models (GMM):** Use soft probabilistic assignments instead of hard cluster membership.
- **t-SNE / UMAP:** Compare non-linear dimensionality reduction for better cluster separation in 2D.
- **Feature engineering:** Derive ratio features (e.g., `CASH_ADVANCE / CREDIT_LIMIT`, `PURCHASES / CREDIT_LIMIT`) before clustering.
- **Cluster stability analysis:** Use bootstrapping to test how stable the 5-cluster solution is across resamples.
- **Time-series extension:** If longitudinal data is available, apply sequence clustering or temporal profiling.
- **Model deployment:** Wrap the trained `full_pipeline` with `pickle` or `joblib` and serve predictions via a REST API (FastAPI/Flask).
- **Dashboard:** Build an interactive segment explorer with Streamlit or Dash for business stakeholders.


---

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core language |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Interactive notebook environment |
| ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) | Data wrangling |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computing |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | ML pipeline, clustering, PCA, metrics |
| ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white) | Hierarchical clustering |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white) | Static charts |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logo=python&logoColor=white) | Statistical plots |
| ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) | Interactive 2D/3D visualizations |


