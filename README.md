# User Behavior Analysis for Improved Movie Recommendations

This repository contains an end‑to‑end movie recommendation system built and evaluated on the **Netflix Prize** dataset. It compares **user‑based CF**, **item‑based CF**, and **matrix factorization (SVD)**, and demonstrates a **hybrid** approach that combines strengths of item‑based CF and SVD for better accuracy and relevance. 

> Core assets:
> - Notebook: `Recommendation_system (2).ipynb`
> - Report: `FinalReport_MovieRecommendation_P32.pdf` (project write‑up and results)
>
> Team: **P32 – Niharika Maruvanahalli Suresh, Surabhi Ravindran Nair, Siri Paidipalli**

---

## 🚀 Overview

Modern streaming platforms rely on recommender systems to personalize content, increase engagement, and reduce search friction. This project explores a **hybrid recommender** that integrates **Collaborative Filtering** (user‑ and item‑based using KNN + cosine similarity) with **Matrix Factorization (SVD)** to address **sparsity**, **cold‑start**, and **scalability** challenges on large, sparse ratings data. 

---

## 📦 Dataset

- **Source:** Netflix Prize (≈100M ratings; ~480k users; ~17k movies)
- **Signal:** Explicit ratings (1–5)
- **Challenge:** Highly sparse user–item matrix → motivates item‑based CF and SVD. 

> In the notebook, we construct a user×item matrix (subset/top‑N users & movies for tractability), normalize ratings, and store data in a **sparse matrix** for efficient similarity and factorization steps. 

---

## 🛠️ Methods

### 1) Collaborative Filtering (CF)
- **User‑based CF:** KNN (cosine) finds similar users; predicts via neighbors’ ratings.  
- **Item‑based CF:** KNN (cosine) finds similar items; predicts via item similarities.  
- **Why item‑based often wins:** item relationships are more stable than user overlaps in sparse data. 

### 2) Matrix Factorization (SVD)
- Decompose ratings matrix into **U Σ Vᵀ**, keep top‑k singular values (e.g., *k*≈200).  
- Reconstruct **Ŕ = Uₖ Σₖ Vᵀₖ** to get predicted ratings for unseen items.  
- Captures **latent user/movie factors**, robust under sparsity. 

### 3) Hybrid (Item‑CF + SVD)
- Combine predictions to reduce error and boost relevance/coverage; mitigates **cold start** and balances novelty vs. accuracy. 

---

## 📊 Results

| Model                         | RMSE    | MAE    | Notes |
|------------------------------:|:-------:|:------:|:------|
| **User‑based CF (KNN, cos)** | 4.0608  | 4.0600 | Struggles with sparsity/cold start; risk of overfitting.  |
| **Item‑based CF (KNN, cos)** | 0.7710  | 0.7709 | Stable item relations improve accuracy.  |
| **SVD (k≈200)**              | 0.7506  | 0.5932 | Best single model; robust latent factors.  |
| **Hybrid (Item‑CF + SVD)**   | **Best overall** | — | Combines strengths; lowest errors in study.  |

> Additional qualitative analysis discusses precision/recall and the hybrid model’s ability to retrieve relevant and diverse items. 

---

## 🗂️ Repository Structure (suggested)

```
.
├── Recommendation_system (2).ipynb   # Main notebook (run this)
├── FinalReport_MovieRecommendation_P32.pdf  # Full write-up & figures
└── README.md
```

---

## ▶️ How to Run

### Option A — Google Colab
1. Upload `Recommendation_system (2).ipynb`.
2. Run cells in order (enable **GPU** if available for speed).
3. Mount/Upload the processed Netflix data subset as directed in the notebook.

### Option B — Local Jupyter
1. Create a virtual environment and install dependencies:
   ```bash
   pip install numpy pandas scipy scikit-learn matplotlib tqdm notebook
   ```
   > If you also experiment with deep models, add: `pip install tensorflow`.
2. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook  # or jupyter lab
   ```
3. Provide the ratings files (or prepared subset) at the paths expected in the notebook.

---

## ✅ Implementation Details

- **Preprocessing:** load ratings & movie metadata; normalize ratings; create user×item pivot; convert to **CSR** sparse matrix. 
- **User/Item CF:** KNN with **cosine similarity** over sparse rows/cols; filter seen items before ranking. 
- **SVD:** truncate to top‑k factors; reconstruct predicted matrix; recommend top‑N unseen items per user. 
- **Evaluation:** hold‑out test; compute **RMSE** and **MAE** (and discuss precision/recall). 

---

## ⚠️ Limitations & Future Work

- **Sparsity & cold start:** hybrids help but metadata/contexts (genres, year, tags) could further boost accuracy. 
- **Bias & popularity drift:** add debiasing or diversity‑aware ranking.  
- **Real‑time updates:** explore online learning / reinforcement learning for streaming feedback. 
- **Representation learning:** try **autoencoders** for latent features and **explainability** tooling to build user trust. 

---

## 📚 References

See the project report for the full bibliography and background on the Netflix Prize and matrix factorization literature.

---

## 🧑‍🤝‍🧑 Authors

**P32** — Niharika Maruvanahalli Suresh · Surabhi Ravindran Nair · Siri Paidipalli
