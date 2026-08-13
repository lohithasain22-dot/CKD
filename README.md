# Chronic Kidney Disease (CKD) Detection using Ensemble Learning & CNN

Two complementary machine learning approaches for detecting Chronic Kidney Disease (CKD):

1. **Ensemble Learning on Tabular Data** — Bagging and Random Subspace ensembles built on KNN, Naïve Bayes, and Decision Tree base learners, applied to a 25-attribute clinical CKD dataset.
2. **CNN Image Classification** — A convolutional neural network trained on kidney CT/ultrasound images to classify disease type across 5 classes.

Originally developed as a B.Tech mini project, *"Development of Chronic Kidney Disease Using Ensemble Approach"*, Department of Computer Science & Engineering, DRK College of Engineering and Technology (JNTU Hyderabad).

## Motivation

CKD is a major global health burden — a 2017 international survey attributed roughly 1.2 million deaths worldwide to CKD, and up to 96% of people with reduced kidney function are unaware of it. Early, accurate diagnosis significantly improves outcomes and reduces reliance on late-stage interventions like dialysis or transplantation. This project explores whether ensemble learning techniques and deep learning can improve diagnostic accuracy over individual classical classifiers, using both structured clinical data and medical imaging.

## Repository Contents

| File | Description |
|---|---|
| `Development_of_ensemble_approach_to_CKD_befinal.ipynb` | Data preprocessing, PSO/PCA-based feature analysis, and ensemble classification (KNN, Naïve Bayes, Decision Tree, Random Forest, Bagging, Random Subspace) on the tabular CKD dataset. |
| `Lohitha_kidney_disease_classifier_with_99_cnn_executed.ipynb` | CNN-based image classifier trained on kidney scan images across 5 disease classes. |
| `final_Documentation_Batch_6.docx` | Full project report: abstract, SRS, methodology, UML diagrams (ER, use case, class, sequence), results, and discussion. |

## Approach 1 — Ensemble Learning on Tabular Data

**Dataset:** CSV dataset of 400 patient instances with 25 attributes (11 numeric, 14 nominal), sourced from Kaggle/UCI, labeled CKD / non-CKD.

**Pipeline:**
1. **Data cleaning** — map categorical text fields (`htn`, `dm`, `cad`, `pe`, `ane`, `rbc`, `pc`, `pcc`, `ba`, `appet`) to binary/numeric values; drop the `id` column.
2. **Missing value handling** — rows with NaNs dropped for primary model training; a secondary evaluation fills missing values with 0 to test robustness on noisier data.
3. **Feature analysis:**
   - **Particle Swarm Optimization (PSO)** via `pyswarms`, used to explore optimal feature/weight configurations.
   - **Principal Component Analysis (PCA)** to visualize the dataset in reduced dimensions and inspect class separability.
   - Correlation heatmaps to examine inter-feature relationships.
4. **Model selection** — `RandomForestClassifier` tuned via `GridSearchCV` (10-fold cross-validation) over `n_estimators`, `max_depth`, and `class_weight`.
5. **Base classifiers** — K-Nearest Neighbors (Minkowski distance), Naïve Bayes, and Decision Tree.
6. **Ensemble methods:**
   - **Bagging** — majority-vote aggregation of base learners trained on bootstrap samples.
   - **Random Subspace** — base learners trained on random feature subsets (attribute bagging).

**Results (from the project report):**
- Individual base classifiers: **KNN achieved the highest individual accuracy at 88%**.
- **Random Subspace + KNN** achieved the best overall ensemble accuracy of **88%**, outperforming Bagging with the same base learner.
- The tuned Random Forest model (evaluated separately in the notebook) reached **88.8% accuracy** with a **99.2% ROC AUC** on the clean (NaN-dropped) test set.

**Key libraries:** `pandas`, `numpy`, `scikit-learn`, `pyswarms`, `seaborn`, `matplotlib`

## Approach 2 — CNN Image Classification

**Dataset:** Grayscale kidney scan images organized into 5 class folders, split 90% / 5% / 5% into train / validation / test sets using `split-folders` (11,215 training images, 627 validation images, 621 test images).

**Model architecture:** Sequential CNN built with Keras —
- 6 `Conv2D` + `MaxPool2D` blocks with increasing filter depth (32 → 32 → 64 → 64 → 128 → 128), `relu` activations
- Flatten + Dense output layer with `softmax` (5-class categorical classification)
- Input images resized to 200×200, single-channel (grayscale)

**Training configuration:**
- Optimizer: `rmsprop`
- Loss: `categorical_crossentropy`
- Metrics tracked: accuracy, precision, recall
- Epochs: 25

**Results:**
- Training accuracy converged to **~99%** within the first few epochs, with validation accuracy reaching **~99.8%**.
- Final test set evaluation: **Accuracy, Precision, Recall, and F1-score all at 1.00 (100%)**, with test loss effectively 0.

**Key libraries:** `keras` / `tensorflow`, `split-folders`, `scikit-learn` (metrics), `seaborn`, `matplotlib`, `scikit-image`

## Software & Hardware Requirements (per project report)

- **OS:** Windows 10 (or any OS with Python 3.6+; notebooks were run on Google Colab)
- **Python:** 3.6+
- **Tools:** Anaconda, Jupyter Notebook, Google Colab (for GPU-backed CNN training and Drive-mounted datasets)
- **Recommended hardware:** Intel Core i5-class processor, 8GB RAM minimum (GPU strongly recommended for the CNN notebook)

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn pyswarms split-folders scikit-image keras tensorflow
   ```

3. **Datasets**
   - The tabular notebook expects a `kidney_disease.csv` file (CKD dataset, UCI/Kaggle) — update the file path in the notebook to point to your local copy.
   - The CNN notebook expects a directory of kidney scan images organized by class — update the dataset path accordingly before running `splitfolders.ratio(...)`.
   - Both notebooks were originally run in Google Colab with datasets stored on Google Drive; the `drive.mount(...)` cells and hardcoded `/content/drive/...` paths should be replaced with local paths if running outside Colab.

4. **Run the notebooks**
   ```bash
   jupyter notebook
   ```
   Open either notebook and run all cells top to bottom.

## Notes on Reuse

- File paths in both notebooks are hardcoded to a specific Google Drive structure (`/content/drive/MyDrive/...`) — update these before running in a different environment.
- The tabular dataset used here is small (400 instances); results should be validated against a larger or more recent CKD dataset before any clinical use.
- Both notebooks report very high accuracy (88% for the ensemble methods, ~99–100% for the CNN); the CNN's near-perfect scores warrant a sanity check for data leakage between train/validation/test splits before drawing strong conclusions, especially given the relatively small validation/test sizes.

