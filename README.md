# Data Mining — Applied Machine Learning Notebooks

Projects completed for the Graduate Certificate in Business Analytics at the University of South Florida (awarded May 2026), where I now continue in the MS in Artificial Intelligence in Business & Enterprise Integration. Each notebook takes a business dataset from raw file to a decision a stakeholder could act on: data preparation, modeling, evaluation, and interpretation.

**Tools:** Python · pandas · scikit-learn · mlxtend · matplotlib · seaborn

---

## 1. Loan Default Prediction — Classification & Model Selection

Predicting small-business loan default across 1,900 borrower records with nine attributes (credit score, business age, annual revenue, missed payments, industry sector, requested loan amount).

**Approach**
- Exploratory analysis: descriptive statistics, categorical distributions, correlation matrix
- Feature preparation: reference-category one-hot encoding, z-score standardization
- K-Means clustering across k = 3, 4, 5, evaluated by silhouette score (k = 4 optimal at 0.479)
- Two classifiers compared — decision tree (max depth 4, min 100 samples per leaf) and a two-layer neural network — each on a full 10-feature model and a reduced 4-feature model
- 5-fold stratified cross-validation throughout, scored on AUC, accuracy, precision, recall, specificity and F1

**Result**
The reduced decision tree was selected as the winning model: AUC 0.807, accuracy 0.758, specificity 0.797. A paired t-test on fold-wise AUC found no significant difference between the full and reduced models (p = 0.934), so the simpler model was chosen for interpretability — a requirement in credit risk, where lending decisions must be explainable to regulators and applicants.

**What the analysis showed**
Credit score and missed payments carried most of the predictive signal; adding industry sector and revenue contributed almost nothing, because several predictors were highly collinear (business age and owner experience correlated at 0.85). Fold-wise AUC ranged from 0.764 to 0.844, indicating performance was not driven by any one subset of borrowers.

**Known limitations**
The reduced neural network failed to converge — three of five folds collapsed to predicting the majority outcome, producing an AUC below 0.5. With only four inputs and a three-node first layer, early stopping halted training before the network learned. Reported as-is rather than removed. Feature scaling was also fit before cross-validation rather than inside each fold, a mild source of leakage.

---

## 2. Market Basket Analysis — Association Rules

Association rule mining on 7,126 grocery transactions to find which products are bought together, and in which direction the relationship runs.

**Approach**
- Transaction encoding into a one-hot basket matrix
- Apriori frequent itemset generation (mlxtend), with support and confidence thresholds tuned to yield a workable number of interpretable rules
- Rules examined by antecedent, by consequent, and for multi-item combinations
- Evaluation on support, confidence and lift

**What the analysis showed**
The most useful finding was directional. Reciprocal rules — ground beef predicting spaghetti, and spaghetti predicting ground beef — carry identical support and lift but very different confidence, because confidence depends on how often the antecedent itself occurs. A shopper buying ground beef is more likely to add spaghetti than the reverse. For a retailer with limited promotional space, that asymmetry decides which product to feature and which to place beside it.

Two further patterns shaped which rules were worth acting on: high lift paired with very low support flags a real association that is too rare to build a campaign around, while rules involving staples like mineral water show high confidence but lift near 1, meaning the co-occurrence is explained by the item's overall popularity rather than by any genuine purchase link.

---

## 3. Principal Component Analysis — Customer Segmentation

Dimensionality reduction on 900 retail customer records spanning age, income, spending score, loyalty years, visit frequency, average transaction value, product affinity and region.

**Approach**
- One-hot encoding of region with reference category, z-score standardization of numeric variables
- PCA across all components, with explained variance, scree plot and cumulative variance against an 80% threshold
- Component loadings extracted and sorted to interpret what each component measures
- PC1 vs PC2 scatterplots, plain and colored by region

**Result**
PC1 alone explains 69.1% of total variance; three components reach 82.2%, clearing the 80% threshold. The scree plot elbows sharply after PC1.

**What the components mean**
PC1 loads positively on income, product affinity, age, average transaction value, spending score, loyalty years and visit frequency together — a single "customer value and engagement" dimension, which is why one component absorbs so much of the variance. PC2 contrasts visit frequency against loyalty years, separating newer high-activity customers from long-tenured, lower-frequency ones. That makes PC2 a lifecycle-stage axis, and the two components together support different retention strategies for each group.

---

## Notes

Datasets are course-provided and are not redistributed here. Notebooks were run in Google Colab; saved outputs are included, so results are readable without re-running. Random seeds are fixed throughout for reproducibility.
