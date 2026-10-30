# Breast cancer classifier comparison

This is an early machine learning practice project that I am revisiting with a more careful evaluation setup. It compares several classifiers on the scikit-learn breast cancer dataset (569 tumour samples, 30 numeric features). The current revision makes preprocessing and model selection explicit so the reported results are easier to interpret and reproduce.

It is an educational benchmark. The results say nothing about diagnosing real patients.

## History

The project started in January 2025 as a learning exercise (`legacy/model_training.py`, unchanged apart from a header comment). The October 2026 revision fixes the following problems in that script:

- **PCA saw the test data.** PCA was fitted on all 569 samples before the train/test split, so the test rows shaped the features used for training. It also ran on unscaled features, where `worst area` (variance about 300,000) dominates.
- **Scaling sat outside the hyperparameter search.** The scaler was fitted on the whole training set before `GridSearchCV` split it into folds, so each validation fold was scaled using its own values.
- **The winner was chosen on the test set.** All seven models were scored on the test set and the best test F1 was reported as the result. That number is optimistic because the test set was used to pick it.
- **The class labels were inverted.** scikit-learn encodes `0 = malignant` and `1 = benign`. The old confusion matrix labelled class 0 "No Cancer", precision and recall were computed for the benign class, and the old README described the encoding the wrong way round.
- **Runs were not repeatable.** Random forest, decision tree and the neural network had no fixed seed, so two runs gave different scores.

## How the evaluation works now

`evaluate.py` runs these steps in order:

1. Split the raw samples into a development set (426) and a held-out test set (143). The split is stratified with seed 42, and the indices are saved in `results/split.json`.
2. Build every model as one scikit-learn `Pipeline`: scaler → optional PCA → classifier. The scaler type (standard or min-max), PCA (off, 5 or 10 components) and the classifier settings are tuned together with `GridSearchCV` on 5-fold stratified CV inside the development set.
3. Compare the seven tuned models with nested cross-validation on the development set (outer: 5 folds × 3 repeats). Two baselines go through the same folds: always predicting the majority class (benign), and an untuned scaled logistic regression.
4. Select the model with the highest mean outer-CV F1 for the malignant class and re-tune it on the full development set. Write the choice to `results/protocol.json` before the test set is used.
5. Score the test set once, for the selected model and the two baselines.

**Positive class.** Malignant is the positive class throughout: the script re-encodes the target as `1 = malignant`, and class names come from the dataset. F1 for the malignant class is the selection metric because it balances missed malignant cases (recall) against false alarms (precision). Every output also reports recall (sensitivity), precision, specificity, balanced accuracy and accuracy.

**Leakage checks.** `tests/test_no_leakage.py` places a recording step at the front of the pipeline. It checks that no fit ever receives a test row, and that the cross-validation fits inside the search only see their training folds. It also checks the label encoding, the split, and that the final scaler's statistics come from the development data only.

## Results

Measured on Linux with Python 3.13.5 and scikit-learn 1.9.1. Two consecutive runs produced byte-identical output files.

**Model comparison** (development set, nested CV, mean ± standard deviation over 15 outer folds, malignant class positive):

| Model | F1 | Recall | Precision |
|---|---|---|---|
| Neural network (MLP), tuned | 0.961 ± 0.031 | 0.954 ± 0.050 | 0.970 ± 0.037 |
| Baseline: logistic regression, untuned | 0.959 ± 0.029 | 0.950 ± 0.051 | 0.972 ± 0.036 |
| Logistic regression, tuned | 0.953 ± 0.034 | 0.939 ± 0.057 | 0.970 ± 0.040 |
| SVM, tuned | 0.948 ± 0.028 | 0.935 ± 0.047 | 0.963 ± 0.038 |
| k-nearest neighbours, tuned | 0.947 ± 0.024 | 0.926 ± 0.037 | 0.971 ± 0.034 |
| Random forest, tuned | 0.933 ± 0.039 | 0.926 ± 0.049 | 0.941 ± 0.055 |
| Decision tree, tuned | 0.906 ± 0.031 | 0.886 ± 0.062 | 0.930 ± 0.039 |
| Gaussian naive Bayes | 0.901 ± 0.037 | 0.880 ± 0.049 | 0.925 ± 0.049 |
| Baseline: majority class | 0.000 | 0.000 | 0.000 |

The MLP was selected, with standard scaling, PCA to 10 components, one hidden layer of 50 units and alpha 0.0001. Its lead over the untuned logistic regression (0.961 vs 0.959) is much smaller than the fold-to-fold variation, so it is best read as a tie, not a real advantage.

**Held-out test set** (143 samples: 53 malignant, 90 benign; scored once):

| Model | F1 | Recall (sensitivity) | Precision | Specificity | Accuracy | Missed malignant |
|---|---|---|---|---|---|---|
| Selected MLP | 0.951 | 0.906 | 1.000 | 1.000 | 0.965 | 5 of 53 |
| Untuned logistic regression | 0.951 | 0.925 | 0.980 | 0.989 | 0.965 | 4 of 53 |
| Majority class | 0.000 | 0.000 | – | 1.000 | 0.629 | 53 of 53 |

The 95% bootstrap interval (2,000 resamples of the test set) for the selected model's F1 is 0.90 to 0.99, and for its recall 0.82 to 0.98. The confusion matrix is in `results/confusion_matrix_test.png`.

**PCA on or off** (development set only, run after the model was chosen and not used to choose it):

| Model | No PCA | PCA 5 | PCA 10 |
|---|---|---|---|
| Logistic regression | 0.961 ± 0.028 | 0.950 ± 0.032 | 0.957 ± 0.025 |
| MLP | 0.964 ± 0.030 | 0.950 ± 0.031 | 0.962 ± 0.026 |

PCA did not help here. Reducing to 5 components was slightly worse, and none of the differences is larger than the fold variation.

## What this shows and what it does not

The tuned models do not beat a plain scaled logistic regression in any meaningful way on this dataset. On the test set, the selected model's five errors are all malignant tumours predicted as benign. These are the costly errors, and F1 does not weight them more heavily than false alarms. A study that cared about missed cases would choose a recall-oriented metric or a decision threshold before looking at the test set.

Limits:
- 143 test samples is small. One more or one fewer mistake moves recall by about 0.02.
- Only one split seed was used for the test set. Another split would give somewhat different numbers.
- The hyperparameter grids are small and were chosen by hand.
- The dataset is a well-known, cleaned benchmark. Results on it do not transfer to clinical data.

## How to run

Tested with Python 3.13.5 on Linux.

```bash
git clone https://github.com/erfanpj/Breast_Cancer_Project.git
cd Breast_Cancer_Project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests      # leakage and label checks
python evaluate.py             # about 3 minutes on 8 CPU cores; writes results/
```

The run should reproduce the files in `results/`. The only expected warnings come from the majority-class baseline, whose precision is undefined because it never predicts malignant.

## Files

- `evaluate.py`: the corrected evaluation
- `tests/test_no_leakage.py`: checks for split boundaries, label encoding and preprocessing leakage
- `results/`: saved split, nested-CV scores, frozen protocol, test metrics and confusion matrix
- `legacy/model_training.py`: the original 2025 script, kept for reference. Do not use its printed scores.

## Attribution

Dataset: Breast Cancer Wisconsin (Diagnostic), as shipped with scikit-learn (`sklearn.datasets.load_breast_cancer`), originally from the UCI Machine Learning Repository. I wrote the original script in 2025. The 2026 evaluation correction was implemented with help from an AI coding assistant. The numbers above come from the saved runs in `results/`.
