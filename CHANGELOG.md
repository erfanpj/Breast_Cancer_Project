# Changelog

## 2026-10-30: licence

- Added the MIT License for the code. The dataset remains under its own CC BY 4.0 licence (UCI) and is not stored in this repository.

## 2026-10-30: corrected evaluation

This release changes how the classifier comparison is evaluated. The original January 2025 script had several problems that made its scores optimistic or mislabelled, so its printed numbers should not be used.

Fixed:
- The train/test split now happens on the raw samples before anything is fitted. The old script fitted PCA on all samples first.
- Scaling, optional PCA and each classifier are tuned together inside one pipeline with cross-validation on the development data only. The old script fitted the scaler outside the hyperparameter search.
- The model is chosen by nested cross-validation on the development data. The test set is scored once, after the choice has been saved to `results/protocol.json`. The old script picked the model with the best test score.
- Malignant is now the positive class, and class names come from the dataset. The old confusion matrix labelled malignant tumours "No Cancer", and precision and recall referred to the benign class.
- Every model has a fixed random seed. Two runs on the tested setup produce byte-identical results.

Added:
- `evaluate.py`, `tests/test_no_leakage.py` (split, label and leakage checks), pinned `requirements.txt` and saved outputs in `results/`.
- Two baselines: majority class and untuned logistic regression.
- A development-only comparison of PCA off, 5 and 10 components.

Results, measured once on the 143-sample held-out test set: the selected model (an MLP chosen by nested CV) reached malignant-class F1 0.950 and recall 0.906 (5 of 53 malignant cases missed). The untuned logistic regression baseline reached F1 0.951 and recall 0.925. The tuned models did not meaningfully beat the simple baseline. Details and limits are in the README.

Changed:
- `model_training.py` moved to `legacy/model_training.py` with a header describing its known problems; its code is unchanged.
- The README was rewritten. It corrects the label encoding description, cites the dataset (CC BY 4.0) and notes that the correction was made with help from an AI coding assistant.
