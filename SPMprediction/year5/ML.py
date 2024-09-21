import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, matthews_corrcoef
import warnings
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance  # Import permutation importance calculation

# Suppress warnings
warnings.filterwarnings('ignore')

# Load training and testing datasets
train_data = pd.read_csv('trainSet.csv', index_col=0)
test_data = pd.read_csv('testSet.csv', index_col=0)

# Preprocessing: Separate features and labels
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class'].apply(lambda x: 1 if x == 'SPM' else 0)  # Binary encoding for the target

X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class'].apply(lambda x: 1 if x == 'SPM' else 0)  # Binary encoding for the target

# Define models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42),
    'LinearSVC': LinearSVC(random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'GaussianNB': GaussianNB()
}

# Parameter grids for hyperparameter tuning
param_grids = {
    'RandomForest': {
        'n_estimators': [50, 100, 200, 500]  # Number of trees
    },
    'XGBoost': {
        'max_depth': [3, 5, 7, 10]  # Depth of the trees
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200, 500]  # Number of boosting stages
    },
    'LogisticRegression': {
        'C': [0.1, 1, 5, 10]  # Inverse regularization strength
    },
    'LinearSVC': {
        'C': [0.1, 1, 5, 10]  # Inverse regularization strength
    },
    'DecisionTree': {
        'max_depth': [1, 3, 5, 10]  # Maximum depth of the tree
    },
    'GaussianNB': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]  # Variance smoothing parameter
    }
}

# Train all models and save the best ones
best_models = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")

    # Grid search with cross-validation for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='recall')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best Model Parameters for {model_name}: {grid_search.best_params_}")

    # Save the best model
    best_models[model_name] = best_model

# Run 10 iterations, each time randomly selecting 50% of the test set for prediction
results = []
importance_results = []  # To store feature importance results

for i in range(10):
    print(f"\nIteration {i + 1}/10")

    # Randomly select 50% of the test set
    X_test_sample, _, y_test_sample, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=i)

    for model_name, best_model in best_models.items():
        print(f"Evaluating {model_name} in iteration {i + 1}...")

        # Predict on the new test sample
        y_pred = best_model.predict(X_test_sample)

        if model_name == 'LinearSVC':
            y_pred_prob = best_model.decision_function(X_test_sample)  # Decision function for SVM
        else:
            y_pred_prob = best_model.predict_proba(X_test_sample)[:, 1]  # Probability prediction for others

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test_sample, y_pred)
        precision = precision_score(y_test_sample, y_pred)
        recall = recall_score(y_test_sample, y_pred)
        f1 = f1_score(y_test_sample, y_pred)
        roc_auc = roc_auc_score(y_test_sample, y_pred_prob)
        mcc = matthews_corrcoef(y_test_sample, y_pred)

        # Print evaluation results
        print(f'{model_name} - Accuracy: {accuracy:.4f}')
        print(f'{model_name} - Precision: {precision:.4f}')
        print(f'{model_name} - Recall: {recall:.4f}')
        print(f'{model_name} - F1 Score: {f1:.4f}')
        print(f'{model_name} - ROC AUC: {roc_auc:.4f}')
        print(f'{model_name} - MCC: {mcc:.4f}')

        # Save evaluation results
        results.append({
            'Iteration': i + 1,
            'Model': model_name,
            'AUC': roc_auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'MCC': mcc
        })

        # **Calculate Permutation Importance**
        perm_importance = permutation_importance(best_model, X_test_sample, y_test_sample,
                                                 n_repeats=5,
                                                 random_state=42)

        # Set all negative values to 0 to avoid negative impact on total importance
        perm_importance.importances_mean[perm_importance.importances_mean < 0] = 0

        # Calculate relative importance by normalizing feature importance scores
        total_importance = perm_importance.importances_mean.sum() + 0.1  # Total importance
        relative_importance = perm_importance.importances_mean / total_importance  # Relative importance

        # Save feature relative importance results
        for feature_idx in range(X_test_sample.shape[1]):
            importance_results.append({
                'Iteration': i + 1,
                'Model': model_name,
                'Feature': X_train.columns[feature_idx],  # Feature name
                'RelativeImportance': relative_importance[feature_idx]  # Relative importance score
            })

        # Print feature relative importance
        print(f"\n{model_name} Feature Importances (Relative):")
        for feature, importance in zip(X_train.columns, relative_importance):
            print(f"{feature}: {importance:.4f}")

# Create DataFrame for all results
results_df = pd.DataFrame(results)
importance_df = pd.DataFrame(importance_results)  # Feature importance results

# Print results
print(results_df)
print(importance_df)

# Save results to CSV files
results_df.to_csv('result.csv', index=False)
importance_df.to_csv('feature_importance.csv', index=False)
