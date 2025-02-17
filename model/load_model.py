import pickle

# Load the GradientBoostingClassifier model
model_path = "model.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Example: Print model parameters
print("Model Parameters:", model.get_params())

# Example: If the model has feature importances, print them
if hasattr(model, "feature_importances_"):
    print("Feature Importances:", model.feature_importances_)

# Example usage: Predict on sample data (modify 'X_sample' accordingly)
# X_sample = [...]  # Provide appropriate input data
# predictions = model.predict(X_sample)
# print("Predictions:", predictions)
