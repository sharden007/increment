from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
import numpy as np

# Step 1: Simulate Streaming Data
# Generate a classification dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# Split data into small batches to simulate streaming
batch_size = 100
n_batches = len(X) // batch_size
X_batches = np.array_split(X, n_batches)
y_batches = np.array_split(y, n_batches)

# Step 2: Initialize the Incremental Learning Model
# SGDClassifier supports incremental learning via partial_fit
model = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, tol=1e-3)

# Initialize classes (required for the first call to partial_fit)
classes = np.unique(y)

# Step 3: Incremental Training and Evaluation
for i, (X_batch, y_batch) in enumerate(zip(X_batches, y_batches)):
    # Train incrementally on each batch
    model.partial_fit(X_batch, y_batch, classes=classes)
    
    # Evaluate the model periodically (e.g., after every 10 batches)
    if i % 10 == 0:
        accuracy = model.score(X_batch, y_batch)
        print(f"Batch {i}: Accuracy = {accuracy:.2f}")

# Step 4: Final Evaluation on Test Data
# Generate a test dataset for evaluation
X_test, y_test = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=24)
test_accuracy = model.score(X_test, y_test)
print(f"Final Test Accuracy: {test_accuracy:.2f}")
