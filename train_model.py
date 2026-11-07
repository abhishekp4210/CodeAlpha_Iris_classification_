from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import pandas as pd
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# 1. Load dataset
df = pd.read_csv("Iris.csv")

# 2. Separate features and target
X = df[['SepalLengthCm', 'SepalWidthCm',
        'PetalLengthCm', 'PetalWidthCm']]

y = df['Species']

# 3. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create ML model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier(n_neighbors=3))
])

# 5. Train the model
model.fit(X_train, y_train)
# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("-------------------------")
print("Accuracy:", accuracy * 100, "%")

# 8. Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save trained model
joblib.dump(model, 'iris_model.pkl')

print("Model trained successfully!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Trained model saved successfully!")