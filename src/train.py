from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def train_models(X_train, y_train):
    """
    Trains multiple classifiers and returns the best model based on accuracy
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC()
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_train, y_train)
        print(f"{name} Training Accuracy: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    print(f"\n✅ Best Model Selected: {best_name} with Training Accuracy: {best_score:.4f}\n")
    return best_model, best_name
