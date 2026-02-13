import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(path):
    """
    Loads CSV, encodes categorical features, scales numerical features,
    and splits into train and test sets.
    Returns:
    X_train, X_test, y_train, y_test, scaler, encoders, df
    """

    df = pd.read_csv(path)

    # Separate target
    y = df['Class']
    X = df.drop('Class', axis=1)

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoders = {}

    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Scale numerical features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, encoders, df
