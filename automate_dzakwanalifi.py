import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def parse_price(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace('Rp', '').replace('.', '').strip()
    if '/' in val_str:
        val_str = val_str.split('/')[0].strip()
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def preprocess_data(raw_csv_path, test_size=0.2, random_state=42):
    df = pd.read_csv(raw_csv_path)
    df = df.drop_duplicates().reset_index(drop=True)
    
    df['clean_original_price'] = df['original_price'].apply(parse_price)
    df['clean_discount_price'] = df['discount_price'].apply(parse_price)
    
    df['rating'] = df['rating'].fillna(df['rating'].median())
    df['clean_original_price'] = df['clean_original_price'].fillna(df['clean_discount_price'])
    
    target_facilities = ['Kasur', 'WiFi', 'Kloset Duduk', 'K. Mandi Dalam', 'AC', 'Akses 24 Jam']
    for fac in target_facilities:
        df[f'fac_{fac.replace(" ", "_").replace(".", "")}'] = df['facilities'].apply(
            lambda x: 1 if pd.notna(x) and fac in str(x) else 0
        )
        
    categorical_cols = ['city', 'gender', 'district']
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
    feature_cols = [
        'rating', 
        'clean_original_price', 
        'city_encoded', 
        'gender_encoded', 
        'district_encoded'
    ] + [f'fac_{fac.replace(" ", "_").replace(".", "")}' for fac in target_facilities]
    
    X = df[feature_cols]
    y = df['clean_discount_price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessing outputs
    preprocessed_df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    preprocessed_df.to_csv("mamikos_preprocessing_dataset.csv", index=False)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data("mamikos_promo_ngebut_indonesia.csv")
    print("Preprocessing automation executed successfully!")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
