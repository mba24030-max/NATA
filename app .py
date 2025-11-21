
import streamlit as st
import pandas as pd
import joblib

st.title("Supermarket Spend Prediction App")

# Load all models from a single file
all_models = joblib.load('all_rf_models.pkl')
models = {
    'MntWines': all_models['MntWines'],
    'MntFruits': all_models['MntFruits'],
    'MntMeatProducts': all_models['MntMeatProducts'],
    'MntFishProducts': all_models['MntFishProducts'],
    'MntSweetProducts': all_models['MntSweetProducts'],
    'MntGoldProds': all_models['MntGoldProds']
}

features = [
    'Age', 'Tenure_days', 'Income', 'Kidhome', 'Teenhome', 'Recency',
    'Total_Spend', 'Total_Purchases', 'Recency_inverse', 'NumDealsPurchases',
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
    'Education', 'Marital_Status', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
    'AcceptedCmp4', 'AcceptedCmp5', 'Complain', 'Income_missing'
]

input_data = {}
for feat in features:
    input_data[feat] = st.number_input(feat, value=0)
input_df = pd.DataFrame([input_data])

if st.button('Predict'):
    for key, model in models.items():
        pred = model.predict(input_df)[0]
        st.write(f"Predicted {key}: {pred:.2f}")
