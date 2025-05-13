# Step 1: Force CPU use and import libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force use of CPU for torch
import torch

def to_cpu(tensor):
    return tensor.cpu() if tensor.is_cuda else tensor

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import pydeck as pdk

# Step 2: Load merged dataset
merged_df = pd.read_csv("cms_with_income_and_location.csv", low_memory=False)
merged_df['ZIP'] = merged_df['ZIP'].astype(str).str.zfill(5)  # Ensure ZIP is string

# Step 3: Setup procedure embedding model
distinct_procedures = merged_df['PROCEDURE'].dropna().unique().tolist()
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
procedure_embeddings = to_cpu(model_embed.encode(distinct_procedures, convert_to_tensor=True))

# Step 4: Show national map of average cost by state
st.title("Medicare Cost Prediction Platform")

# Search bar and below that show national map
procedure_query = st.text_input("Search your procedure or symptom (e.g., MRI, shoulder pain, biopsy):")
zipcode_input = st.text_input("Enter your ZIP code (optional):")

# Show US map of average cost by state
state_costs = merged_df.groupby("state_name")[["Avg_Mdcr_Pymt_Amt"]].mean().reset_index()
states_latlng = merged_df.groupby("state_name")[["lat", "lng"]].median().reset_index()
state_map_df = pd.merge(state_costs, states_latlng, on="state_name")

# Compute color column
min_cost = state_map_df['Avg_Mdcr_Pymt_Amt'].min()
max_cost = state_map_df['Avg_Mdcr_Pymt_Amt'].max()
norm = (state_map_df['Avg_Mdcr_Pymt_Amt'] - min_cost) / (max_cost - min_cost)
state_map_df['r'] = (norm * 255).astype(int)
state_map_df['g'] = (255 - state_map_df['r']).astype(int)
state_map_df['b'] = 120
state_map_df['color'] = state_map_df[['r', 'g', 'b']].apply(lambda row: [row['r'], row['g'], row['b']], axis=1)

st.subheader("National View: Most vs Least Expensive States")
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(latitude=37.5, longitude=-95.0, zoom=3.5, pitch=0),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=state_map_df,
            get_position='[lng, lat]',
            get_color='color',
            get_radius=80000,
            pickable=True
        )
    ],
    tooltip={"text": "{state_name}\nAvg Cost: ${Avg_Mdcr_Pymt_Amt:.2f}"}
))

if procedure_query:
    query_embedding = to_cpu(model_embed.encode(procedure_query, convert_to_tensor=True))
    similarity_scores = util.cos_sim(query_embedding, procedure_embeddings)[0].cpu().numpy()
    top_idx = int(np.argmax(similarity_scores))
    matched_procedure = distinct_procedures[top_idx]
    st.markdown(f"### Matched Procedure: **{matched_procedure}**")

    filtered_df = merged_df[merged_df['PROCEDURE'] == matched_procedure].dropna(subset=['Avg_Mdcr_Pymt_Amt', 'ZIP', 'Households Median Income (Dollars)', 'population'])
    filtered_df = filtered_df.drop(columns=['Avg_Mdcr_Alowd_Amt', 'Avg_Tot_Sbmtd_Chrgs'], errors='ignore')
    filtered_df.columns = filtered_df.columns.str.replace(r'[\s\(\)]', '_', regex=True)
    filtered_df = filtered_df.select_dtypes(include=[np.number, 'bool']).copy()

    target = 'Avg_Mdcr_Pymt_Amt'
    features = [col for col in filtered_df.columns if col != target]
    X = filtered_df[features]
    y = filtered_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=64, max_depth=10, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    example_input = X.mean().to_frame().T
    st.success(f"LightGBM MAE: ${mae:.2f} | Typical ZIP Predicted Cost: ${model.predict(example_input)[0]:.2f}")

    mean_input = X.mean()
    def predict_cost(zip_val, median_income, mean_income, pop, affordability):
        input_data = mean_input.copy()
        input_data['ZIP'] = int(zip_val)
        input_data['Households_Median_Income_Dollars'] = median_income
        input_data['Households_Mean_Income_Dollars'] = mean_income
        input_data['population'] = pop
        input_data['Affordability_Score'] = affordability
        input_df = pd.DataFrame([input_data])[X.columns]
        return model.predict(input_df)[0]

    zip_group = merged_df.groupby("ZIP")[[
        "Households Median Income (Dollars)",
        "Households Mean Income (Dollars)",
        "population",
        "Affordability_Score",
        "city",
        "state_name"
    ]].median().dropna().reset_index()
    zip_group['ZIP'] = zip_group['ZIP'].astype(str).str.zfill(5)
    zip_latlng = merged_df[['ZIP', 'lat', 'lng']].dropna().drop_duplicates()
    zip_latlng['ZIP'] = zip_latlng['ZIP'].astype(str).str.zfill(5)

    zip_group['Predicted_Cost'] = zip_group.apply(
        lambda row: predict_cost(
            zip_val=row['ZIP'],
            median_income=row['Households Median Income (Dollars)'],
            mean_income=row['Households Mean Income (Dollars)'],
            pop=row['population'],
            affordability=row['Affordability_Score']
        ), axis=1
    )

    map_df = pd.merge(zip_group, zip_latlng, on='ZIP', how='left').dropna(subset=['lat', 'lng'])

    st.subheader("Top 5 Cheapest ZIPs")
    display_cols = ['ZIP', 'Households Median Income (Dollars)', 'Affordability_Score', 'Predicted_Cost', 'city', 'state_name']
    st.dataframe(map_df[display_cols].sort_values("Predicted_Cost").head())

    st.subheader("Predicted Medicare Costs Map")
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(latitude=37.5, longitude=-95.0, zoom=3.5, pitch=0),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=map_df,
                get_position='[lng, lat]',
                get_color='[int((Predicted_Cost - map_df["Predicted_Cost"].min()) / (map_df["Predicted_Cost"].max() - map_df["Predicted_Cost"].min()) * 255), 255 - int((Predicted_Cost - map_df["Predicted_Cost"].min()) / (map_df["Predicted_Cost"].max() - map_df["Predicted_Cost"].min()) * 255), 100]',
                get_radius=25000,
                pickable=True
            )
        ],
        tooltip={"text": "ZIP: {ZIP}\nCost: ${Predicted_Cost:.2f}"}
    ))

    if zipcode_input:
        zipcode_input = str(zipcode_input).zfill(5)
        if zipcode_input in map_df['ZIP'].values:
            user_cost = map_df[map_df['ZIP'] == zipcode_input]['Predicted_Cost'].values[0]
            st.markdown(f"### Cost at ZIP {zipcode_input}: ${user_cost:.2f}")
            st.markdown("#### Consider traveling to these cheaper ZIPs:")
            st.dataframe(map_df.sort_values("Predicted_Cost").head(10))
        else:
            st.warning("ZIP code not found in dataset.")
else:
    st.info("Please enter a symptom or procedure to begin.")
