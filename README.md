# HealthcareCost
Medicare Cost Prediction Platform

A Streamlit-based data science project to enhance transparency in U.S. healthcare pricing by predicting Medicare procedure costs, allowing users to search by symptom or procedure and compare affordability across ZIP codes.

🔍 Project Overview

Millions of Americans struggle to estimate healthcare costs, especially for outpatient and inpatient procedures. Our platform addresses this challenge by:

Predicting the average Medicare payment for a given procedure
Providing affordability scores based on ZIP code income
Recommending cheaper nearby ZIP codes
Supporting natural language queries like "chest pain"
🎯 Key Features

✅ Symptom-to-procedure mapping using Sentence Transformers and FuzzyWuzzy
✅ LightGBM model for cost prediction (MAE ≈ $65)
✅ ZIP-level cost comparison with affordability context
✅ Interactive Streamlit interface with map visualizations
✅ Highlighting top cheapest ZIPs and most/least affordable states

📊 Dataset Sources

CMS Medicare Inpatient & Outpatient Datasets (2013–2022)
U.S. ZIP-level income & population data
Location coordinates (lat/lng) from OpenDataSoft
Manually created affordability score:
Affordability = Predicted Cost / Median Household Income


📦 Tech Stack

Area	           Tools
Language	       Python 3.12
Data Viz	       Streamlit, PyDeck, Matplotlib, Seaborn
ML Model	       LightGBM
NLP	             SentenceTransformers (MiniLM), FuzzyWuzzy
Data Handling	   Pandas, NumPy
Deployment	     Local / Streamlit Cloud (TBD)


📌 Future Improvements

Add hospital quality scores
Support private insurance cost comparisons
Mobile app version with accessibility options
Allow real user input of bills for community comparison
