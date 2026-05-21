import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="ML Rekomendasi Wisata Yogya",
    page_icon="🏖️",
    layout="wide"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
.stMetric { 
    background-color: rgba(151,151,151,0.1); 
    padding: 15px; 
    border-radius: 10px; 
}

.rec-card {
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    background-color: rgba(151,151,151,0.1); 
    border-left: 8px solid #4CAF50;
}

.rec-card h3 {
    color: #4CAF50 !important;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    
    tour_url = "https://raw.githubusercontent.com/AwalDinz/rekomendasi-wisata-yogya/main/dataset/tour.csv"
    rating_url = "https://raw.githubusercontent.com/AwalDinz/rekomendasi-wisata-yogya/main/dataset/tour_rating.csv"

    tour = pd.read_csv(tour_url)
    rating = pd.read_csv(rating_url)

    return tour, rating

tour, rating = load_data()

# ===============================
# PREPROCESSING
# ===============================

# Gabungkan dataset
df = rating.merge(tour, on='Place_Id')

# Encode kategori & kota
le_category = LabelEncoder()
le_city = LabelEncoder()

df['Category_Enc'] = le_category.fit_transform(df['Category'])
df['City_Enc'] = le_city.fit_transform(df['City'])

# Feature dan target
X = df[['User_Id', 'Place_Id', 'Price', 'Rating', 'Category_Enc', 'City_Enc']]
y = df['Place_Ratings']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ===============================
# MACHINE LEARNING REGRESSION
# ===============================

@st.cache_resource
def train_model():

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

model = train_model()

# ===============================
# EVALUASI MODEL
# ===============================

pred_test = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred_test))

# ===============================
# RECOMMENDATION FUNCTION
# ===============================

def recommend_places(user_id, top_n=5):

    visited_places = rating[rating['User_Id'] == user_id]['Place_Id'].tolist()

    unvisited = tour[~tour['Place_Id'].isin(visited_places)].copy()

    if unvisited.empty:
        return pd.DataFrame()

    # Encode
    unvisited['Category_Enc'] = le_category.transform(unvisited['Category'])
    unvisited['City_Enc'] = le_city.transform(unvisited['City'])

    # Feature untuk prediksi
    predict_data = pd.DataFrame({
        'User_Id': [user_id] * len(unvisited),
        'Place_Id': unvisited['Place_Id'],
        'Price': unvisited['Price'],
        'Rating': unvisited['Rating'],
        'Category_Enc': unvisited['Category_Enc'],
        'City_Enc': unvisited['City_Enc']
    })

    # Prediksi rating
    unvisited['Predicted_Rating'] = model.predict(predict_data)

    # Urutkan
    recommendations = unvisited.sort_values(
        by='Predicted_Rating',
        ascending=False
    ).head(top_n)

    return recommendations

# ===============================
# SIDEBAR
# ===============================

with st.sidebar:

    st.title("⚙️ Panel Kontrol")

    menu = st.radio(
        "Pilih Menu",
        ["📊 Dashboard", "🎯 Rekomendasi ML"]
    )

    st.divider()

    if menu == "🎯 Rekomendasi ML":

        selected_user = st.selectbox(
            "Pilih User ID",
            sorted(rating['User_Id'].unique())
        )

        top_n = st.slider(
            "Jumlah Rekomendasi",
            3,
            10,
            5
        )

    st.info("Model menggunakan Machine Learning Regression (Random Forest Regressor)")

# ===============================
# DASHBOARD
# ===============================

if menu == "📊 Dashboard":

    st.title("📊 Dashboard Analisis Wisata")

    m1, m2, m3 = st.columns(3)

    m1.metric("Total Wisata", len(tour))
    m2.metric("Total User", rating['User_Id'].nunique())
    m3.metric("RMSE Model", round(rmse, 3))

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("📍 Top 10 Wisata")

        top_10 = rating.merge(
            tour,
            on='Place_Id'
        )['Place_Name'].value_counts().head(10)

        fig, ax = plt.subplots()

        sns.barplot(
            x=top_10.values,
            y=top_10.index,
            palette='magma',
            ax=ax
        )

        st.pyplot(fig)

    with c2:

        st.subheader("🏷️ Kategori Wisata")

        cat = tour['Category'].value_counts()

        fig, ax = plt.subplots()

        ax.pie(
            cat,
            labels=cat.index,
            autopct='%1.1f%%'
        )

        st.pyplot(fig)

# ===============================
# RECOMMENDATION PAGE
# ===============================

else:

    st.title("🎯 Rekomendasi Wisata dengan Machine Learning")

    st.write(f"Menampilkan rekomendasi untuk User ID: **{selected_user}**")

    if st.button("✨ Cari Rekomendasi"):

        with st.spinner("Melatih model dan memprediksi..."):

            recommendations = recommend_places(
                selected_user,
                top_n
            )

            if not recommendations.empty:

                st.success("Rekomendasi berhasil dibuat!")

                for _, row in recommendations.iterrows():

                    st.markdown(f"""
                    <div class="rec-card">

                    <h3>{row['Place_Name']}</h3>

                    <p>
                    📍 {row['City']} |
                    🏷️ <b>{row['Category']}</b>
                    </p>

                    <hr>

                    <p>
                    💰 Harga Tiket: Rp {int(row['Price']):,}
                    </p>

                    <p>
                    ⭐ Rating Asli: {row['Rating']}
                    </p>

                    <p style="color:#4CAF50;">
                    🔮 Prediksi Rating User:
                    <b>{row['Predicted_Rating']:.2f}</b>
                    </p>

                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.error("Tidak ada rekomendasi ditemukan.")

st.markdown("---")
st.caption("Machine Learning Recommendation System - Random Forest Regression")
