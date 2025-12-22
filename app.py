import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Retail AI Dashboard",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    div[data-testid="metric-container"] {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load_data_from_db():
    try:
        conn = mysql.connector.connect(
            host="202.10.34.23",
            user="root",
            password="AKJWHd898a7wd12@#jhasf",
            database="tubes_dwbi"
        )
        
        query = """
        SELECT 
            f.Transaction_ID, 
            f.Customer_ID, 
            f.Store_ID, 
            f.Quantity, 
            f.Total_Amount,
            c.Age, 
            c.Gender,
            d.Full_Date,
            p.Product_Category
        FROM Fact_Sales f
        LEFT JOIN Dim_Customer c ON f.Customer_ID = c.Customer_ID
        LEFT JOIN Dim_Date d ON f.Date_ID = d.Date_ID
        LEFT JOIN Dim_Product p ON f.Product_ID = p.Product_ID
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        return df
        
    except mysql.connector.Error as err:
        st.error(f"Gagal Connect ke Database Error: {err}")
        return None

def process_data(df):
    if df is None or df.empty: return None

    df['Full_Date'] = pd.to_datetime(df['Full_Date'])
    df['Total_Amount'] = df['Total_Amount'].astype(float)
    df['Quantity'] = df['Quantity'].astype(int)

    df = df[df['Full_Date'].dt.year == 2023]
    
    df['Month'] = df['Full_Date'].dt.month
    df['Quarter'] = df['Full_Date'].dt.quarter
    
    store_map = {'S001': 'Bekasi', 'S002': 'Solo', 'S003': 'Jakarta', 'S004': 'Banjarmasin', 'S005': 'Bekasi'}
    df['City'] = df['Store_ID'].map(store_map).fillna('Unknown')
    
    def classify_age(age):
        if age < 25: return 'Young (<25)'
        elif age <= 50: return 'Adult (25-50)'
        else: return 'Senior (>50)'
    df['Age_Group'] = df['Age'].apply(classify_age)

    cust_stats = df.groupby('Customer_ID').agg({'Total_Amount': 'sum', 'Transaction_ID': 'count'}).reset_index()
    
    if len(cust_stats) >= 3:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cust_stats[['Total_Amount', 'Transaction_ID']])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cust_stats['Cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_means = cust_stats.groupby('Cluster')['Total_Amount'].mean().sort_values()
        labels = ['Hemat', 'Reguler', 'Sultan']
        mapping = {k: v for k, v in zip(cluster_means.index, labels)}
        cust_stats['Category'] = cust_stats['Cluster'].map(mapping)
        
        df = df.merge(cust_stats[['Customer_ID', 'Category']], on='Customer_ID', how='left')
    else:
        df['Category'] = 'Unknown'

    return df

raw_df = load_data_from_db()
df = process_data(raw_df)

if df is None:
    st.stop()


df_display = df[df['Quarter'] != 1]

st.sidebar.title("Filter Laporan")

avail_qs = sorted(df_display['Quarter'].unique())
if not avail_qs:
    st.warning("Data kosong atau belum ada data Q2-Q4 di database.")
    st.stop()
    
selected_q = st.sidebar.radio("Pilih Kuartal (2023):", avail_qs, format_func=lambda x: f"Q{x}", index=0)

df_curr = df_display[df_display['Quarter'] == selected_q]
df_prev = df[df['Quarter'] == (selected_q - 1)]

def get_prediction(current_df):
    if current_df.empty: return 0, 0, pd.DataFrame()
    daily_sales = current_df.groupby('Full_Date')['Total_Amount'].sum().reset_index()
    daily_sales['Day_Index'] = range(1, len(daily_sales) + 1)
    
    if len(daily_sales) > 1:
        model = LinearRegression()
        model.fit(daily_sales[['Day_Index']], daily_sales['Total_Amount'])
        future_days = np.array([[len(daily_sales) + i] for i in range(1, 31)])
        forecast = model.predict(future_days)
        total_forecast = forecast.sum()
        
        last_date = daily_sales['Full_Date'].iloc[-1]
        future_dates = [last_date + pd.DateOffset(days=i) for i in range(1, 31)]
        df_forecast = pd.DataFrame({'Full_Date': future_dates, 'Total_Amount': forecast, 'Type': 'Forecast'})
        daily_sales['Type'] = 'Actual'
        combined = pd.concat([daily_sales, df_forecast])
        return total_forecast, model.coef_[0], combined
    return 0, 0, pd.DataFrame()

pred_val, trend_val, df_trend = get_prediction(df_curr)
trend_status = "Naik" if trend_val > 0 else "Turun"

act_rev = df_curr['Total_Amount'].sum()
tgt_rev = df_prev['Total_Amount'].sum() * 1.05 if not df_prev.empty else act_rev

if not df_curr.empty:
    top_cat = df_curr.groupby('Product_Category')['Total_Amount'].sum().idxmax()
    top_age = df_curr.groupby('Age_Group')['Total_Amount'].sum().idxmax()
    top_city = df_curr.groupby('City')['Total_Amount'].sum().idxmax()
    
    rev_male = df_curr[df_curr['Gender']=='Male']['Total_Amount'].sum()
    rev_female = df_curr[df_curr['Gender']=='Female']['Total_Amount'].sum()
    
    if rev_male > rev_female:
        dom_gender = "Pria"
        diff_pct = ((rev_male - rev_female) / rev_female) * 100 if rev_female > 0 else 100
        delta_msg = f"Dominan +{diff_pct:.0f}%"
    elif rev_female > rev_male:
        dom_gender = "Wanita"
        diff_pct = ((rev_female - rev_male) / rev_male) * 100 if rev_male > 0 else 100
        delta_msg = f"Dominan +{diff_pct:.0f}%"
    else:
        dom_gender = "Seimbang"
        delta_msg = "Sama Kuat"
else:
    top_cat, dom_gender, top_age, top_city = "-", "-", "-", "-"
    delta_msg, rev_male, rev_female = "", 0, 0

def make_gauge(value, target, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta", value = value,
        number = {'valueformat': ",.0f"},
        title = {'text': title, 'font': {'size': 14, 'color': "gray"}},
        delta = {'reference': target, 'relative': True},
        gauge = {
            'axis': {'range': [None, target*1.2]}, 
            'bar': {'color': "#00CC96"},
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': target}
        }
    ))
    fig.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10))
    return fig

st.title(f"Dashboard Q{selected_q}")
st.markdown(f"**Status Prediksi:** Tren bulan depan diprediksi **{trend_status}**")

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.plotly_chart(make_gauge(act_rev, tgt_rev, "Revenue (Act vs Tgt)"), use_container_width=True)
with c2: st.metric("Prediksi Revenue", f"Rp {pred_val:,.0f}", "Next 30 Days")
with c3: st.metric("Top Kategori", top_cat)
with c4: 
    st.metric("Gender Dominan", dom_gender, delta_msg)
    st.caption(f"M: {rev_male:,.0f} | F: {rev_female:,.0f}")
with c5: st.metric("Top Lokasi", top_city)

st.divider()

col_L, col_R = st.columns([2, 1])

with col_L:
    st.subheader(f"Tren & Prediksi (Q{selected_q} + Next Month)")
    if not df_trend.empty:
        fig_trend = px.line(df_trend, x='Full_Date', y='Total_Amount', color='Type',
                            color_discrete_map={'Actual': '#636EFA', 'Forecast': '#FFA15A'},
                            markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Data tidak cukup untuk prediksi.")

with col_R:
    st.subheader("AI Segmentasi")
    if not df_curr.empty:
        user_view = df_curr.groupby(['Customer_ID', 'Category', 'Age']).agg({'Total_Amount':'sum'}).reset_index()
        fig_ai = px.scatter(user_view, x='Total_Amount', y='Age', color='Category',
                            color_discrete_map={'Sultan': '#FFD700', 'Reguler': '#00BFFF', 'Hemat': '#A9A9A9'},
                            title="Clustering User Q" + str(selected_q))
        st.plotly_chart(fig_ai, use_container_width=True)

c_b1, c_b2 = st.columns(2)
with c_b1:
    st.subheader("Kategori Produk Terlaris")
    if not df_curr.empty:
        fig_pie = px.pie(df_curr, values='Total_Amount', names='Product_Category', hole=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)
with c_b2:
    st.subheader("Peta Kota")
    if not df_curr.empty:
        city_data = df_curr.groupby('City')['Total_Amount'].sum().reset_index().sort_values('Total_Amount')
        fig_bar = px.bar(city_data, y='City', x='Total_Amount', orientation='h', color='Total_Amount')
        st.plotly_chart(fig_bar, use_container_width=True)

c_b1, c_b2 = st.columns(2)
with c_b1:
    st.subheader("Siapa yang Paling Boros? (Usia)")
    if not df_curr.empty:
        age_bar = df_curr.groupby('Age_Group')['Total_Amount'].sum().reset_index()
        color_map = {'Adult (25-50)': '#636EFA', 'Senior (>50)': '#1f77b4', 'Young (<25)': '#FF99CC'}
        
        fig_age = px.bar(age_bar, x='Age_Group', y='Total_Amount', color='Age_Group', 
                         title="Total Belanja per Kelompok Usia", text_auto='.2s',
                         color_discrete_map=color_map)
        st.plotly_chart(fig_age, use_container_width=True)
with c_b2:
    st.subheader("Analisis Gender")
    if not df_curr.empty:
        gender_pie = df_curr.groupby('Gender')['Total_Amount'].sum().reset_index()
        fig_gen = px.pie(gender_pie, values='Total_Amount', names='Gender', hole=0.5,
                         color_discrete_sequence=['#1f77b4', '#FF99CC'])
        st.plotly_chart(fig_gen, use_container_width=True)