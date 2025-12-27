import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import mysql.connector
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Retail Intelligence Enterprise",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
    --background-color: #f5f6fa;
    --surface: #ffffff;
    --text-color: #1f2937;
}
html, body, [class*="css"]  {
    background: var(--background-color) !important;
    color: var(--text-color) !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load_data():
    conn = mysql.connector.connect(
        host="202.10.34.23",
        user="root",
        password="AKJWHd898a7wd12@#jhasf",
        database="tubes_dwbi"
    )
    query = """
    SELECT 
        f.fact_id,
        f.sk_customer,
        f.final_qty as Quantity, 
        f.final_amount as Total_Amount,
        c.`Customer ID` as Customer_Code,
        c.Age, 
        c.Gender,
        d.Date as Full_Date,
        d.Month,
        d.Quarter,
        p.`Product Category`,
        s.City
    FROM fact_sales f
    LEFT JOIN dim_customer c ON f.sk_customer = c.`technical/surrogate key field`
    LEFT JOIN dim_date d ON f.sk_date = d.smart_key
    LEFT JOIN dim_product p ON f.sk_product = p.`technical/surrogate key field`
    LEFT JOIN dim_store s ON f.sk_store = s.`technical/surrogate key field`
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def preprocess(df):
    df['Full_Date'] = pd.to_datetime(df['Full_Date'])
    df['Age_Group'] = df['Age'].apply(
        lambda a: 'Gen Z (<25)' if a < 25 else ('Millennial (25-50)' if a <= 50 else 'Boomer (>50)')
    )
    
    freq = df.groupby('Customer_Code')['fact_id'].count().rename("Frequency")
    df = df.merge(freq, on='Customer_Code', how='left')

    cust_stats = df.groupby('Customer_Code').agg({
        'Total_Amount': 'sum',
        'fact_id': 'count'
    }).reset_index()

    if len(cust_stats) >= 3:
        scaler = StandardScaler()
        X = scaler.fit_transform(cust_stats[['Total_Amount', 'fact_id']])
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        cust_stats['Cluster'] = kmeans.fit_predict(X)
        mapping = {0: 'Low Value', 1: 'Medium Value', 2: 'High Value'}
        cust_stats['Cluster_Label'] = cust_stats['Cluster'].map(mapping)
        df = df.merge(cust_stats[['Customer_Code', 'Cluster_Label']], on='Customer_Code', how='left')
    else:
        df['Cluster_Label'] = 'Uncategorized'

    return df

raw = load_data()
df = preprocess(raw)


st.sidebar.title("Control Panel")
quarter = st.sidebar.selectbox("Pilih Quarter", sorted(df['Quarter'].unique()))
filtered = df[df['Quarter'] == quarter]

total_revenue = filtered['Total_Amount'].sum()
total_trx = filtered['fact_id'].count()
avg_order = total_revenue / total_trx if total_trx else 0
top_category = filtered.groupby('Product Category')['Total_Amount'].sum().idxmax()
most_frequent_customer = filtered.groupby('Customer_Code')['Frequency'].max().idxmax()
city_perf = filtered.groupby('City')['Total_Amount'].sum()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>Total Revenue</h3>
        <h2>Rp {total_revenue:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>Total Transactions</h3>
        <h2>{total_trx}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>Average Order Value</h3>
        <h2>Rp {avg_order:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>Top Category</h3>
        <h2>{top_category}</h2>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>Most Frequent Customer</h3>
        <h2>{most_frequent_customer}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("###")


st.subheader("Gender × Product Category Contribution")

gender_category = filtered.groupby(['Gender', 'Product Category'])['Total_Amount'].sum().reset_index()

fig_stack = px.bar(
    gender_category,
    x="Product Category",
    y="Total_Amount",
    color="Gender",
    barmode="stack",
    template="plotly_white"
)

st.plotly_chart(fig_stack, use_container_width=True)


colA, colB = st.columns(2)

with colA:
    st.subheader("Age Group Contribution")
    age_rev = filtered.groupby("Age_Group")['Total_Amount'].sum().reset_index()
    fig_age = px.bar(age_rev, x='Age_Group', y='Total_Amount', template='plotly_white')
    st.plotly_chart(fig_age, use_container_width=True)

with colB:
    st.subheader("Customer Frequency Distribution")
    freq_df = filtered.groupby('Customer_Code')['Frequency'].mean().reset_index()
    fig_freq = px.histogram(freq_df, x='Frequency', nbins=10, template='plotly_white')
    st.plotly_chart(fig_freq, use_container_width=True)

colC, colD = st.columns(2)

with colC:
    st.subheader("Sales by City")
    city_df = filtered.groupby("City")['Total_Amount'].sum().reset_index()
    fig_city = px.bar(city_df, x='City', y='Total_Amount', template='plotly_white')
    st.plotly_chart(fig_city, use_container_width=True)

with colD:
    st.subheader("Sales Trend by Date")
    time_df = filtered.groupby("Full_Date")['Total_Amount'].sum().reset_index()
    fig_time = px.line(time_df, x='Full_Date', y='Total_Amount', template='plotly_white')
    st.plotly_chart(fig_time, use_container_width=True)
