import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="2023 Performance Dashboard",
    page_icon="👑",
    layout="wide"
)

# Custom CSS buat KPI Card biar rapi
st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    div[data-testid="metric-container"] {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD DATA (HANYA 2023) ---
@st.cache_data
def load_data():
    try:
        # Load CSV
        df = pd.read_csv('final_data_warehouse.csv')
        df['Full_Date'] = pd.to_datetime(df['Full_Date'])
        
        # 🧹 FILTER KERAS: Ambil Cuma 2023
        df = df[df['Full_Date'].dt.year == 2023]
        
        # Enrich Data (Bikin Kolom Tambahan)
        df['Month'] = df['Full_Date'].dt.month_name()
        df['Quarter'] = df['Full_Date'].dt.quarter
        
        # Age Grouping
        def classify_age(age):
            if age < 25: return 'Young (<25)'
            elif age <= 50: return 'Adult (25-50)'
            else: return 'Senior (>50)'
        
        if 'Age' in df.columns:
            df['Age_Group'] = df['Age'].apply(classify_age)
        else:
            df['Age_Group'] = "Unknown"

        # Mapping Kota (Store Locations)
        store_map = {
            'S001': 'Bekasi', 'S002': 'Solo', 'S003': 'Jakarta',
            'S004': 'Banjarmasin', 'S005': 'Bekasi'
        }
        df['City'] = df['Store_ID'].map(store_map).fillna('Unknown')
        
        return df
    except FileNotFoundError:
        return None

df = load_data()
if df is None:
    st.error("❌ File 'final_data_warehouse.csv' gak ketemu! Pastikan satu folder sama app.py")
    st.stop()

if df.empty:
    st.error("❌ Data 2023 kosong! Cek file CSV lo.")
    st.stop()

# --- 3. FUNGSI GAUGE CHART ---
def make_gauge(value, target, title, suffix="", color_hex="#29B5E8"):
    # Max value dinamis
    max_val = target * 1.2 if target > 0 else (value * 1.2 if value > 0 else 100)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        number = {'suffix': suffix, 'font': {'size': 20}, 'valueformat': ",.0f"}, # Format angka
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 14, 'color': "gray"}},
        delta = {'reference': target, 'relative': True, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1},
            'bar': {'color': color_hex},
            'bgcolor': "white",
            'steps': [{'range': [0, target], 'color': "#f2f2f2"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': target}
        }
    ))
    fig.update_layout(height=160, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# --- 4. SIDEBAR (PILIH QUARTER) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920349.png", width=80)
st.sidebar.title("Filter Laporan")
st.sidebar.info("📅 **Tahun Laporan: 2023**") # Statis karena cuma 2023

# Pilih Quarter
avail_qs = sorted(df['Quarter'].unique())
selected_q = st.sidebar.radio("Pilih Kuartal (Quarter):", avail_qs, format_func=lambda x: f"Q{x} (2023)", index=0)

# Filter Dataset
df_curr = df[df['Quarter'] == selected_q]

# Siapkan Data Kuartal Sebelumnya (Buat Target)
if selected_q == 1:
    # Kalau Q1, kita anggap targetnya = Actual (Baseline), karena gak ada data 2022
    df_prev = pd.DataFrame() 
else:
    df_prev = df[df['Quarter'] == (selected_q - 1)]

# --- 5. HITUNG 5 KPI ---

# KPI 1: Total Revenue
act_rev = df_curr['Total_Amount'].sum()
# Target Q2/Q3/Q4 = Quarter Sebelumnya + 5%. Target Q1 = Actual Q1.
tgt_rev = df_prev['Total_Amount'].sum() * 1.05 if not df_prev.empty else act_rev

# KPI 2: Top Category Contribution
if not df_curr.empty:
    cat_stats = df_curr.groupby('Product_Category')['Total_Amount'].sum()
    top_cat = cat_stats.idxmax()
    top_cat_pct = (cat_stats.max() / act_rev) * 100
else:
    top_cat, top_cat_pct = "-", 0

# KPI 3: Gender Ratio (AOV Male / AOV Female)
male_avg = df_curr[df_curr['Gender']=='Male']['Total_Amount'].mean() if not df_curr[df_curr['Gender']=='Male'].empty else 0
fem_avg = df_curr[df_curr['Gender']=='Female']['Total_Amount'].mean() if not df_curr[df_curr['Gender']=='Female'].empty else 0
ratio_gender = (male_avg / fem_avg) if fem_avg > 0 else 0
ratio_str = f"{ratio_gender:.2f}x" if fem_avg > 0 else "N/A"

# KPI 4: Top Age Group Spending
if not df_curr.empty:
    age_grp = df_curr.groupby('Age_Group')['Total_Amount'].sum()
    top_age = age_grp.idxmax()
    top_age_val = age_grp.max()
else:
    top_age, top_age_val = "-", 0

# KPI 5: Top Store Location
if not df_curr.empty:
    city_grp = df_curr.groupby('City')['Total_Amount'].sum()
    top_city = city_grp.idxmax()
    top_city_val = city_grp.max()
else:
    top_city, top_city_val = "-", 0

# --- 6. VISUALISASI UTAMA ---
st.title(f"👑 Laporan Kinerja Q{selected_q} - 2023")
st.markdown("Evaluasi KPI per Kuartal (Target Growth: +5% QoQ)")

# Row 1: KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)

with c1: st.plotly_chart(make_gauge(act_rev, tgt_rev, "Total Revenue (Rp)", color_hex="#00CC96"), use_container_width=True)
with c2: st.metric("🏆 Top Kategori", top_cat, f"{top_cat_pct:.1f}% Share")
with c3: st.metric("👫 Rasio Gender (M/F)", ratio_str, "AOV Pria vs Wanita")
with c4: st.metric("👥 Grup Usia Dominan", top_age, f"Rp {top_age_val:,.0f}")
with c5: st.metric("📍 Lokasi Juara", top_city, f"Rp {top_city_val:,.0f}")

st.divider()

# Row 2: Charts
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"📈 Tren Transaksi Harian (Q{selected_q})")
    daily = df_curr.groupby('Full_Date')['Total_Amount'].sum().reset_index()
    if not daily.empty:
        fig_trend = px.line(daily, x='Full_Date', y='Total_Amount', markers=True, 
                            title="Grafik Detak Jantung Bisnis", color_discrete_sequence=['#636EFA'])
        fig_trend.update_xaxes(title="Tanggal")
        fig_trend.update_yaxes(title="Omzet (Rp)")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Data kosong.")

with col_right:
    st.subheader("🍰 Kontribusi Kategori")
    if not df_curr.empty:
        cat_pie = df_curr.groupby('Product_Category')['Total_Amount'].sum().reset_index()
        fig_pie = px.pie(cat_pie, values='Total_Amount', names='Product_Category', hole=0.5,
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)

# Row 3: Deep Dive (Age & City)
c_b1, c_b2 = st.columns(2)

with c_b1:
    st.subheader("👥 Siapa yang Paling Boros? (Usia)")
    if not df_curr.empty:
        age_bar = df_curr.groupby('Age_Group')['Total_Amount'].sum().reset_index()
        fig_age = px.bar(age_bar, x='Age_Group', y='Total_Amount', color='Age_Group', 
                         title="Total Belanja per Kelompok Usia", text_auto='.2s')
        st.plotly_chart(fig_age, use_container_width=True)

with c_b2:
    st.subheader("📍 Performa Cabang (Kota)")
    if not df_curr.empty:
        city_bar = df_curr.groupby('City')['Total_Amount'].sum().reset_index().sort_values('Total_Amount')
        fig_city = px.bar(city_bar, y='City', x='Total_Amount', orientation='h', 
                          title="Ranking Kota Berdasarkan Omzet", color='Total_Amount', color_continuous_scale='Viridis')
        st.plotly_chart(fig_city, use_container_width=True)

# Expander Dokumentasi
with st.expander("📖 Baca Penjelasan Teknis KPI (Untuk Dosen)"):
    st.markdown("""
    **Catatan Target:**
    Dashboard ini fokus pada **Tahun Fiskal 2023**. 
    * Target **Q1** diset *Baseline* (sama dengan Actual) karena tidak ada data 2022.
    * Target **Q2, Q3, Q4** menggunakan metode *Quarter-on-Quarter (QoQ)* dengan ekspektasi pertumbuhan **+5%** dari kuartal sebelumnya.
    
    **Definisi 5 KPI:**
    1. **Total Sales Revenue:** Indikator utama arus kas masuk (Cash In).
    2. **Product Category Share:** Menunjukkan produk pareto (paling laku).
    3. **Gender Ratio:** Mengukur daya beli relatif. Jika > 1, berarti Pria belanja lebih mahal per kunjungan dibanding Wanita.
    4. **Age Group:** Segmentasi pasar berdasarkan umur (Young/Adult/Senior).
    5. **Store Location:** Evaluasi kinerja geografis cabang.
    """)