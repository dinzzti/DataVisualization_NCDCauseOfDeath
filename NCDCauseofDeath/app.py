import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import warnings 

warnings.filterwarnings('ignore') 

DELIMITER = ';' 
FILE_PATH_IDN_FACTOR = 'INDONCDfactorCVD.csv' 

st.set_page_config(
    page_title="Analisis Kematian NCD Komprehensif Asia",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(file_path, delimiter):
    """Memuat data dari file CSV dengan encoding latin-1 dan membersihkan kolom."""
    try:
        data = pd.read_csv(file_path, encoding='latin-1', sep=delimiter)
        
        data.columns = [col.strip().replace('_', ' ').title() for col in data.columns]
        
        for col in ['Region', 'Country', 'Sex', 'Type Disease', 'Cause Of Death']:
            if col in data.columns:
                data[col] = data[col].astype(str).str.strip()
        
        if 'Year' in data.columns:
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce').astype('Int64')
        if 'Total Ncd Death' in data.columns:
            data['Total Ncd Death'] = pd.to_numeric(data['Total Ncd Death'], errors='coerce')
        
        
        if 'Rate Per 100K' in data.columns:
            
            data['Rate Per 100K'] = data['Rate Per 100K'].astype(str)
            
            data['Rate Per 100K'] = data['Rate Per 100K'].str.replace(',', '.', regex=False)
            
            data['Rate Per 100K'] = pd.to_numeric(data['Rate Per 100K'], errors='coerce')

        data.dropna(subset=[col for col in data.columns if col in ['Year', 'Total Ncd Death', 'Rate Per 100K']], inplace=True)
        
        return data
    except Exception as e:
        st.sidebar.error(f"Gagal memuat file: {file_path}. Error: ({e})")
        return pd.DataFrame()

@st.cache_data
def load_and_clean_factor_data(file_path):
    """Memuat dan membersihkan data faktor risiko NCD provinsi Indonesia."""
    try:
        
        df_idn = pd.read_csv(file_path, sep=',', on_bad_lines='skip')

        df_idn.columns = [
            col.strip().replace('2017', '17').replace('2016', '16').replace('_', '').replace(' ', '').title() 
            for col in df_idn.columns
        ]
        
        factor_cols_to_keep = [
            'Locationname', 'SexName',
            'Smoke17M', 'Sbp17M', 'Bmi17T', 'Fpg17T', 'Sodium17T', 
            'Pmeat17M', 'Ambientpm17T', 'Sdi2016T', 'Fiber17T',
            'Ncd17T', 'Cvd2017T' 
        ]
        
        df_idn_clean = df_idn[[col for col in factor_cols_to_keep if col in df_idn.columns]].copy()
     
        for col in df_idn_clean.columns:
            if col not in ['Locationname', 'SexName']:
                df_idn_clean[col] = pd.to_numeric(df_idn_clean[col], errors='coerce')

       
        df_idn_agg = df_idn_clean.groupby('Locationname').mean(numeric_only=True).reset_index()
     
        df_idn_agg = df_idn_agg[df_idn_agg['Locationname'] != 'Indonesia'].dropna(axis=1, how='all')
        
        return df_idn_agg
    except Exception as e:
        st.warning(f"Gagal memuat atau membersihkan data faktor risiko provinsi. Error: {e}")
        return pd.DataFrame()


df_raw = load_data('NCD_Deaths_ALL_ASIA_CLEANED.csv', DELIMITER)
df_cause_death = load_data('Cause_of_Death_IDN_CLEANED.csv', DELIMITER) 
df_idn_factors = load_and_clean_factor_data(FILE_PATH_IDN_FACTOR) # Memuat data faktor risiko provinsi

if df_raw.empty:
    st.title("Dashboard NCD Asia")
    st.error("Gagal memuat data utama. Mohon periksa file.")
    st.stop()


st.sidebar.header("Filter Analisis")

all_regions = sorted(df_raw['Region'].unique())
selected_regions = st.sidebar.multiselect(
    "🌍 Filter Region/Wilayah (Asia)",
    options=all_regions,
    default=all_regions, 
)
df_regional = df_raw[df_raw['Region'].isin(selected_regions)].copy()

if df_regional.empty:
    st.warning("Pilih minimal satu Region untuk melanjutkan analisis.")
    st.stop()

all_countries_in_region = sorted(df_regional['Country'].unique())
selected_countries = st.sidebar.multiselect(
    "🗺 Filter Negara",
    options=all_countries_in_region,
    default=all_countries_in_region
)

all_sex = df_regional['Sex'].unique()
selected_sex = st.sidebar.multiselect(
    "🚻 Filter Jenis Kelamin",
    options=all_sex,
    default=all_sex 
)

min_year = int(df_regional['Year'].min())
max_year = int(df_regional['Year'].max())
year_range = st.sidebar.slider(
    "📅 Filter Rentang Tahun",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1
)

df_filtered = df_regional[
    (df_regional['Country'].isin(selected_countries)) &
    (df_regional['Sex'].isin(selected_sex)) &
    (df_regional['Year'] >= year_range[0]) &
    (df_regional['Year'] <= year_range[1])
].copy()

if df_filtered.empty:
    st.title("🌏 Dashboard angka kematian NCD (Asia)")
    st.error("Data kosong. Sesuaikan filter di sidebar.")
    st.stop()

def calculate_growth(df):
    df = df.sort_values('Year')
    df['Prev_Death'] = df['Total Ncd Death'].shift(1)
    df['Growth_Rate'] = np.where(df['Prev_Death'] != 0, ((df['Total Ncd Death'] - df['Prev_Death']) / df['Prev_Death']) * 100, 0)
    return df.drop(columns=['Prev_Death'])

df_filtered_growth = df_filtered.groupby(['Country', 'Sex']).apply(calculate_growth).reset_index(drop=True)

df_filtered_growth['Moving_Avg_3Y'] = df_filtered_growth.groupby(['Country', 'Sex'])['Total Ncd Death'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

df_ratio_base = df_filtered.groupby(['Country', 'Year', 'Sex'])['Total Ncd Death'].sum().reset_index()
df_male = df_ratio_base[df_ratio_base['Sex'] == 'Male'].rename(columns={'Total Ncd Death': 'Male_Death'})
df_female = df_ratio_base[df_ratio_base['Sex'] == 'Female'].rename(columns={'Total Ncd Death': 'Female_Death'})
df_ratio = pd.merge(df_male[['Country', 'Year', 'Male_Death']], 
                    df_female[['Country', 'Year', 'Female_Death']], 
                    on=['Country', 'Year'], how='inner')
df_ratio['Male_Female_Ratio'] = np.where(df_ratio['Female_Death'] != 0, df_ratio['Male_Death'] / df_ratio['Female_Death'], np.nan)
df_ratio.replace([np.inf, -np.inf], np.nan, inplace=True)

st.title("🌏 Analisis Disparitas Regional & Agka Kematian NCD di Asia")
st.markdown(f"*Cakupan Data Utama:* Region *{', '.join(selected_regions)}* | *{len(selected_countries)}* Negara | Tahun: *{year_range[0]} - {year_range[1]}*")


with st.container():
    col1, col2, col3, col4 = st.columns(4)
    total_deaths = df_filtered['Total Ncd Death'].sum()
    avg_growth = df_filtered_growth['Growth_Rate'].mean()
    avg_ratio = df_ratio['Male_Female_Ratio'].mean()

    avg_moving_avg = df_filtered_growth['Moving_Avg_3Y'].mean() 

    with col1:
        st.metric(label="Total Kematian NCD", value=f"{total_deaths:,.0f}")
    with col2:
        st.metric(label="Rata-rata Pertumbuhan Tahunan (%)", value=f"{avg_growth:.2f}%")
    with col3:
        st.metric(label="Rata-rata Rasio Pria/Wanita", value=f"{avg_ratio:.2f}")
    with col4:
        st.metric(label="Rata-rata Bergerak 3-Tahun (Global)", value=f"{avg_moving_avg:,.0f}")

st.markdown("---")

st.header("🗺 Disparitas Regional: Peta Tingkat Kematian NCD")

df_map_data = df_filtered.groupby('Country')['Total Ncd Death'].sum().reset_index()


fig_map = px.choropleth(df_map_data, 
                        locations='Country', 
                        locationmode='country names', 
                        color='Total Ncd Death', 
                        hover_name='Country', 
                        color_continuous_scale=px.colors.sequential.Plasma, 
                        title=f"Total Kematian NCD di {', '.join(selected_regions)} ({year_range[0]} - {year_range[1]})",
                        height=600
                       )
                       
fig_map.update_traces(marker_line_color='black', marker_line_width=1.5)
                       
fig_map.update_layout(
    margin={"r":0,"t":50,"l":0,"b":0},
    geo=dict(
        showframe=True, 
        framecolor='black',
        showcoastlines=False,
        projection_type='natural earth', 
        scope='asia', 
        center={'lat': 25, 'lon': 90} 
    )
)

st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

col_vis1, col_vis2 = st.columns(2)
with col_vis1:
    st.subheader("📊 Total Kematian NCD per Negara")
    df_bar_data = df_filtered.groupby('Country')['Total Ncd Death'].sum().reset_index().sort_values(by='Total Ncd Death', ascending=False)
    fig_bar = px.bar(df_bar_data, x='Country', y='Total Ncd Death', color='Country', title="Peringkat Beban Kematian NCD per Negara", template="plotly_white", height=450)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_vis2:
    st.subheader("🚹🚺 Proporsi Jenis Kelamin Berdasarkan Kematian")
    df_pie = df_filtered.groupby('Sex')['Total Ncd Death'].sum().reset_index()
    fig_pie = px.pie(df_pie, names='Sex', values='Total Ncd Death', hole=.4, color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'], title="Perbandingan Kematian Pria vs Wanita", height=450)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

col_vis3, col_anomali = st.columns(2)
with col_vis3:
    st.subheader("📈 Tren Total Kematian NCD Tahunan (Per Negara)")
    df_line = df_filtered.groupby(['Year', 'Country'])['Total Ncd Death'].sum().reset_index()
    fig_line = px.line(df_line, x='Year', y='Total Ncd Death', color='Country', markers=True, title="Tren Kematian NCD Tahunan", template="plotly_white", height=450)
    fig_line.update_xaxes(tickformat="d")
    st.plotly_chart(fig_line, use_container_width=True)

with col_anomali:
    st.subheader("🚨 Deteksi Anomali (Outlier)")
    df_filtered['Z_Score'] = (df_filtered['Total Ncd Death'] - df_filtered['Total Ncd Death'].mean()) / df_filtered['Total Ncd Death'].std()
    df_anomalies = df_filtered[(df_filtered['Z_Score'].abs() > 3)]
    
    if not df_anomalies.empty:
        st.error(f"Ditemukan *{len(df_anomalies)}* Anomali Kematian Signifikan!")
        st.dataframe(df_anomalies[['Country', 'Year', 'Sex', 'Total Ncd Death', 'Z_Score']].sort_values(by='Z_Score', ascending=False).head(10))
    else:
        st.success("Tidak ditemukan anomali signifikan (Z-Score > 3) dalam data yang difilter.")

st.markdown("---")

st.header("📉 Analisis Tren Halus: Kematian Aktual vs Rata-Rata Bergerak 3-Tahun")

df_ma_vis = df_filtered_growth.groupby(['Year', 'Country'])[['Total Ncd Death', 'Moving_Avg_3Y']].sum().reset_index()

df_ma_long = pd.melt(df_ma_vis, id_vars=['Year', 'Country'], 
                    value_vars=['Total Ncd Death', 'Moving_Avg_3Y'],
                    var_name='Metric', value_name='Value')

df_ma_long['Line_Type'] = df_ma_long['Metric'].apply(
    lambda x: 'Kematian Aktual' if x == 'Total Ncd Death' else 'Rata-Rata Bergerak'
)

fig_ma = px.line(df_ma_long, x='Year', y='Value', color='Country', line_dash='Line_Type',
                 labels={'Value': 'Total Kematian NCD', 'Year': 'Tahun', 'Country': 'Negara', 'Line_Type': 'Metrik'},
                 title='Perbandingan Kematian NCD Aktual vs Rata-Rata Bergerak 3 Tahun (Tergantung Filter)',
                 template="plotly_white", height=500)
fig_ma.update_xaxes(tickformat="d")


fig_ma.for_each_trace(lambda t: t.update(name=t.name.split(', ')[0].strip()))

fig_ma.update_layout(legend_title_text='Negara', 
                     legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5))

st.plotly_chart(fig_ma, use_container_width=True)

st.markdown("---")

is_indonesia_only = (len(selected_countries) == 1) and (selected_countries[0] == 'Indonesia')

if is_indonesia_only:
    st.header("🇮🇩 DEEP DIVE: Analisis Faktor Risiko dan Penyebab Kematian (Indonesia)")

    if not df_cause_death.empty:
        df_idn_ncd = df_cause_death[df_cause_death['Type Disease'] == 'Non Communicable Disease'].copy()
        
        if 'Rate Per 100K' in df_idn_ncd.columns and 'Cause Of Death' in df_idn_ncd.columns:
            df_idn_ncd_sorted = df_idn_ncd.sort_values(by='Rate Per 100K', ascending=False).head(15) 
            
            st.subheader("1. 15 Penyebab Utama Kematian NCD (Tingkat Kematian)")
            st.caption("Data ini bersifat agregat dan tidak dipengaruhi oleh filter Negara/Tahun/Jenis Kelamin.")
            
            
            fig_cause = px.bar(df_idn_ncd_sorted, x='Rate Per 100K', y='Cause Of Death', orientation='h',
                color='Rate Per 100K', labels={'Rate Per 100K': "Tingkat Kematian per 100K Penduduk"},
                title="15 Penyebab Utama Kematian NCD di Indonesia", template="plotly_white", height=600
            )
            fig_cause.update_layout(yaxis={'categoryorder': 'total ascending'}) 
            st.plotly_chart(fig_cause, use_container_width=True)
        else:
            st.error("Gagal memproses data penyebab kematian.")
    else:
        st.warning("Visualisasi Tingkat Kematian Penyebab Indonesia tidak dapat dimuat.")
    
    st.markdown("---")

    if not df_idn_factors.empty:
        st.subheader("2. Heatmap Korelasi Antar Faktor Risiko NCD (Provinsi)")
        st.caption("Menunjukkan seberapa kuat hubungan antar faktor risiko")
        st.markdown("""
        *🔍 Keterangan (Hubungan Antar Faktor):**
        * 🔴 **Merah Pekat (Mendekati +1):** Hubungan **Positif Kuat**. Contoh: Jika A naik, B juga cenderung naik.
        * ⚪ **Putih/Netral (Mendekati 0):** Hubungan **Sangat Lemah**. Tidak ada kaitan yang jelas.
        * 🔵 **Biru Pekat (Mendekati -1):** Hubungan **Negatif Kuat**. Contoh: Jika A naik, B cenderung turun.""")

        factor_only_cols = [
            'Smoke17M', 'Sbp17M', 'Bmi17T', 'Fpg17T', 'Sodium17T', 
            'Pmeat17M', 'Ambientpm17T', 'Sdi2016T', 'Fiber17T'
        ]
        cols_to_correlate = [col for col in factor_only_cols if col in df_idn_factors.columns]
        
        if len(cols_to_correlate) > 1:
            df_factor_corr = df_idn_factors[cols_to_correlate].corr(numeric_only=True)

            fig_factor_heatmap = px.imshow(
                df_factor_corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Korelasi Antar Faktor Risiko NCD di Provinsi Indonesia",
                zmin=-1, zmax=1 
            )
            fig_factor_heatmap.update_layout(height=700, width=700, yaxis={'title': ''}, xaxis={'title': ''})
            st.plotly_chart(fig_factor_heatmap, use_container_width=True)
        else:
            st.warning("Tidak cukup kolom faktor risiko yang relevan untuk membuat Heatmap Korelasi.")
else:
    st.info("Visualisasi *Deep Dive* spesifik Indonesia (Penyebab Kematian & Korelasi Faktor Risiko) akan muncul jika Anda **hanya memilih 'Indonesia'** pada filter Negara.")

st.markdown("---")