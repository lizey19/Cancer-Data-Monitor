import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title("🩺 Cancer Patient Data Monitor")

# File upload
uploaded_file = st.file_uploader("Upload Cancer Dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        fig = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

    # Risk factor analysis
    st.subheader("Risk Factor Analysis")
    col = st.selectbox("Select a medical variable", df.columns)
    if pd.api.types.is_numeric_dtype(df[col]):
        fig = px.histogram(df, x=col, nbins=30, marginal="box", title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.bar(df[col].value_counts().reset_index(), x="index", y=col, title=f"Count of {col}")
        st.plotly_chart(fig, use_container_width=True)

    # Outlier detection
    st.subheader("🚨 Outlier Detection (IQR Method)")
    for col in numeric_df.columns:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        st.write(f"{col}: {len(outliers)} outliers")

    # 3D scatter
    if len(numeric_df.columns) >= 3:
        st.subheader("3D Visualization")
        cols = st.multiselect("Pick 3 numeric features", numeric_df.columns, numeric_df.columns[:3])
        if len(cols) == 3:
            fig3d = px.scatter_3d(df, x=cols[0], y=cols[1], z=cols[2], color=df[cols[0]])
            st.plotly_chart(fig3d, use_container_width=True)
