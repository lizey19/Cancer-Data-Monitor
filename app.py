
    import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

st.title("🩺 Cancer Patient Data Monitor (Large Dataset Friendly)")

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

    # Sample size control for big datasets
    st.sidebar.header("⚙️ Sampling & Settings")
    sample_size = st.sidebar.slider("Select sample size for plots", min_value=500, max_value=min(10000, len(df)), value=2000, step=500)
    df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

    st.sidebar.write(f"Using {len(df_sample)} rows out of {len(df)} for plotting.")

    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap (Sampled Data)")
    numeric_df = df_sample.select_dtypes(include=np.number)
    if not numeric_df.empty:
        fig_corr = px.imshow(numeric_df.corr(), text_auto=False, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

    # Column analysis with sampling
    st.subheader("📌 Column Analysis")
    column = st.selectbox("Select a column", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(df_sample, x=column, nbins=40, marginal="box", title=f"Distribution of {column}")
        st.plotly_chart(fig, use_container_width=True)

        # Violin plot
        fig_violin = px.violin(df_sample, y=column, box=True, points="all", title=f"Violin Plot of {column}")
        st.plotly_chart(fig_violin, use_container_width=True)

        # KDE Density Plot
        fig_kde = ff.create_distplot([df_sample[column].dropna()], [column], show_hist=False)
        st.plotly_chart(fig_kde, use_container_width=True)

    else:
        top_n = st.slider("Show top N categories", min_value=5, max_value=30, value=10)
        value_counts = df[column].value_counts().nlargest(top_n).reset_index()
        fig = px.bar(value_counts, x="index", y=column, title=f"Top {top_n} Categories in {column}")
        st.plotly_chart(fig, use_container_width=True)

        # Pie chart
        fig_pie = px.pie(value_counts, names="index", values=column, title=f"Distribution of {column}")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Outlier detection (sampled)
    st.subheader("🚨 Outlier Detection (IQR Method, Sampled Data)")
    for col in numeric_df.columns:
        Q1, Q3 = df_sample[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = df_sample[(df_sample[col] < Q1 - 1.5*IQR) | (df_sample[col] > Q3 + 1.5*IQR)]
        st.write(f"{col}: {len(outliers)} outliers (out of {len(df_sample)})")

    # 3D scatter with sampling
    if len(numeric_df.columns) >= 3:
        st.subheader("🌐 3D Visualization (Sampled Data)")
        cols = st.multiselect("Pick 3 numeric features", numeric_df.columns, numeric_df.columns[:3])
        if len(cols) == 3:
            fig3d = px.scatter_3d(df_sample, x=cols[0], y=cols[1], z=cols[2], color=cols[0], opacity=0.6)
            st.plotly_chart(fig3d, use_container_width=True)

    # Pairwise scatter matrix
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        st.subheader("🔗 Pairwise Scatter Matrix")
        fig_matrix = px.scatter_matrix(df_sample, dimensions=numeric_df.columns[:5], title="Scatter Matrix of Numeric Features")
        st.plotly_chart(fig_matrix, use_container_width=True)

    # Boxplots for multiple features
    if not numeric_df.empty:
        st.subheader("📦 Multiple Feature Boxplots")
        selected_cols = st.multiselect("Select numeric columns for boxplots", numeric_df.columns, numeric_df.columns[:3])
        if selected_cols:
            fig_box = px.box(df_sample[selected_cols], points="all")
            st.plotly_chart(fig_box, use_container_width=True)
