import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import calendar
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

st.set_page_config(page_title="Advanced Clustering Tool", layout="wide")

st.title("📊 Advanced Load Profile Clustering Tool")

# -------------------------------
# Smart datetime parser
# -------------------------------
def parse_datetime_column(df, col):
    series = df[col].astype(str).str.strip()

    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)

    if dt.isna().mean() > 0.3:
        dt = pd.to_datetime(series, errors="coerce", format="mixed")

    if dt.isna().mean() > 0.5:
        try:
            dt = pd.to_datetime(
                series.astype(float),
                unit='d',
                origin='1899-12-30',
                errors='coerce'
            )
        except:
            pass

    return dt


# -------------------------------
# Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(data.head())

    # -------------------------------
    # Datetime handling
    # -------------------------------
    datetime_col = st.selectbox("Select datetime column", data.columns)

    parsed_dt = parse_datetime_column(data, datetime_col)

    invalid_count = parsed_dt.isna().sum()
    if invalid_count > 0:
        st.warning(f"{invalid_count} invalid datetime rows removed")

    data = data.loc[~parsed_dt.isna()].copy()
    data["Datetime"] = parsed_dt.dropna()

    data = data.set_index("Datetime").sort_index()
    data = data[~data.index.duplicated()]

    st.success("Datetime parsed successfully")

    # -------------------------------
    # Select variables
    # -------------------------------
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    selected_cols = st.multiselect(
        "Select variables for clustering",
        numeric_cols,
        default=numeric_cols[:2]
    )

    if len(selected_cols) == 0:
        st.warning("Please select at least one column")
        st.stop()

    df = data[selected_cols]

    # -------------------------------
    # Resample hourly
    # -------------------------------
    df_hourly = df.resample("h").mean()

    df_hourly["date"] = df_hourly.index.date
    df_hourly["hour"] = df_hourly.index.hour

    pivot_df = df_hourly.pivot_table(
        index="date",
        columns="hour",
        values=selected_cols
    )

    # Flatten columns
    pivot_df.columns = [f"{c[0]}_h{c[1]}" for c in pivot_df.columns]

    # -------------------------------
    # Handle missing values
    # -------------------------------
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(pivot_df)

    # -------------------------------
    # Elbow + Silhouette
    # -------------------------------
    st.subheader("📈 Optimal Cluster Detection")

    max_k = 10
    inertia = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)

    for k_test in K_range:
        model = KMeans(n_clusters=k_test, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        inertia.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(K_range, inertia, marker='o')
        ax1.set_title("Elbow Method")
        ax1.set_xlabel("Clusters")
        ax1.set_ylabel("Inertia")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(K_range, silhouette_scores, marker='o')
        ax2.set_title("Silhouette Score")
        ax2.set_xlabel("Clusters")
        ax2.set_ylabel("Score")
        st.pyplot(fig2)

    optimal_k = K_range[np.argmax(silhouette_scores)]
    st.success(f"Suggested optimal clusters: {optimal_k}")

    # -------------------------------
    # Cluster slider
    # -------------------------------
    k = st.slider("Select number of clusters", 2, max_k, int(optimal_k))

    # -------------------------------
    # Final clustering
    # -------------------------------
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(X)

    pivot_df["Cluster"] = clusters

    st.subheader("Clustered Data")
    st.write(pivot_df.head())

    # -------------------------------
    # CONSISTENT COLORS
    # -------------------------------
    base_cmap = plt.get_cmap("tab10")
    cluster_colors = [base_cmap(i) for i in range(k)]

    cmap = ListedColormap(cluster_colors)
    norm = BoundaryNorm(np.arange(-0.5, k + 0.5, 1), cmap.N)

    # -------------------------------
    # Cluster profiles
    # -------------------------------
    st.subheader("Cluster Profiles")

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    hours = list(range(24))

    for c in range(k):
        cluster_data = pivot_df[pivot_df["Cluster"] == c].drop("Cluster", axis=1)

        if not cluster_data.empty:
            mean_profile = cluster_data.mean().values
            ax3.plot(
                hours,
                mean_profile,
                marker='o',
                color=cluster_colors[c],
                label=f"Cluster {c}"
            )

    ax3.set_title("Average Daily Profiles")
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Value")
    ax3.set_xticks(hours)
    ax3.set_xlim(0, 23)
    ax3.legend()

    st.pyplot(fig3)

    # -------------------------------
    # Calendar plot (MULTI-YEAR)
    # -------------------------------
    st.subheader("📅 Calendar Plot Showing Cluster Distribution (Full Period)")

    cluster_series = pivot_df["Cluster"].copy()
    cluster_series.index = pd.to_datetime(cluster_series.index)

    years = sorted(cluster_series.index.year.unique())

    for year in years:

        st.markdown(f"### Year {year}")

        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        axes = axes.flatten()

        for month in range(1, 13):
            ax = axes[month - 1]

            cal = calendar.monthcalendar(year, month)
            cal_array = np.array(cal)

            plot_data = np.full(cal_array.shape, np.nan)

            for i in range(cal_array.shape[0]):
                for j in range(7):
                    day = cal_array[i, j]
                    if day != 0:
                        date = pd.Timestamp(year=year, month=month, day=day)

                        if date in cluster_series.index:
                            plot_data[i, j] = cluster_series.loc[date]

            ax.imshow(plot_data, cmap=cmap, norm=norm)

            ax.set_title(calendar.month_name[month], fontsize=11, pad=10)
            ax.set_xticks(range(7))
            ax.set_xticklabels(
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                fontsize=7
            )
            ax.set_yticks([])

            for i in range(cal_array.shape[0]):
                for j in range(7):
                    day = cal_array[i, j]
                    if day != 0:
                        ax.text(j, i, str(day), ha='center', va='center', fontsize=7)

        plt.subplots_adjust(hspace=0.4, wspace=0.2)

        patches = [
            mpatches.Patch(color=cluster_colors[i], label=f"Cluster {i}")
            for i in range(k)
        ]

        fig.legend(handles=patches, loc="lower center", ncol=k, frameon=False)

        st.pyplot(fig)

    # -------------------------------
    # Downloads
    # -------------------------------
    st.subheader("📥 Export Results")

    csv = pivot_df.to_csv().encode("utf-8")
    st.download_button("Download CSV", csv, "clusters.csv")

    fig3.savefig("cluster_profiles.png")

    with open("cluster_profiles.png", "rb") as f:
        st.download_button("Download Cluster Plot", f, "cluster_profiles.png")

else:
    st.info("Upload a CSV file to start")
