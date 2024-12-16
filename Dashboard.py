import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go

# =========================
# Load Data Function
# =========================
@st.cache_data
def load_data():
    # Charger les données
    df = pd.read_csv("Cleaned.csv")
    return df

# Load data
st.title("Fraud Detection - Data Insights Dashboard")
st.write("### Explore relationships between features and fraud status")
df = load_data()

# =========================
# Section 1: Overview of the Data
# =========================
st.subheader("Dataset Overview")
st.write(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
st.write(df.head())

# =========================
# Section 2: Feature Insights - Numerical Features with Plotly Distribution
# =========================
st.subheader("Feature Distribution: Compare Fraudulent vs. Legitimate Transactions")

# Select feature for distribution plot
selected_numerical_feature = st.selectbox(
    "Select a numerical feature for distribution plot",
    ['session_duration', 'transaction_amount', 'time_spent_on_payment_page']
)

# Filter data for fraud (class_1) and legitimate (class_0)
class_0 = df.loc[df['flag'] == 0][selected_numerical_feature]
class_1 = df.loc[df['flag'] == 1][selected_numerical_feature]

# Create the distribution plot
hist_data = [class_0, class_1]
group_labels = ['Legitimate', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig.update_layout(
    title=f"Density Plot for {selected_numerical_feature}",
    xaxis_title=selected_numerical_feature,
    yaxis_title="Density",
    template="plotly"
)

# Display the figure in Streamlit
st.plotly_chart(fig)

# =========================
# Section 3: Categorical Features Insights
# =========================
st.subheader("Categorical Feature Analysis")

categorical_features = [
    'transaction_status', 'transaction_type', 
    'customer_ip_location', 'payment_method', 'login_status', 'visit_origin', 'device_type'
]

# Choose categorical feature
selected_cat_feature = st.selectbox("Select a categorical feature", categorical_features)

# Plot counts of the categorical feature by fraud status
fig_cat = go.Figure()
fig_cat.add_trace(go.Bar(
    x=df[selected_cat_feature].value_counts().index,
    y=df.loc[df['flag'] == 0][selected_cat_feature].value_counts(),
    name="Legitimate",
    marker_color="blue"
))
fig_cat.add_trace(go.Bar(
    x=df[selected_cat_feature].value_counts().index,
    y=df.loc[df['flag'] == 1][selected_cat_feature].value_counts(),
    name="Fraud",
    marker_color="red"
))

fig_cat.update_layout(
    title=f"{selected_cat_feature} Distribution by Fraud Status",
    xaxis_title=selected_cat_feature,
    yaxis_title="Count",
    barmode='group'
)

st.plotly_chart(fig_cat)

# =========================
# Section 4: Correlation Heatmap
# =========================
st.subheader("Correlation Heatmap")

# Compute correlation matrix
corr = df.select_dtypes(include=['float64', 'int64']).corr()

# Plot heatmap
fig_corr = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale="Viridis",
    zmin=-1, zmax=1
))

fig_corr.update_layout(
    title="Feature Correlation Heatmap",
    xaxis_title="Features",
    yaxis_title="Features"
)

st.plotly_chart(fig_corr)

