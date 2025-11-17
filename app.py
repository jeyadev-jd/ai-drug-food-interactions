# ============================================================
# COMPLETE DRUG-FOOD INTERACTION ANALYSIS WITH NGROK
# ============================================================

import subprocess
import sys
import time
import os

# STEP 1: Install core packages
print("📦 Installing packages...")
packages = ['streamlit', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("✅ Packages installed!\n")

# STEP 2: Upload CSV
print("📁 Upload your drug_food_interactions.csv file:")
from google.colab import files
uploaded = files.upload()

for filename in uploaded.keys():
    with open('drug_food_interactions.csv', 'wb') as f:
        f.write(uploaded[filename])
    print(f"\n✅ {filename} uploaded!\n")

# STEP 3: Create Streamlit app
print("📝 Creating app.py...")

app_code = '''import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Drug-Food Interaction", page_icon="💊", layout="wide")

st.markdown("""
<style>
.main {background-color: #f5f5f5;}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("💊 Drug–Food Interaction Analysis System")
st.markdown("### 🧬 AI-Powered NLP Classification + Knowledge Graph")

@st.cache_data
def load_data():
    df = pd.read_csv("drug_food_interactions.csv")
    df["food_interactions"] = df["food_interactions"].astype(str)
    return df

df = load_data()

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\\\\s]", " ", text).lower().strip()

df["clean_text"] = df["food_interactions"].apply(clean_text)
df["food_count"] = df["food_interactions"].apply(lambda x: len(x.split(",")))
df["has_restriction"] = df["food_interactions"].str.contains("avoid|contraindicated|danger|toxic", case=False, na=False)

def classify_interaction(text):
    text = text.lower()
    if any(k in text for k in ["contraindicated", "fatal", "severe toxicity"]):
        return "Contraindicated"
    if any(k in text for k in ["avoid", "not recommended", "dangerous"]):
        return "Avoid"
    if any(k in text for k in ["cyp3a4", "cyp2d6", "enzyme inhibition"]):
        return "Interacts with CYP enzymes"
    if any(k in text for k in ["metabolism", "metabolize", "alters metabolism"]):
        return "Alters metabolism"
    if any(k in text for k in ["enhance absorption", "increase absorption", "bioavailability"]):
        return "Enhances absorption"
    return "Safe"

df["interaction_category"] = df["clean_text"].apply(classify_interaction)

def label_severity(text):
    text = text.lower()
    if any(k in text for k in ["avoid", "contraindicated", "severe", "danger", "toxic"]):
        return "severe"
    elif any(k in text for k in ["increase", "reduce", "limit", "interfere"]):
        return "moderate"
    return "mild"

df["severity_label"] = df["clean_text"].apply(label_severity)

@st.cache_resource
def train_models(texts, labels):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1,2), min_df=2)
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42, stratify=labels)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted")
        }
    
    best = max(results, key=lambda x: results[x]["f1_score"])
    return vectorizer, X, results, best

vectorizer, tfidf_matrix, model_results, best_model = train_models(df["clean_text"], df["severity_label"])

st.sidebar.header("📊 Model Performance")
for name, metrics in model_results.items():
    st.sidebar.subheader(f"📌 {name}")
    st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    st.sidebar.metric("F1 Score", f"{metrics['f1_score']:.3f}")
    st.sidebar.markdown("---")

st.sidebar.success(f"🏆 Best: {best_model}")
st.sidebar.header("📈 Dataset Stats")
st.sidebar.metric("Total Records", len(df))
st.sidebar.metric("Unique Drugs", df['name'].nunique())
st.sidebar.metric("Avg Interactions", f"{df['food_count'].mean():.1f}")

st.header("🔍 Drug Search & Recommendation")
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("🔎 Enter condition/drug:", placeholder="e.g., aspirin, diabetes")
with col2:
    top_n = st.number_input("Results:", 5, 50, 10)

if st.button("🔍 Search", type="primary"):
    if query.strip():
        with st.spinner("Searching..."):
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            results = df.copy()
            results["similarity"] = similarities
            results = results.sort_values("similarity", ascending=False).head(top_n)
            results = results[results["similarity"] > 0.01]
            
            if len(results) > 0:
                st.success(f"✅ Found {len(results)} drugs")
                
                for _, row in results.iterrows():
                    icon = {"severe": "🔴", "moderate": "🟡", "mild": "🟢"}[row['severity_label']]
                    
                    with st.expander(f"{icon} {row['name']} ({row['similarity']:.2%})"):
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.markdown(f"**Reference:** {row['reference']}")
                            st.info(row['food_interactions'])
                        
                        with col_b:
                            st.metric("Category", row['interaction_category'])
                            st.metric("Severity", row['severity_label'].upper())
                            st.metric("Foods", row['food_count'])
            else:
                st.error("❌ No results")

st.header("🧠 Knowledge Graph (DFIKG)")
st.markdown("**Structure:** `Drug → Interaction Type → Food/Supplement`")

@st.cache_data
def build_kg(dataframe):
    kg_rows = []
    for _, row in dataframe.iterrows():
        foods = [f.strip() for f in row["food_interactions"].split(",") if f.strip()]
        for food in foods:
            kg_rows.append({
                "Drug": row["name"],
                "Interaction_Type": row["interaction_category"],
                "Severity": row["severity_label"],
                "Food_Supplement": food,
                "Reference": row["reference"]
            })
    return pd.DataFrame(kg_rows)

kg_df = build_kg(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Triplets", len(kg_df))
col2.metric("Drugs", kg_df['Drug'].nunique())
col3.metric("Foods", kg_df['Food_Supplement'].nunique())
col4.metric("Types", kg_df['Interaction_Type'].nunique())

st.subheader("🔎 Query Knowledge Graph")
filter_by = st.selectbox("Filter:", ["All", "By Drug", "By Food", "By Type", "By Severity"])

if filter_by == "By Drug":
    drug = st.selectbox("Drug:", sorted(kg_df['Drug'].unique()))
    filtered = kg_df[kg_df['Drug'] == drug]
elif filter_by == "By Food":
    food = st.selectbox("Food:", sorted(kg_df['Food_Supplement'].unique()))
    filtered = kg_df[kg_df['Food_Supplement'] == food]
elif filter_by == "By Type":
    itype = st.selectbox("Type:", sorted(kg_df['Interaction_Type'].unique()))
    filtered = kg_df[kg_df['Interaction_Type'] == itype]
elif filter_by == "By Severity":
    sev = st.selectbox("Severity:", ["severe", "moderate", "mild"])
    filtered = kg_df[kg_df['Severity'] == sev]
else:
    filtered = kg_df.head(100)

st.dataframe(filtered, use_container_width=True, height=400)

csv = kg_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download KG", csv, "drug_food_kg.csv", "text/csv")

st.header("📊 Visualizations")
tab1, tab2, tab3 = st.tabs(["Distribution", "Categories", "Severity"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        df["name"].value_counts().head(15).plot(kind="barh", ax=ax1, color='steelblue')
        plt.tight_layout()
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        df["food_count"].hist(bins=30, ax=ax2, color='coral')
        plt.tight_layout()
        st.pyplot(fig2)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        counts = df["interaction_category"].value_counts()
        ax3.pie(counts, labels=counts.index, autopct='%1.1f%%')
        st.pyplot(fig3)
    with col2:
        cat_df = pd.DataFrame({'Category': counts.index, 'Count': counts.values})
        st.dataframe(cat_df, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sev = df["severity_label"].value_counts()
        colors = {'severe': '#e74c3c', 'moderate': '#f39c12', 'mild': '#2ecc71'}
        color_list = [colors.get(x, '#95a5a6') for x in sev.index]
        sev.plot(kind='bar', ax=ax4, color=color_list)
        plt.tight_layout()
        st.pyplot(fig4)
    with col2:
        fig5, ax5 = plt.subplots(figsize=(6, 6))
        rest = df["has_restriction"].value_counts()
        ax5.pie(rest, labels=["Has Restriction", "No Restriction"], autopct='%1.1f%%')
        st.pyplot(fig5)

st.markdown("---")
st.markdown("<div style='text-align: center;'>💊 Drug-Food Interaction Analysis | ML & NLP</div>", unsafe_allow_html=True)
'''

with open('app.py', 'w') as f:
    f.write(app_code)

print("✅ app.py created!\n")

# STEP 4: Install ngrok manually (avoiding pyngrok timeout)
print("🔧 Installing ngrok...")

ngrok_path = "/usr/local/bin/ngrok"

if not os.path.exists(ngrok_path):
    # Download ngrok directly
    print("📥 Downloading ngrok v3...")
    subprocess.run([
        "wget", "-q", "-O", "ngrok.tgz",
        "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
    ], check=True)
    
    # Extract to /usr/local/bin
    subprocess.run(["tar", "xzf", "ngrok.tgz", "-C", "/usr/local/bin"], check=True)
    subprocess.run(["chmod", "+x", ngrok_path], check=True)
    
    print("✅ ngrok installed!\n")
else:
    print("✅ ngrok already installed!\n")

# STEP 5: Configure ngrok with your token
NGROK_TOKEN = "enter your ngrok token"

print("🔑 Configuring ngrok...")
subprocess.run([ngrok_path, "config", "add-authtoken", NGROK_TOKEN], check=True)
print("✅ Token configured!\n")

# STEP 6: Start Streamlit
print("🚀 Starting Streamlit...")
subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

print("⏳ Waiting for Streamlit to start...")
time.sleep(12)
print("✅ Streamlit started!\n")

# STEP 7: Start ngrok tunnel
print("🌐 Creating ngrok tunnel...\n")

ngrok_process = subprocess.Popen(
    [ngrok_path, "http", "8501", "--log=stdout"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

time.sleep(4)

# Get URL from ngrok API
import requests
try:
    response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
    data = response.json()
    
    if 'tunnels' in data and len(data['tunnels']) > 0:
        url = data['tunnels'][0]['public_url']
        
        print("="*70)
        print("✅ YOUR APP IS LIVE!")
        print("="*70)
        print(f"\n🌐 Public URL: {url}")
        print(f"\n📱 Also works with: {url.replace('http://', 'https://')}")
        print("\n"+ "="*70)
        print("\n📌 INSTRUCTIONS:")
        print("   1. Click the URL above")
        print("   2. Your app will open - NO PASSWORD NEEDED!")
        print("   3. Share this link with anyone")
        print("\n⚠️  IMPORTANT:")
        print("   • Keep this cell running")
        print("   • Free tier: 2 hours per session")
        print("   • Link expires when you stop the cell")
        print("\n🎉 Enjoy your app!")
        print("="*70)
    else:
        print("⚠️  Tunnel created but couldn't get URL automatically")
        print("📍 Check: http://localhost:4040")
        
except Exception as e:
    print(f"⚠️  Couldn't fetch URL from API: {e}")
    print("📍 Manual check: http://localhost:4040")
    print("    Or wait 10 seconds and check ngrok dashboard")

# Keep running
print("\n⚡ App is running... Press Ctrl+C to stop (or stop the cell)")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Stopping app...")
    subprocess.run(["pkill", "-9", "ngrok"])
    subprocess.run(["pkill", "-9", "streamlit"])

