"""
ITviec Company Analyzer – Streamlit Dashboard  (two-page edition)
-----------------------------------------------------------------
"""
# ╔═ IMPORTS ════════════════════════════════════════════════════════════════╗
import os, re, joblib, numpy as np, pandas as pd, streamlit as st
import streamlit.components.v1 as components
from gensim.models.doc2vec import Doc2Vec

# ╔═ PAGE CONFIG ════════════════════════════════════════════════════════════╗
st.set_page_config(page_title="ITViec Company Analyzer", layout="wide")

# ╔═ CONSTANTS ################################################################
ARTIFACT_PATH = "data/dashboard_artifacts"
VIZ_PATH = os.path.join(ARTIFACT_PATH, "viz")
SUBSCORE_COLS = [
    "Salary & benefits", "Training & learning", "Management cares about me",
    "Culture & fun", "Office & workspace"
]

# ╔═ HELPERS ##################################################################
def clean_text(q:str) -> str:
    q = re.sub(r"[^\w\s]", " ", q.lower())
    q = re.sub(r"\d+", " ", q)
    return re.sub(r"\s+", " ", q).strip()

def imp(model, feats):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        co = np.abs(model.coef_)
        return co.mean(0) if co.ndim > 1 else co
    return np.zeros(len(feats))

@st.cache_data(show_spinner="Loading data & models …")
def load_artifacts(path):
    feats = pd.read_csv(os.path.join(path, "final_features.csv"))
    clf   = joblib.load(os.path.join(path, "classification_model.joblib"))
    feat_cols = list(joblib.load(os.path.join(path, "classification_features.joblib")))
    d2v   = Doc2Vec.load(os.path.join(path, "doc2vec.model"))
    return feats, clf, feat_cols, d2v

try:
    features_df, clf_model, clf_cols, d2v_model = load_artifacts(ARTIFACT_PATH)
except FileNotFoundError:
    st.error("Artifacts not found. Run the notebook first.")
    st.stop()

IMP_DF = (
    pd.DataFrame({"feature": clf_cols,
                  "importance": imp(clf_model, clf_cols)})
      .sort_values("importance", ascending=False)
      .reset_index(drop=True)
)

# ╔═ SIDEBAR NAVIGATION #######################################################
page = st.sidebar.radio(
    "📑 Select a page",
    ("🔍 Search & Evaluate", "📊 Project Results")
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
**Yêu cầu 1:** Dựa trên những thông tin từ các công ty đăng trên ITViec để gợi ý các công ty tương tự dựa trên nội dung mô tả.).

**Yêu cầu 2:** Dựa trên những thông tin từ review của ứng viên/ nhân viên đăng trên ITViec để dự đoán khả năng “Recommend” công ty..
""")

st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Group information:**")
st.sidebar.write("1. Yan Shih Siang")  
st.sidebar.write("• Email: ssyan110@gmail.com")  
st.sidebar.write("2. Phạm Tiến Triển ")  
st.sidebar.write("• Email: Phamtrien0211@gmail.com")



if page.startswith("🔍 Search & Evaluate"):

    st.title("👩‍💻 ITViec Company Analyzer – Search & Evaluate")

    # ---- search controls --------------------------------------------------
    col_q, col_sliders = st.columns([2,3], gap="large")

    with col_q:
        query = st.text_input("Company name or keywords", placeholder="e.g. FPT, fintech high salary")
        if st.button("Search"):
            st.session_state.search_active = True
            st.session_state.last_query   = query

    with col_sliders:
        st.write("**Set your minimum ratings:**")
        prefs={}
        c1,c2,c3=st.columns(3)
        with c1:
            prefs["Salary & benefits"] = st.slider("Salary",1.,5.,1.,.1)
            prefs["Training & learning"]= st.slider("Training",1.,5.,1.,.1)
        with c2:
            prefs["Management cares about me"]= st.slider("Management",1.,5.,1.,.1)
            prefs["Culture & fun"]= st.slider("Culture",1.,5.,1.,.1)
        with c3:
            prefs["Office & workspace"]= st.slider("Workspace",1.,5.,1.,.1)

    # ---- Search engine ----------------------------------------------------
    if st.session_state.get("search_active", False):

        q = st.session_state.get("last_query","")
        df = features_df.copy()

        # pin direct name matches
        pin = pd.DataFrame()
        if q:
            m = df["Company Name"].str.lower().str.contains(q.lower())
            pin, df = df[m], df[~m]

        # rating filters
        for col,t in prefs.items(): df = df[df[col]>=t]

        # preference score
        df["pref_match"]=df[list(prefs)].mean(1)
        if not pin.empty: pin["pref_match"]=pin[list(prefs)].mean(1)

        # semantic search
        if q:
            sims = d2v_model.dv.most_similar([d2v_model.infer_vector(clean_text(q).split())],
                                             topn=len(d2v_model.dv))
            sem = (pd.DataFrame(sims, columns=["idx","kw_sim"])
                     .merge(df.reset_index(), left_on="idx", right_on="index"))
            sem["kw_sim"]*=100
            sem=sem.sort_values("kw_sim", ascending=False)
            pin["kw_sim"]=100.0
            final = pd.concat([pin, sem]).drop_duplicates("id")
        else:
            final = df.sort_values("pref_match", ascending=False)

        st.markdown("---")
        st.subheader(f"🔎  {len(final)} matching companies")
        with st.expander("ℹ️  Badge legend"):
            st.markdown(
                "✅ ≥60 % confident good &nbsp;|&nbsp; 🟡 30-60 % uncertain "
                "| ⚠️ <30 % potential issues &nbsp;|&nbsp; ⚪ <5 reviews"
            )

        for _,row in final.head(15).iterrows():
            proba = float(clf_model.predict_proba(row[clf_cols].values.reshape(1,-1))[0,1])
            with st.expander(row["Company Name"]):
                c1,c2,c3 = st.columns(3)

                with c1:
                    st.write(f"**Industry:** {row.get('Company industry','–')}")
                    st.write(f"**Size:** {row.get('Company size','–')}")
                    if pd.notna(row.get("Href")):
                        st.markdown(f"[ITviec page]({row['Href']})")
                    st.metric("Preference", f"{row['pref_match']:.2f}/5")
                    if "kw_sim" in row: st.metric("Keyword", f"{row['kw_sim']:.0f}%")

                with c2:
                    if proba>=0.6:   st.success(f"✅ {proba:.0%} might be a good fit")
                    elif proba>=0.4: st.warning(f"🟡 {proba:.0%} uncertain")
                    else:            st.error  (f"⚠️ {proba:.0%} re-consider")
                    if row.get("few_reviews",False):
                        st.info("⚪ Few reviews")

                with c3:
                    st.write("**Ratings:**")
                    for s in SUBSCORE_COLS:
                        if s in row: st.text(f"{s}: {row[s]:.2f}")

    else:
        st.info("Enter a query and press **Search** to begin.")

elif page == "📊 Project Results":
    import os

    OUTPUTS_PATH = os.path.join(os.path.dirname(__file__), "outputs")

    CUSTOM_CAPTIONS = {
        "roc_curve.png": "ROC Curve: All Models (Sklearn & PySpark)",
        "model_performance_metrics.png": "Model Performance Comparison",
        "recommendation_coverage_visualization.png": "Recommendation Coverage",
        "confusion_matrix_RandomForest.png": "Confusion Matrix: Random Forest",
        "confusion_matrix_LogisticRegression.png": "Confusion Matrix: Logistic Regression",
        "confusion_matrix_KNN.png": "Confusion Matrix: K-Nearest Neighbors",
        "recommended_result_by_ID_1.png": "Recommendation Result: Company ID 1",
        "recommended_result_by_ID_2.png": "Recommendation Result: Company ID 2",
        "recommended_result_by_ID_3.png": "Recommendation Result: Company ID 3",
        "recommended_result_by__keyword_1.png": "Recommendation by Keyword 1",
        "recommended_result_by__keyword_2.png": "Recommendation by Keyword 2",
        "recommended_result_by__keyword_3.png": "Recommendation by Keyword 3",
    }

    INSIGHTS = {
        "confusion_matrix_LogisticRegression.png": (
            "Best at finding \"Not Recommended\" companies: "
            "Fewest false positives, but misses more \"Recommended\" cases than other models."
        ),
        "confusion_matrix_KNN.png": (
            "Best at detecting \"Recommended\" companies: "
            "Very few missed recommendations, but tends to over-recommend and makes more mistakes for \"Not Recommended.\""
        ),
        "confusion_matrix_RandomForest.png": (
            "Most balanced model: Low error for both classes—reliable at identifying both \"Recommended\" and \"Not Recommended\" companies."
        ),
        "recommendation_coverage_visualization.png": (
            "Hybrid approach works best: LightFM is best for similar companies; "
            "Doc2Vec is best for keyword matches. Using both gives the most relevant recommendations."
        ),
        "model_performance_metrics.png": (
            "Random Forest (PySpark) is the top performer for balanced precision and recall. "
            "Random Forest (Sklearn) achieves the highest recall. Logistic Regression models are solid and interpretable. "
            "KNN has lower precision, leading to more false positives."
        ),
        "roc_curve.png": (
            "Random Forest (Sklearn) has near-perfect class separation (AUC=0.99). "
            "Logistic Regression (Sklearn) also performs excellently. "
            "Random Forest (PySpark) is strong and reliable. "
            "KNN is weaker at separating classes. "
            "Logistic Regression (PySpark) is decent but has the lowest AUC."
        ),
        "recommended_result_by_ID_1.png": (
            "LightFM is the most powerful when users want to find *similar companies* based on a specific company (e.g., “show me companies like FPT Software”).  \n"
            "LightFM combines both company profiles and user-review patterns, providing the most diverse and accurate alternatives."
        ),
        "recommended_result_by_ID_2.png": (
            "LightFM is the most powerful when users want to find *similar companies* based on a specific company (e.g., “show me companies like FPT Software”).  \n"
            "LightFM combines both company profiles and user-review patterns, providing the most diverse and accurate alternatives."
        ),
        "recommended_result_by_ID_3.png": (
            "LightFM is the most powerful when users want to find *similar companies* based on a specific company (e.g., “show me companies like FPT Software”).  \n"
            "LightFM combines both company profiles and user-review patterns, providing the most diverse and accurate alternatives."
        ),
        "recommended_result_by_keyword_1.png": (
            "Doc2Vec excels at *semantic keyword search* (e.g., “high salary”, “good work-life balance”).  \n"
            "It surfaces companies whose descriptions and reviews best match the meaning of your search terms."
        ),
        "recommended_result_by_keyword_2.png": (
            "Doc2Vec excels at *semantic keyword search* (e.g., “high salary”, “good work-life balance”).  \n"
            "It surfaces companies whose descriptions and reviews best match the meaning of your search terms."
        ),
        "recommended_result_by_keyword_3.png": (
            "Doc2Vec excels at *semantic keyword search* (e.g., “high salary”, “good work-life balance”).  \n"
            "It surfaces companies whose descriptions and reviews best match the meaning of your search terms."
        ),    
    }

    def get_plot_type(f):
        if f.startswith("confusion_matrix"):
            return "Confusion Matrix"
        elif f.startswith("roc_curve"):
            return "ROC Curve"
        elif f.startswith("model_performance"):
            return "Model Performance"
        elif f.startswith("recommendation_coverage"):
            return "Recommendation Coverage"
        elif f.startswith("recommended_result_by_ID"):
            return "Recommended Result by ID"
        elif f.startswith("recommended_result_by_keyword"):
            return "Recommended Result by Keyword"
        else:
            return "Other"

    def get_caption(file):
        return CUSTOM_CAPTIONS.get(file, file.replace('_', ' ').replace('.png', '').title())

    def get_insight(file):
        return INSIGHTS.get(file, "")

    st.title("📊 Project Results & Visualizations")

    files = [f for f in os.listdir(OUTPUTS_PATH) if f.endswith(".png")]
    if not files:
        st.warning("No output images found in outputs folder.")
        st.stop()

    plot_types = sorted(set(get_plot_type(f) for f in files))
    selected_type = st.selectbox("Select plot type", plot_types)
    filtered_files = [f for f in files if get_plot_type(f) == selected_type]
    if not filtered_files:
        st.info("No images found for this plot type.")
    else:
        for img_file in filtered_files:
            file_path = os.path.join(OUTPUTS_PATH, img_file)
            caption = get_caption(img_file)
            insight = get_insight(img_file)
            st.image(file_path, caption=caption)
            if insight:
                st.markdown(
                    f"<div style='margin-bottom:40px; font-size: 1.08rem; color:#4ec9b0;'>"
                    f"<b>Insight:</b> {insight}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown("<br>", unsafe_allow_html=True)
