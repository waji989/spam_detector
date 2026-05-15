import streamlit as st
import pandas as pd
from model import load_models, predict

st.set_page_config(page_title="Spam Detector", layout="centered")

st.title("📧 AI Spam Detection System")

vectorizer, nb_model, lr_model = load_models()

# ---------------- TEXT INPUT ----------------
st.subheader("✍️ Manual Input")
text = st.text_area("Enter Email Content", height=150)

if st.button("Analyze Text"):
    if text.strip() == "":
        st.warning("Enter text")
    else:
        nb, lr = predict(text, vectorizer, nb_model, lr_model)
        avg = (nb + lr) / 2
        
        st.success("Analysis Complete")
        st.write(f"Naive Bayes: {nb:.2f}%")
        st.write(f"Logistic Regression: {lr:.2f}%")
        
        if avg > 50:
            st.error("🚨 SPAM")
        else:
            st.success("✅ NOT SPAM")

# ---------------- FILE UPLOAD ----------------
st.subheader("📁 Upload Email File")

uploaded_file = st.file_uploader(
    "Upload .txt or .csv file",
    type=["txt", "csv"]
)

if uploaded_file is not None:

    # ---- TXT FILE ----
    if uploaded_file.name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8")
        
        st.text_area("File Content", content, height=150)
        
        nb, lr = predict(content, vectorizer, nb_model, lr_model)
        avg = (nb + lr) / 2
        
        st.subheader("Result")
        st.write(f"Naive Bayes: {nb:.2f}%")
        st.write(f"Logistic Regression: {lr:.2f}%")
        
        if avg > 50:
            st.error("🚨 SPAM")
        else:
            st.success("✅ NOT SPAM")

    # ---- CSV FILE ----
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

        # Assume column name is 'message'
        if 'message' not in df.columns:
            st.error("CSV must contain a 'message' column")
        else:
            results = []

            for msg in df['message']:
                nb, lr = predict(str(msg), vectorizer, nb_model, lr_model)
                avg = (nb + lr) / 2
                label = "SPAM" if avg > 50 else "NOT SPAM"

                results.append({
                    "message": msg,
                    "NB %": round(nb, 2),
                    "LR %": round(lr, 2),
                    "Result": label
                })

            result_df = pd.DataFrame(results)

            st.subheader("Prediction Results")
            st.dataframe(result_df)

            # Download option
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results",
                csv,
                "spam_results.csv",
                "text/csv"
            )