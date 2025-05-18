import streamlit as st
from utils import generate_text
import json
import csv
import io

st.set_page_config(page_title="Generative Text UI", page_icon="🖊️", layout="wide")

st.title("🧠 Generative Text Model")
st.markdown("### 📘 Create text using GPT-2 on specific topics")
st.markdown("---")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_length = st.slider("Max Length", 50, 500, 150)
    top_k = st.slider("Top-k Sampling", 0, 100, 50)
    top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.95)
    export_format = st.radio("Export Format", ["TXT", "CSV", "JSON"])
    st.markdown("---")

# Prompt input
prompt = st.text_area("Enter your prompt here:", height=150)

# Generate
if st.button("Generate Text"):
    if not prompt.strip():
        st.warning("🚨 Please enter a prompt.")
    else:
        with st.spinner("Generating text..."):
            output = generate_text(prompt, temperature, max_length, top_k, top_p)
            st.success("Text generated successfully!")
            st.markdown("### 🔍 Output")
            st.write(output)

            # Export
            if export_format == "TXT":
                st.download_button("Download TXT", output, file_name="output.txt")
            elif export_format == "CSV":
                csv_buf = io.StringIO()
                csv.writer(csv_buf).writerow(["Generated Text"])
                csv.writer(csv_buf).writerow([output])
                st.download_button("Download CSV", csv_buf.getvalue(), file_name="output.csv")
            elif export_format == "JSON":
                json_str = json.dumps({"generated_text": output}, indent=2)
                st.download_button("Download JSON", json_str, file_name="output.json")

st.markdown("---")
st.caption("Built with ❤️ by CodTech Intern")
