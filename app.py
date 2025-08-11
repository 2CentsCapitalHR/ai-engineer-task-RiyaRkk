import streamlit as st
import tempfile
import os

# Import refactored wrappers from your existing scripts
from Step1 import classify_document
from missing_docs_check import find_missing_documents
from ingest_adm import load_or_build_vector_db
from red_flag_check import check_red_flags
from comment_adder import add_comments_to_doc

st.set_page_config(page_title="ADGM Compliance Agent", layout="wide")
st.title("📄 ADGM Compliance Agent")
st.markdown(
    "Upload your legal/corporate document (DOCX or PDF) to check compliance with ADGM regulations."
)

uploaded_file = st.file_uploader("Upload a DOCX or PDF", type=["docx", "pdf"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Step 1: Classification
        with st.spinner("Step 1: Classifying document..."):
            try:
                doc_type = classify_document(input_path)
                st.success(f"Document classified as: **{doc_type}**")
            except Exception as e:
                st.error(f"Classification failed: {e}")
                st.stop()

        # Step 2: Check for missing documents
        with st.spinner("Step 2: Checking for missing documents..."):
            try:
                missing_items = find_missing_documents(input_path, doc_type)
                st.write(f"Missing documents/items found: **{len(missing_items)}**")
            except Exception as e:
                st.error(f"Missing documents check failed: {e}")
                st.stop()

        # Step 3: Load or build ADGM rules vector DB
        with st.spinner("Step 3: Loading ADGM rules database..."):
            try:
                load_or_build_vector_db()
                st.success("ADGM rules loaded and ready.")
            except Exception as e:
                st.error(f"Failed to load ADGM rules database: {e}")
                st.stop()

        # Step 4: Red flag detection
        with st.spinner("Step 4: Running red flag detection..."):
            try:
                red_flags, json_path, tsv_path = check_red_flags(input_path)
                st.write(f"Red flags detected: **{len(red_flags)}**")
            except Exception as e:
                st.error(f"Red flag detection failed: {e}")
                st.stop()

        # Step 5: Add comments (annotations) to document
        with st.spinner("Step 5: Adding comments to document..."):
            try:
                annotated_path = os.path.join(tmpdir, "annotated_output.docx")
                add_comments_to_doc(input_path, tsv_path, annotated_path)
                st.success("Annotated document created.")
            except Exception as e:
                st.error(f"Adding comments failed: {e}")
                st.stop()

        # Step 6: Display summary and download option
        st.subheader("📊 Summary Report")
        summary = {
            "Document Type": doc_type,
            "Missing Items": missing_items,
            "Red Flags": red_flags,
        }
        st.json(summary)

        st.download_button(
            label="📥 Download Annotated Document",
            data=open(annotated_path, "rb").read(),
            file_name="annotated_document.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
