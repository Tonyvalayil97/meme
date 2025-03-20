import streamlit as st
import main  # Import functions from main.py

st.title("📄 Invoice Data Extraction App")

# Upload multiple PDF files
uploaded_files = st.file_uploader("Upload Invoice PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success("✅ Files uploaded successfully! Extracting data...")

    # Process invoices and generate Excel file
    excel_file, df = main.process_invoices(uploaded_files)

    # Display extracted data in Streamlit
    st.write("### 📊 Extracted Invoice Data")
    st.dataframe(df)

    # Download Excel file
    with open(excel_file, "rb") as f:
        st.download_button(
            label="📥 Download Extracted Data as Excel",
            data=f,
            file_name="extracted_invoices.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.success("✅ Data extraction complete!")
