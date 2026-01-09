import os
import streamlit as st
from rag_doc import process_document, answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("Document QA with RAG")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join(working_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    process_document(uploaded_file.name)  # Process once
    st.info("Document uploaded  successfully!")
    
    user_question = st.text_area("Enter your question about the document:")

    if st.button("Get Answer"):
        answer = answer_question(user_question)

        st.markdown("### Answer:")
        st.markdown(answer)
