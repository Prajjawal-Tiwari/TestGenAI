import os
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

# --- Set Hugging Face API Token Directly in the Script ---
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# --- Page config ---
st.set_page_config(page_title="TestGenAI – Simple and clear: Test Generation using AI", layout="centered")

# --- Title ---
st.title("🧠 TestGenAI")

# --- RAG Toggle ---
use_rag = st.checkbox("Enable RAG")

# --- Use Case Selector (Non-editable dropdown) ---
use_case = st.selectbox(
    "Choose a Use Case",
    options=[
        "Test Case Generator",
        "Incident/Bug Chatbot",
        "Automation Script Generator"
    ],
    index=0,
    disabled=False
)

# --- File Upload (Only if RAG is enabled) ---
uploaded_files = None
if use_rag:
    uploaded_files = st.file_uploader(
        "Upload your reference documents (Structured/Unstructured):",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "csv"]
    )
else:
    st.info("📂 File upload is only available when RAG is enabled.", icon="🔒")

# --- Prompt input ---
user_prompt = st.text_area("Enter your prompt or requirement", height=150)

# --- Fallback: Use Hugging Face model if RAG is not enabled or for any other case ---
if use_rag or not uploaded_files:
    # Set up Hugging Face model from Hugging Face Hub
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",  # Use a smaller model for quicker responses
        model_kwargs={"temperature": 0.7, "max_length": 150}
    )

    # Create a prompt template (if needed, adjust this)
    prompt_template = PromptTemplate(input_variables=["user_prompt"], template="{user_prompt}")

    # Build the chain for generating responses
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # --- Submit Button ---
    if st.button("Generate Response"):
        if not user_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            st.success("Processing your request...")
            result = chain.run(user_prompt=user_prompt)  # Run the chain to generate the response
            st.write(result)
else:
    st.warning("Please enable RAG and upload files to generate a response.")
