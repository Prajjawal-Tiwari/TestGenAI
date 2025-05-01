from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from ctransformers import AutoModelForCausalLM



def build_rag_chain(vectorstore):
    # llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama")
    llm = AutoModelForCausalLM.from_pretrained(
        model_path="models/llms/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",               # 'llama' or 'mistral' or 'gptj' depending on the model
        max_new_tokens=512,
        temperature=0.7
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())