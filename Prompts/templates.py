from langchain.prompts import PromptTemplate

TEMPLATES = {
    "test_case_gen": PromptTemplate.from_template("Generate test cases for:\n{context}\nUser Prompt: {query}"),
    "chatbot": PromptTemplate.from_template("Analyze this bug report:\n{context}\nUser Prompt: {query}"),
    "script_gen": PromptTemplate.from_template("Create Java Selenium BDD code for:\n{context}\nUser Prompt: {query}")
}
