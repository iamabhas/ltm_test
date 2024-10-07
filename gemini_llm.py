# import google.generativeai as genai
# import os
# from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

# if not api_key:
#     raise ValueError("GEMINI_API_KEY not found in .env file")

# genai.configure(api_key=api_key)

# class GeminiLLM:
#     def __init__(self, model_name="gemini-1.5-flash"):
#         self.model = genai.GenerativeModel(model_name)

#     def call_api(self, prompt):
#         try:
#             response = self.model.generate_content(prompt)
#             return response.text
#         except Exception as e:
#             print(f"Error calling Gemini API: {e}")
#             return None


"""First install these things before running this file:
1. langchain_groq
2. python-dotenv


"""
from langchain_groq import ChatGroq

import os
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
load_dotenv()


api_key = os.getenv("GROQ_API")
def get_groq_llm(model_name:str="mixtral-8x7b-32768"):
    
    llm = ChatGroq(api_key=api_key,
                          model="mixtral-8x7b-32768",
                          temperature=0,
                          max_tokens=None,
                          timeout=None,
                          max_retries=2,)
    return llm

def query_llm(query:str, memory):
    # if memory ==[]:
    #     messages = [
    #             ("system", "You are a helpful assistant. Answer the user query provided to you."),
    #             ("human", query),
    #         ]
    # else:
    #      messages = [
    #             ("system", f"You are a helpful assistant. Answer the user query provided to you based on the memory.\n\n The memory is:{memory}"),
    #             ("human", query),
    #      ]
    template = """You are a nice chatbot having a conversation with a human.
                Previous conversation:
                {chat_history}
                New human question: {question}
                Response:"""
    
    
    prompt = PromptTemplate.from_template(template)

    llm = get_groq_llm()
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    response = conversation({"question":query})


    # response = llm.invoke(messages)
    return response['text']

# Example usage
# response = ""
# while True:
#     query = input("Please enter your query:")
#     # response = query_llm("What is Takayasu Arteritis?")
#     response = query_llm(query)
#     if str.lower(response) == "q":
#         print(response)
#         break
#     print(response)
    