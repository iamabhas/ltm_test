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

from langchain.memory import ConversationBufferMemory

"""First install these things before running this file:
1. langchain_groq
2. python-dotenv


"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_groq import ChatGroq

import os
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
load_dotenv()

from tests.llm_test_cases import test_cases
api_key = os.getenv("GROQ_API")
def get_groq_llm(model_name:str="mixtral-8x7b-32768"):
    
    llm = ChatGroq(api_key=api_key,
                          model="mixtral-8x7b-32768",
                          temperature=0,
                          max_tokens=None,
                          timeout=None,
                          max_retries=2,)
    return llm

def query_llm(query:str, set_memory = ['set', 'unset']):
    """
    set_memory = ['set', 'unset']
    set: when we want to put messages to memory variable.
    unset: When we want the memory to be empty and llm to use its own memory 
    
    """
    llm = get_groq_llm()

    prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for i in test_cases:
        question = i['question']
        response = i['expected']    
        memory.save_context({"user": question}, {"response": response} )
    
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    if set_memory == 'set':
        response = conversation.invoke({"question":query})
        return response['text']
       
    elif set_memory == 'unset':
        memory.clear()
        response = conversation.invoke({"question":query})
        return response['text']
        
    else:
        return f"The available values for set_memory are only 'set' and 'unset'. But you provided {set_memory} which is invalid."

    
    # return response

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
    