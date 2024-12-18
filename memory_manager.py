from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key='chat_history')

def save_facts_to_memory(facts):
    for fact in facts:
        memory.save_context({"user": fact}, {"response": ""}) 

def clear_memory():
    memory.clear()
