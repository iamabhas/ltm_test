from gemini_llm import GeminiLLM
from memory_manager import save_facts_to_memory, clear_memory, memory
from semantic_similarity import check_accuracy_semantic
from utils.append_to_file import append_to_file  
from tests.llm_test_cases import test_cases 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


llm = GeminiLLM()

facts = [
   "Python is a popular programming language.",
"AI stands for Artificial Intelligence.",
   "The sun is a star.",
"Water boils at 100 degrees Celsius.",
"The earth revolves around the sun."
]


save_facts_to_memory(facts)
output_file = "llm_test_results.txt"
results = []
iterations = 2


for i in range(iterations):
    clear_memory() 

    for test in test_cases:
        response_no_memory = llm.call_api(test["question"])
        is_accurate_no_memory, similarity_no_memory = check_accuracy_semantic(response_no_memory, test["expected"])

        no_memory_log = f"--- Test {i + 1} (No Memory) ---\nQuestion: {test['question']}\nExpected: {test['expected']}\nResponse: {response_no_memory}\nAccuracy: {is_accurate_no_memory}\nSimilarity: {similarity_no_memory * 100:.2f}%\n"
        append_to_file(output_file, no_memory_log)


        results.append({
            "question": test["question"],
            "expected": test["expected"],
            "accuracy_no_memory": is_accurate_no_memory,
            "similarity_no_memory": similarity_no_memory
        })


        memory.save_context({"user": test["question"]}, {"response": ""})
        response_with_memory = llm.call_api(test["question"])
        is_accurate_with_memory, similarity_with_memory = check_accuracy_semantic(response_with_memory, test["expected"])


        memory_log = f"--- Test {i + 1} (With Memory) ---\nQuestion: {test['question']}\nExpected: {test['expected']}\nResponse: {response_with_memory}\nAccuracy: {is_accurate_with_memory}\nSimilarity: {similarity_with_memory * 100:.2f}%\n"
        append_to_file(output_file, memory_log)


        results.append({
            "question": test["question"],
            "expected": test["expected"],
            "accuracy_with_memory": is_accurate_with_memory,
            "similarity_with_memory": similarity_with_memory
        })

accuracy_no_memory = np.mean([r['accuracy_no_memory'] for r in results if 'accuracy_no_memory' in r]) * 100
accuracy_with_memory = np.mean([r['accuracy_with_memory'] for r in results if 'accuracy_with_memory' in r]) * 100

similarity_no_memory = np.mean([r['similarity_no_memory'] for r in results if 'similarity_no_memory' in r]) * 100
similarity_with_memory = np.mean([r['similarity_with_memory'] for r in results if 'similarity_with_memory' in r]) * 100


summary = f"\n--- Summary ---\nAverage Accuracy Without Memory: {accuracy_no_memory:.2f}%\nAverage Accuracy With Memory: {accuracy_with_memory:.2f}%\nAverage Similarity Without Memory: {similarity_no_memory:.2f}%\nAverage Similarity With Memory: {similarity_with_memory:.2f}%\n"
append_to_file(output_file, summary)

labels = ['Accuracy Without Memory', 'Accuracy With Memory', 'Similarity Without Memory', 'Similarity With Memory']
sizes = [accuracy_no_memory, accuracy_with_memory, similarity_no_memory, similarity_with_memory]
colors = ['red', 'green', 'blue', 'yellow']

patches = [
    mpatches.Patch(color='red', label=f'Accuracy Without Memory: {accuracy_no_memory:.2f}%'),
    mpatches.Patch(color='green', label=f'Accuracy With Memory: {accuracy_with_memory:.2f}%'),
    mpatches.Patch(color='blue', label=f'Similarity Without Memory: {similarity_no_memory:.2f}%'),
    mpatches.Patch(color='yellow', label=f'Similarity With Memory: {similarity_with_memory:.2f}%')
]

plt.figure(figsize=(4, 4))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.legend(handles=patches, loc='lower left', bbox_to_anchor=(1.2, 1))


plt.title('Accuracy and Similarity: LLM with and without Memory')
plt.show()
