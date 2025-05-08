import json
import requests
import jsonlines
from tqdm import tqdm
from threading import Thread
import threading
import time
from datasets import load_dataset
import os

access_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

def ernie_speed_chat(prompt,temperature):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed?access_token=" + access_token

    disable_search = True

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
         ],
         
         "temperature": temperature,
         "response_format": "text",
         "max_output_tokens":2048,
         "disable_search":disable_search,
         "enable_trace":True
    })
    headers = {
        'Content-Type': 'application/json'
    }
 
    response = requests.request("POST", url, headers=headers, data=payload)
    res = json.loads(response.text)
    return res

ner_prompt = """**Task**:
Please generate high-quality Named Entity Recognition (NER) instruction-tuning data based on the provided medical knowledge.

**Medical Knowledge**:
```{entity_knowledge}```

**Please adhere to the following guidelines**:
- The instruction should clearly describe the task objective, including the types of entities to be identified (e.g., diseases, medications, symptoms).
- The input text should contain sufficient context to accurately recognize entities.
- The output should list all identified entities along with their categories, formatted as: Entity Name|Category.
- If applicable, provide any potential aliases or abbreviations.
- Ensure that both instructions and outputs comply with industry standards and terminology in the medical field.

**Return Format**:
{
    "instruction": "A specific NER instruction, e.g., Identify and label all disease names from the given text.",
    "input": "The medical text for analysis, e.g., The patient reported symptoms of headache and fever and has been diagnosed with influenza.",
    "output": "The correct output corresponding to the input, e.g., headache|symptom, fever|symptom, influenza|disease"
  }
"""

text_classification_prompt = """**Task**:
Please generate high-quality medical text classification instruction-tuning data based on the provided medical knowledge.

**Medical Knowledge**:
```{entity_knowledge}```

**Please adhere to the following guidelines**:
- The instruction should clearly describe the task objective, including the types of texts to be classified (e.g., diagnostic reports, treatment plans, medical records).
- The input text should contain sufficient context for accurate classification.
- The output should provide the correct classification labels for each input text.
- Classification labels should be clear and adhere to standard medical terminology.
- If applicable, explain any subtle differences or special cases between categories.

**Return Format**:
{
    "instruction": "A specific medical text classification instruction, e.g., Classify the given diagnostic report into the appropriate disease category.",
    "input": "The medical text for analysis, e.g., The patient complains of headache, accompanied by nausea and blurred vision...",
    "output": "The correct classification corresponding to the input, e.g., Migraine"
}
"""

relation_extract_prompt = """**Task**:
Please construct a high-quality Instruction fine-tuning sample for relation extraction based on the provided medical knowledge, intended for the analysis and understanding of medical text information.

**Medical Knowledge**:
```{entity_knowledge}```

**Please follow these guidelines to create the dataset**:
1. Ensure each question focuses on extracting specific types of information from the given medical knowledge.
2. Design questions to be specific enough to avoid ambiguity or multiple interpretations.
3. Answers should accurately reflect the facts presented in the medical knowledge and include as much detail as possible.
4. Where applicable, include relationships between entities such as drugs and diseases, symptoms and diagnoses.
5. Use professional terminology while ensuring that the text remains clear and comprehensible for medical professionals.

**Return Format**:
{
    "question": "A specific, focused task for extracting information from medical text",
    "answer": "A precise answer to the question, including all necessary details"
}
"""

qa_prompt = """**Task**:
 Please construct a set of high-quality, practically applicable medical Q&A data based on the provided medical knowledge. Ensure that each question is clear, specific, and that the answers accurately reflect the latest medical insights and practices.

**Medical Knowledge**:
```{entity_knowledge}```

**Please follow these guidelines to create the dataset**:
1. Questions should be directly based on the provided medical knowledge, avoiding overly broad or unrelated questions.
2. Answers should include sufficient explanatory information to ensure understanding by non-specialists.
3. Where applicable, include key points such as preventive measures, symptom recognition, treatment options, and expected outcomes.
4. For potentially confusing concepts or terms, provide concise definitions or clarifications.
5. If the medical knowledge mentions statistics or research findings, cite this information in your answers to enhance authority.

**Return Format**:
{
    "question": "Specific high-quality medical question",
    "answer": "Correct response to the question, including all necessary explanations and background information"
}
"""

multi_choice_prompt = """**Task**:
Please construct a set of high-quality, practically applicable multiple-choice questions (MCQs) based on the provided medical knowledge. Ensure that each question is clear and specific, and that the options are well-designed, including one correct answer and several plausible distractors (incorrect but seemingly reasonable options). Additionally, the answer section should clearly indicate the correct option and may optionally provide an explanation to aid learners in understanding.

**Medical Knowledge**:
```{entity_knowledge}```

**The created MCQs should meet the following criteria**:
1. Questions should be directly based on the provided medical knowledge, avoiding overly broad or unrelated questions.
2. Each question should have four options: one correct answer and three distractors. The distractors should be plausible incorrect options that appear reasonable to non-experts.
3. Answers should clearly specify which option is correct and may optionally include a brief explanation or additional information to clarify why this option is correct.
4. Where applicable, include key points such as preventive measures, symptom recognition, treatment options, and expected outcomes.
5. For potentially confusing concepts or terms, provide concise definitions or clarifications within the question.
6. If the medical knowledge mentions statistics or research findings, reference this information in the question or explanation to enhance credibility.

**Return Format**:
{
    "question": "Specific high-quality medical multiple-choice question",
    "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
    ],
    "correct_answer": "Correct answer letter (e.g., A, B, C, or D)",
    "explanation": "Explanation of why this option is correct, along with any other relevant information"
}
"""

dialog_prompt = """**Task**: 
Extract key medical knowledge with educational and clinical application value from the provided professional medical text, and design a dialogue scenario that includes at least six rounds of interaction. This dialogue should guide users to gain a deeper understanding of medical knowledge step by step while ensuring that information is communicated clearly, accurately, and in an accessible manner.

**Medical Text**:
````{entity_knowledge}```

**Please adhere to the following guidelines**:
1. The dialogue should flow naturally, simulating authentic doctor-patient communication.
2. Each round of dialogue must include a user's question or statement, followed by the AI assistant's response.
3. The assistant's responses should be based on the provided medical text but explain complex concepts in an easy-to-understand way.
4. The assistant should pose questions or offer suggestions at appropriate times to encourage user engagement and reflection.
5. Ensure all critical medical knowledge points are covered within the dialogue.
6. By the end of the dialogue, the user should have a comprehensive understanding of the discussed topic.

**Return Format**:
[
    {
        "human": "User's question or statement",
        "assistant": "AI assistant's response or advice"
    },
    {
        "human": "Possible further questions or reactions from the user",
        "assistant": "Further explanation or guidance from the AI assistant"
    },
    ...
]
"""

prompt_list = [ner_prompt,text_classification_prompt,relation_extract_prompt,qa_prompt,multi_choice_prompt,dialog_prompt]
task_type_list = ["ner","text_classification","relation_extract","qa","multi_choice","dialog"]


def sub_random_generate_task(i,item,pbar):
    target_dir = r".\format"
    temp = item

    entity_knowledge = item["entity_knowledge"].replace("\n\n","\n").replace("### ","").replace("**","").replace("* ","").replace("\t","")

    temperature = 0.1

    cur = 0
    while cur<3:
        try:
            if len(entity_knowledge)<50:
                break

            randam_num = i%3

            task_type = task_type_list[randam_num]
            file_name = task_type+".jsonl"
            target_path = os.path.join(target_dir,file_name)

            temp["task_type"] = task_type

            prompt = prompt_list[randam_num].format(entity_knowledge = entity_knowledge)

            res = ernie_speed_chat(prompt,temperature)

            temp["questions"] = res["result"]

            with jsonlines.open(target_path,"a") as writer:
                writer.write(temp)

            break
        except Exception as e:
            print(res)
            print(e)
            time.sleep(8)
            cur+=1

    pbar.update(1)

def random_generate_task():

    path= r".\raw\raw_know.jsonl"

    with jsonlines.open(path,"r") as items:
        count = sum(1 for _ in items)

    with jsonlines.open(path,"r") as items:
        pbar = tqdm(total=count)

        for i,item in enumerate(items):

            while threading.active_count()>40:
                time.sleep(8)

            cur_thread=Thread(target=sub_random_generate_task,args=(i,item,pbar))

            cur_thread.start()



def main():
    random_generate_task()

if __name__ == "__main__":
    main()
