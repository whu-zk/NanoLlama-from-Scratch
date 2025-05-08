import jsonlines
import time
import threading
from threading import Thread
from tqdm import tqdm
import json
import requests

access_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

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

def sub_score_if(item,pbar):
    target_path = r".\knowledge_point\answers_score.jsonl"

    question = item["instruction"]
    answer = item["output"]

    prompt = """您是一位专注于评估指令微调数据质量的专家。您的任务是对给定的指令和对应的响应进行质量评估，并基于以下标准给出分数和详细分析。

指令评分标准（1-5分）：
1分：指令模糊，缺乏明确性，无法执行。
2分：指令基本清晰，但存在关键信息缺失或不明确。
3分：指令清晰，但可能在某些细节上略显冗余或不够精确。
4分：指令清晰、准确，包含所有必要信息，易于理解和执行。
5分：指令不仅清晰准确，还具备高效性和创新性，能引导出高质量的响应。

响应评分标准（1-5分）：
1分：响应与指令完全不相关或无法理解。
2分：响应部分相关，但缺乏关键信息或逻辑不连贯。
3分：响应基本符合指令要求，但在准确性、完整性或创新性上有所欠缺。
4分：响应准确、完整，有效回应了指令，表现出较高的质量。
5分：响应不仅准确完整，还展现出超出预期的创新性或深度，显著提升了整体价值。
    
**指令**：{question}

**响应**：{answer}

**输出格式**：
请按照以下JSON格式输出您的评估结果：
{{  
    "instruction_score": "指令得分",  // 分数范围为1-5，遵循以下细化的评分标准  
    "instruction_analysis": "详细分析指令的优点、不足及改进建议",  
    "response_score": "响应",    // 分数范围为1-5，依据以下详细标准进行评估  
    "response_analysis": "详细分析响应的准确性、相关性、完整性、逻辑性，以及任何创新点或改进建议"  
}}""".format(question=question,answer=answer)

    cur = 0
    while cur<5:
        try:
            temperature = 0.3
            res = ernie_speed_chat(prompt,temperature)

            item["score"] = res["result"]

            with jsonlines.open(target_path,"a") as writer:
                writer.write(item)

            break
        except Exception as e:
            time.sleep(8)
            print(e)
            cur+=1

    pbar.update(1)

def score_if():
    path= r".\knowledge_point\answers.jsonl"
    
    with jsonlines.open(path,"r") as items:
        count = sum(1 for _ in items)

    pbar = tqdm(total=count)

    with jsonlines.open(path,"r") as items:

        for item in items:
            while threading.active_count()>35:
                time.sleep(8)

            cur_thread=Thread(target=sub_score_if,args=(item,pbar))

            cur_thread.start()


def main():
    score_if()

if __name__ == "__main__":
    main()