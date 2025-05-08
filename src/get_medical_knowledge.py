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

def sub_get_entity_knowledge(item,pbar):
    target_path = r".\raw\after_diease_know.jsonl"
    temp = item
    entity = temp["entity"]
    prompt = "详细介绍医学术语:{entity}".format(entity=entity)
    temperature = 0.5
    cur = 0
    while cur<3:
        try:
            res = ernie_speed_chat(prompt,temperature)

            temp["entity_knowledge"] = res["result"]
            temp["search_info"] = res["search_info"]

            with jsonlines.open(target_path,"a") as writer:
                writer.write(temp)

            break
        except Exception as e:
            print(e)
            time.sleep(8)
            cur+=1

    pbar.update(1)

def get_entity_knowledge():
    path= r".\raw\after_diease.jsonl"

    with jsonlines.open(path,"r") as items:
        count = sum(1 for _ in items)

    pbar = tqdm(total=count)

    with jsonlines.open(path,"r") as items:

        for item in items:

            while threading.active_count()>40:
                time.sleep(8)

            cur_thread=Thread(target=sub_get_entity_knowledge,args=(item,pbar))

            cur_thread.start()


def main():
    get_entity_knowledge()

if __name__=="__main__":
    main()