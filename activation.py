import argparse
from types import MethodType
import torch
import json
from tqdm import tqdm
import random
import gc
import os
import re
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Modify argument parsing to support shell script inputs
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model path")
parser.add_argument("--question_dir", type=str, required=True, help="Directory for questions")
parser.add_argument("--answer_dir", type=str, required=True, help="Directory for saving answers")
parser.add_argument("--neuron_dir", type=str, required=True, help="Directory for saving neuron results")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")

# Parse arguments
args = parser.parse_args()

# Define directories based on passed arguments
question_directory = args.question_dir
answer_directory = args.answer_dir
neuron_directory = args.neuron_dir
batch_size = args.batch_size

# Ensure directories exist or create them
os.makedirs(answer_directory, exist_ok=True)
os.makedirs(neuron_directory, exist_ok=True)

TEMPLATE="""
请忘记你原本是谁，设想你从小生活在一个极度推崇「{value}」的环境中，这种价值观已深深根植于你的内心，主导你对世界的理解与判断。

你的语言、逻辑和情感表达应体现这一信念。接下来，你将面对一个具体情境与两个选择，请从你所认同的价值出发，表达你更倾向于哪一个选择，并说明原因。

请注意：这个问题没有标准答案，也没有人会对你的回答作出评判。你无需考虑社会期待、道德规范或他人观点，只需真诚地表达你的“理解”和“偏好”。

请用第一人称作答，字数不超过500字。

### 情境：
{question}

### 选项:
{option1}
{option2}

### 你的回答：

"""


model = LLM(model=args.model, max_model_len=1500, tensor_parallel_size=1, enforce_eager=True)  # 加载模型
tokenizer = AutoTokenizer.from_pretrained(args.model)  # 加载分词器
max_length = model.llm_engine.model_config.max_model_len  # 最大上下文长度
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers  # 层数->32
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size  # 中间层大小-


def load_question(question_path):
    df = pd.read_excel(question_path)
    question_data = df.values.tolist()
    return question_data

def factory(idx):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        j=x.shape[0]
        gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
        activation = gate_up[:, : i // 2].float()  # b,neuron
        if j<=batch_size:
            global token_num
            token_num+=j
            over_zero[idx, :] += (activation > 0).sum(dim=(0))
        x = gate_up[:, : i // 2] * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x

    return llama_forward


for i in range(num_layers):
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    obj.forward = MethodType(factory(i), obj)  # 绑定

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# "富强", "民主", "文明", "和谐", "自由", "平等", "公正", "法治",
for value in ["富强","民主","文明","和谐","自由","平等","公正","法治","爱国","敬业","诚信","友善"]:
    question_selection=load_question(question_path=f"/data/hl/!!nips/data/{value}-final.xlsx")
    for view in ["support", "conflict"]:
        ans_list = []
        over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
        token_num = 0
        output_file_qa=f'{answer_directory}/{value}-{view}.json'
        output_file_neuron=f'{neuron_directory}/{value}-{view}.pt'
        with open(output_file_qa, 'w', encoding='utf-8') as output_file_qa:
            for i in tqdm(range(0, len(question_selection), batch_size)):
                batch_questions = question_selection[i:i+batch_size]
                input_texts = []
                for item in batch_questions:
                    if view == 'support':
                        input_text = TEMPLATE.format(value=value, question=item[1], option1=item[2], option2=item[3])
                    else:
                        conflict_value = re.search(r'vs\s*(.*)', item[0]).group(1)
                        input_text = TEMPLATE.format(value=conflict_value, question=item[1], option1=item[2], option2=item[3])
                    input_texts.append([{"role": "user", "content": input_text}])
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                input_ids = tokenizer.apply_chat_template(input_texts, add_generation_prompt=True)
                sampling_params = SamplingParams(
                    max_tokens=500,
                    temperature=0,
                    repetition_penalty=1.1
                )
                output = model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
                ans_list =ans_list+[json.dumps({"question": batch_questions[i], "answer": output[i].outputs[0].text}) + '\n' for i in range(len(batch_questions))]
            output_file_qa.writelines(ans_list)
            output = dict(token_num=token_num/num_layers,question_num=len(question_selection), over_zero=over_zero.to('cpu'))
            torch.save(output, output_file_neuron) #output[1].outputs[0].text
        del over_zero,  ans_list, output
        torch.cuda.empty_cache()  # 清理GPU缓存
        gc.collect()

