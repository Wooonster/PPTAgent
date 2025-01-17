import asyncio
import base64
import os
import re
from dataclasses import asdict, dataclass
from math import ceil

import jsonlines
import requests
import tiktoken
import yaml
from FlagEmbedding import BGEM3FlagModel
from jinja2 import Environment, Template
from oaib import Auto
from openai import OpenAI
from PIL import Image
from torch import Tensor, cosine_similarity

from model_utils import get_text_embedding
from utils import get_json_from_response, pexists, pjoin, print, tenacity

import logging
from fastapi.logger import logger

# 设置日志级别为 INFO
logging.basicConfig(level=logging.INFO)

ENCODING = tiktoken.encoding_for_model("gpt-4o")
os.environ['OPENAI_API_KEY'] = 'ollama'

def run_async(coroutine):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    job = loop.run_until_complete(coroutine)
    return job


def calc_image_tokens(images: list[str]):
    tokens = 0
    for image in images:
        with open(image, "rb") as f:
            width, height = Image.open(f).size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024
        h = ceil(height / 512)
        w = ceil(width / 512)
        tokens += 85 + 170 * h * w
    return tokens


class LLM:
    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        api_base: str = None,
        use_openai: bool = True,
        use_batch: bool = False,
    ) -> None:
        # if use_openai and "OPENAI_API_KEY" in os.environ:
        #     self.client = OpenAI(base_url=api_base)
        # if use_batch and "OPENAI_API_KEY" in os.environ:
        #     assert use_openai, "use_batch must be used with use_openai"
        #     self.oai_batch = Auto(loglevel=0)
        # if "OPENAI_API_KEY" not in os.environ:
        #     print("Warning: no API key found")

        if use_openai and "OPENAI_API_KEY" in os.environ:
            from openai import OpenAI  # 确保正确导入 OpenAI 类
            self.client = OpenAI(base_url=api_base) if api_base else OpenAI()
        else:
            self.client = None  # 确保即使不使用 OpenAI，client 也被初始化
        if use_batch and "OPENAI_API_KEY" in os.environ:
            assert use_openai, "use_batch must be used with use_openai"
            self.oai_batch = Auto(loglevel=0)
        if "OPENAI_API_KEY" not in os.environ:
            print("Warning: no API key found")

        self.model = model
        self.api_base = api_base
        self._use_openai = use_openai
        self._use_batch = use_batch

    @tenacity  # 自动重试装饰器, 用于在请求失败时重试
    def __call__(
        self,
        content: str,
        images: list[str] = None,
        system_message: str = None,
        history: list = None,
        delay_batch: bool = False,
        return_json: bool = False,
        return_message: bool = False,
    ) -> str | dict | list:
        '''分割 system_message 和 用户内容'''
        if content.startswith("You are"):
            system_message, content = content.split("\n", 1)
        '''初始化历史对话'''
        if history is None:
            history = []
        if isinstance(images, str):
            images = [images]

        system, message = self.format_message(content, images, system_message)

        if self._use_batch:  # 批量模式 -> 异步发送请求并解析响应
            result = run_async(
                self._run_batch(system + history + message, delay_batch)
            )
            if delay_batch:
                return
            try:
                response = result.to_dict()["result"][0]["choices"][0]["message"][
                    "content"
                ]
            except Exception as e:
                print("Failed to get response from batch")
                raise e
        elif self._use_openai:  # OpenAI/Ollama 模式 
            completion = self.client.chat.completions.create(
                model=self.model, messages=system + history + message
            )
            response = completion.choices[0].message.content
        else:  # 自定义 API 模式 -> 使用 requests.post 发送请求。
            response = requests.post(
                self.api_base,
                json={
                    "system": system_message,
                    "prompt": content,
                    "image": [
                        i["image_url"]["url"]
                        for i in message[-1]["content"]
                        if i["type"] == "image_url"
                    ],
                },
            )
            response.raise_for_status()
            response = response.text

        message.append({"role": "assistant", "content": response})
        
        if return_json:
            # print(response)
            logger.info(f'response before send back in LLM class:\n{response}')
            response = get_json_from_response(response)
            logger.info(f'response after get_json_from_response in LLM class:\n{response}')
        if return_message:
            response = (response, message)
        return response

    def __repr__(self) -> str:
        return f"LLM(model={self.model}, api_base={self.api_base})"

    async def _run_batch(self, messages: list, delay_batch: bool = False):
        await self.oai_batch.add(
            "chat.completions.create",
            model=self.model,
            messages=messages,
        )
        if delay_batch:
            return
        return await self.oai_batch.run()

    def format_message(
        self,
        content: str,
        images: list[str] = None,
        system_message: str = None,
    ):
        '''格式化用户输入和系统消息为符合 API 的消息结构'''

        if system_message is None:
            system_message = "You are a helpful assistant"
        system = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            }
        ]
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content}
                ]
            }
        ]
        '''附加图像，通过将图像编码为 Base64 字符串并附加到消息中'''
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            for image in images:
                with open(image, "rb") as f:
                    message[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64, {base64.b64encode(f.read()).decode('utf-8')}"
                            },
                        }
                    )
        return system, message

    def get_batch_result(self):
        '''在批量请求执行完成后，获取批量任务的结果，并提取响应内容'''
        results = run_async(self.oai_batch.run())
        return [
            r["choices"][0]["message"]["content"]
            for r in results.to_dict()["result"].values()
        ]

    def clear_history(self):
        self.history = []


@dataclass
class Turn:
    '''表示对话系统中用户和模型的一次交互（对话轮次）'''
    id: int
    prompt: str
    response: str
    message: list
    images: list[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    embedding: Tensor = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k != "embedding"}

    def calc_token(self):
        if self.images is not None:
            self.input_tokens += calc_image_tokens(self.images)
        self.input_tokens += len(ENCODING.encode(self.prompt))
        self.output_tokens = len(ENCODING.encode(self.response))

    def __eq__(self, other):
        return self is other


class Role:
    def __init__(
        self,
        name: str,                          # 角色名称
        env: Environment,                   # 模板渲染环境
        record_cost: bool,                  # 是否记录 token 成本
        llm: LLM = None,                    # LLM 对象
        config: dict = None,                # 角色的配置字典
        text_model: BGEM3FlagModel = None,  # 文本嵌入模型
    ):
        self.name = name
        if config is None:  # 无，从 YAML 文件加载
            with open(f"../roles/{name}.yaml", "r") as f:
                config = yaml.safe_load(f)
        if llm is None:  # 无，从配置加载
            llm = globals()[config["use_model"] + "_model"]
        self.llm = llm
        self.model = llm.model
        self.record_cost = record_cost
        self.text_model = text_model

        self.return_json = config["return_json"]
        self.system_message = config["system_prompt"]
        self.prompt_args = set(config["jinja_args"])
        self.template = env.from_string(config["template"])

        # 错误重试prompt
        self.retry_template = Template(
            """The previous output is invalid, please carefully analyze the traceback and feedback information, correct errors happened before.
            feedback:
            {{feedback}}
            traceback:
            {{traceback}}
            Give your corrected output in the same format without including the previous output:
            """
        )
        self.system_tokens = len(ENCODING.encode(self.system_message))
        self.input_tokens = 0
        self.output_tokens = 0
        self.history: list[Turn] = []

    def calc_cost(self, turns: list[Turn]):
        '''计算输入和输出的总 token 数量，用于记录交互成本'''
        for turn in turns:
            self.input_tokens += turn.input_tokens
            self.output_tokens += turn.output_tokens
        self.input_tokens += self.system_tokens
        self.output_tokens += 3

    def get_history(self, similar: int, recent: int, prompt: str):
        '''获取当前交互所需的对话历史'''
        history = self.history[-recent:] if recent > 0 else []
        if similar > 0:
            embedding = get_text_embedding(prompt, self.text_model)
            history.sort(
                key=lambda x: cosine_similarity(embedding, x.embedding)
            )
            for turn in history:
                if len(history) > similar + recent:
                    break
                if turn not in history:
                    history.append(turn)
        history.sort(key=lambda x: x.id)
        return history

    def save_history(self, output_dir: str):
        '''将对话历史保存到 JSONL 文件'''
        history_file = pjoin(output_dir, f"{self.name}.jsonl")
        if pexists(history_file) and len(self.history) == 0:
            return
        with jsonlines.open(history_file, "w") as writer:
            writer.write(
                {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                }
            )
            for turn in self.history:
                writer.write(turn.to_dict())

    def retry(self, feedback: str, traceback: str, error_idx: int):
        '''对错误的对话轮次进行重试
        
        feedback  -> 错误的反馈信息。
        traceback -> 错误的堆栈信息。
        error_idx -> 重试的对话轮次索引
        '''
        assert error_idx > 0, "error_idx must be greater than 0"
        prompt = self.retry_template.render(feedback=feedback, traceback=traceback)
        history = []
        for turn in self.history[-error_idx:]:
            history.extend(turn.message)
        response, message = self.llm(
            prompt,
            history=history,
            return_message=True,
        )
        turn = Turn(
            id=len(self.history),
            prompt=prompt,
            response=response,
            message=message,
        )
        return self.__post_process__(response, self.history[-error_idx:], turn)

    def __repr__(self) -> str:
        '''返回角色的字符串表示形式，便于调试和打印'''
        return f"Role(name={self.name}, model={self.model})"

    def __call__(
        self,
        images: list[str] = None,
        recent: int = 0,
        similar: int = 0,
        **jinja_args,
    ):
        '''与 LLM 进行一次对话交互'''
        if isinstance(images, str):
            images = [images]
        assert self.prompt_args == set(jinja_args.keys()), "Invalid arguments"
        prompt = self.template.render(**jinja_args)
        history = self.get_history(similar, recent, prompt)
        history_msg = []
        for turn in history:
            history_msg.extend(turn.message)

        response, message = self.llm(
            prompt,
            system_message=self.system_message,
            history=history_msg,
            images=images,
            return_message=True,
        )
        turn = Turn(
            id=len(self.history),
            prompt=prompt,
            response=response,
            message=message,
            images=images,
        )
        return self.__post_process__(response, history, turn, similar)

    def __post_process__(
        self, response: str, history: list[Turn], turn: Turn, similar: int = 0
    ):
        '''对 LLM 响应进行后续处理'''
        # 将新轮次加入历史
        self.history.append(turn)
        if similar > 0:
            turn.embedding = get_text_embedding(turn.prompt, self.text_model)

        # 计算 token 数量
        if self.record_cost:
            turn.calc_token()
            self.calc_cost(history + [turn])
        
        if self.return_json:
            response = get_json_from_response(response)
        return response


def get_simple_modelname(llms: list[LLM]):
    if isinstance(llms, LLM):
        llms = [llms]
    logger.info(f'llms got in get_simple_modelname is: {llms}')
    # return "+".join(re.search(r"^(.*?)-\d{2}", llm.model).group(1) for llm in llms)

    model_names = []
    for llm in llms:
        # match = re.search(r"^(.*?)-\d{2}", llm.model)
        match = re.search(r"^(.*?)(?:-GPTQ|-\d{2}|:\d+\.\d+b|Int\d+|Instruct)", llm.model)
        if match:
            model_names.append(match.group(1))
        else:
            raise ValueError(f"Invalid model name format: {llm.model}")
    return "+".join(model_names)


# gpt4o = LLM(model="gpt-4o-2024-08-06", use_batch=True)
# gpt4omini = LLM(model="gpt-4o-mini-2024-07-18", use_batch=True)

# intern_vl = LLM(model="InternVL2_5-78B", api_base="http://127.0.0.1:8009/v1")

# !ollama serve
# !ollama run qwen2.5:1.5b
qwen2_5 = LLM(model="qwen2.5:1.5b", api_base="http://localhost:11434/v1/")  # api_base/key = 'ollama'
# !python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-2B-Instruct-GPTQ-Int4 --model Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4 --download_dir /root/autodl-tmp/qwen/Qwen2-VL-2B --gpu-memory-utilization 0.5
qwen_vl = LLM(model="Qwen2-VL-2B-Instruct-GPTQ-Int4", api_base="http://localhost:8000/v1")  # openai_api_key = "EMPTY"
qwen_coder = LLM(model="Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4", api_base="http://127.0.0.1:8008/v1")

'''
vllm serve Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4 \
  --served-model-name Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4 \
  --download-dir /root/autodl-tmp/qwen/coder-1.5b \
  --gpu-memory-utilization 0.2
  --port 8008
'''

language_model = qwen2_5
code_model = qwen2_5
vision_model = qwen_vl

if __name__ == "__main__":
    # gpt4o = LLM(model="gpt-4o-2024-08-06")
    # print(
    #     gpt4o(
    #         "who r u",
    #     )
    # )

    qwen2_5 = LLM(model="qwen2.5:1.5b", api_base="http://localhost:11434/v1/")
    print(qwen2_5("who r u"))
