### `llms.py`

配置 llm
- language model
    - gpt4o: `gpt4o = LLM(model="gpt-4o-2024-08-06", use_batch=True)`
    - gpt4omini: `gpt4omini = LLM(model="gpt-4o-mini-2024-07-18", use_batch=True)`
    - qwen2_5: `qwen2_5 = LLM(model="Qwen2.5-72B-Instruct-GPTQ-Int4", api_base="http://124.16.138.143:7812/v1")`

- vision model
    - qwen_vl: `qwen_vl = LLM(model="Qwen2-VL-72B-Instruct", api_base="http://124.16.138.144:7999/v1")`

- class LLM：提供与大语言模型交互的核心接口。支持单次调用、多模态输入（文本+图像）、上下文管理、批量请求等功能

- class Role: 定义对话中的角色（如助手、用户等），负责管理多轮对话、上下文选择、错误重试等

- class Turn: 表示对话中的一次交互（对话轮次）。包含用户输入、模型响应、消息历史、相关图像等信息

### `api.py`

定义 API 调用和代码执行框架，包括：
- `del_paragraph`/`del_image`/`replace_paragraph`/`replace_image`/`clone_paragraph`

### `multimodal.py`

定义 `ImageLabler` 类，用于处理演示文稿中的图片，包括收集图片信息、生成图片的描述（caption）、以及在幻灯片中应用这些描述。

用于收集ppt中图片类的信息

### `presentation.py`

ppt中元素 特性、处理等

### `induct.py`

定义 SlideInducter 类，用于对演示文稿中的幻灯片进行分类和布局归纳，生成模板结构和内容模式，并将结果以 JSON 格式缓存和存储

即论文 Stage I.


### `model_utils.py`

一些辅助函数

### `preprocess.py`

一些预处理函数, 包括文件操作、PDF和PPT处理、多线程任务执行、幻灯片分析与嵌入生成等任务

### `pptgen.py`

定义一个抽象基类 PPTGen 和它的子类 PPTCrew，用于生成演示文稿 (PPT) 的完整框架。该框架结合了语言模型 (LLM)、图像模型以及高度模块化的流程，用于自动化生成具有结构化内容和一致布局的 PPT 文件

整体流程
1. 初始化生成器：
    - 使用文本模型和配置初始化 PPTGen 或其子类。
2.	设置模板和归纳信息：
	- 加载模板演示文稿和布局归纳结果。
3.	生成大纲：
	- 根据文档内容和输入配置生成演示文稿的大纲。
4.	逐张生成幻灯片：
	- 调用子类实现的逻辑生成每张幻灯片的具体内容和布局。
5.	保存结果：
	- 保存生成的演示文稿，并记录生成过程中的操作历史。

