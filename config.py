import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def _get_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


class Config():

    #retrievor参数
    topd = 3    #召回文章的数量
    topt = 6    #召回文本片段的数量
    maxlen = 128  #召回文本片段的长度
    topk = 5    #query召回的关键词数量
    bert_path = os.getenv('BERT_PATH', '/workspace/model/embedding/tao-8k')   #无本地Bert模型，需修改
    recall_way = os.getenv('RAG_RECALL_WAY', 'embed')  #召回方式 ,keyword,embed

    #generator参数
    max_source_length = 767  #输入的最大长度
    max_target_length = 256  #生成的最大长度
    model_max_length = 1024  #序列最大长度
    
    #embedding API 参数 - 用于 text2vec.py
    use_api = _get_bool('RAG_USE_API', True)  # 是否使用API而非本地模型
    api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY') or "sk-xx"
    base_url = os.getenv('DASHSCOPE_BASE_URL', "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = os.getenv('DASHSCOPE_EMBEDDING_MODEL', "text-embedding-v3")
    dimensions = _get_int('DASHSCOPE_EMBEDDING_DIMENSIONS', 1024)  #将输入转为1024维的向量
    batch_size = _get_int('RAG_EMBEDDING_BATCH_SIZE', 10)
    
    #LLM API 参数 - 用于 rag.py
    llm_api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY') or api_key  # 与embedding共用同一个key
    llm_base_url = os.getenv('DASHSCOPE_BASE_URL', base_url)  # 与embedding共用同一个URL
    llm_model = os.getenv('DASHSCOPE_LLM_MODEL', "qwen-plus")  # 默认使用的LLM模型
    
    # 知识库配置
    kb_base_dir = os.getenv('RAG_KB_BASE_DIR', "knowledge_bases")  # 知识库根目录
    default_kb = os.getenv('RAG_DEFAULT_KB', "default")  # 默认知识库名称
    
    # 输出目录配置 - 现在用作临时文件目录
    output_dir = os.getenv('RAG_OUTPUT_DIR', "output_files")
