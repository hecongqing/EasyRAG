import os
import json
import glob
import math
from typing import List, Dict, Any, Tuple

import jieba
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

import yaml

# 可选：用于编码与重排
from transformers import AutoTokenizer, AutoModel
import torch


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_stopwords(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f if line.strip()])


def iter_txt_files(base_dir: str) -> List[str]:
    return [p for p in glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True)]


def read_pathmap(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return {}
    return json.loads(open(path, "r", encoding="utf-8").read())


def tokenize_zh(text: str, stopwords: set) -> List[str]:
    return [w for w in jieba.cut(text) if w.strip() and w not in stopwords]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_es_client(cfg: Dict[str, Any]) -> Elasticsearch:
    es_conf = cfg["elasticsearch"]
    es = Elasticsearch(hosts=es_conf["hosts"], basic_auth=es_conf.get("basic_auth"))
    return es


def ensure_es_index(es: Elasticsearch, index_name: str, mappings: Dict[str, Any]):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, mappings=mappings)


def index_text_chunks_to_es(cfg: Dict[str, Any]) -> None:
    base_dir = os.path.abspath(cfg["processed_data_dir"])  # ../data/format_data_with_img
    stopwords = read_stopwords(os.path.abspath(cfg["stopwords_path"]))
    pathmap = read_pathmap(os.path.abspath(cfg["pathmap_path"]))

    es = build_es_client(cfg)
    index_name = cfg["elasticsearch"]["index_text"]

    mappings = {
        "properties": {
            "file_path": {"type": "keyword"},
            "know_path": {"type": "keyword"},
            "dir": {"type": "keyword"},
            "content": {"type": "text", "analyzer": "standard"},
            "tokens": {"type": "keyword"},
        }
    }
    ensure_es_index(es, index_name, mappings)

    chunk_size = cfg["chunk_size"]
    overlap = cfg["chunk_overlap"]

    docs = []
    for txt_path in tqdm(iter_txt_files(base_dir), desc="Index ES text chunks"):
        try:
            content = open(txt_path, "r", encoding="utf-8").read()
        except Exception:
            content = open(txt_path, "r", encoding="gb2312").read()

        # 统一 file_path 作为相对路径（包含包名）
        file_path_rel = os.path.relpath(txt_path, base_dir)
        # 反推包名目录
        parts = file_path_rel.split(os.sep)
        dir_name = parts[0] if parts else ""
        know_path = pathmap.get("/".join([dir_name, file_path_rel.replace(os.sep, "/")]), [])

        # 文档扩展：路径 + 内容
        # 但ES检索仍以 content/tokens 为主
        chunks = chunk_text(content, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            tokens = tokenize_zh(chunk, stopwords)
            docs.append({
                "_index": index_name,
                "_id": f"{file_path_rel}:::{i}",
                "_source": {
                    "file_path": file_path_rel.replace(os.sep, "/"),
                    "know_path": "/".join(know_path) if know_path else "",
                    "dir": dir_name,
                    "content": chunk,
                    "tokens": tokens,
                }
            })
            if len(docs) >= 2000:
                helpers.bulk(es, docs)
                docs = []
    if docs:
        helpers.bulk(es, docs)


def index_paths_to_es(cfg: Dict[str, Any]) -> None:
    # 将知识路径作为“文档”放入 ES 用于路径检索
    es = build_es_client(cfg)
    index_name = cfg["elasticsearch"]["index_path"]
    mappings = {
        "properties": {
            "know_path": {"type": "text", "analyzer": "standard"},
            "dir": {"type": "keyword"},
        }
    }
    ensure_es_index(es, index_name, mappings)

    pathmap = read_pathmap(os.path.abspath(cfg["pathmap_path"]))
    docs = []
    for file_path, know_path in tqdm(pathmap.items(), desc="Index ES paths"):
        dir_name = file_path.split("/")[0]
        path_text = "/".join(know_path)
        docs.append({
            "_index": index_name,
            "_id": file_path,
            "_source": {
                "dir": dir_name,
                "know_path": path_text,
            }
        })
        if len(docs) >= 2000:
            helpers.bulk(es, docs)
            docs = []
    if docs:
        helpers.bulk(es, docs)


# ======== 向量部分（Milvus） ========

def connect_milvus(cfg: Dict[str, Any]):
    milvus = cfg["milvus"]
    connections.connect(alias="default", host=milvus["host"], port=str(milvus["port"]))


def ensure_milvus_collection(name: str, dim: int, cfg: Dict[str, Any]) -> Collection:
    milvus = cfg["milvus"]
    if utility.has_collection(name):
        return Collection(name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="dir", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="know_path", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="aiops24 text chunks")
    coll = Collection(name, schema)
    coll.create_index(
        field_name="embedding",
        index_params={
            "index_type": milvus["index_type"],
            "metric_type": milvus["metric_type"],
            "params": {"nlist": milvus["nlist"]},
        },
    )
    return coll


class GTEEncoder:
    def __init__(self, model_name: str):
        # 教学版可用轻量模型替换；保持接口不变
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        # 取 cls pooling 或 mean pooling 简化
        if hasattr(outputs, "last_hidden_state"):
            emb = outputs.last_hidden_state.mean(dim=1)
        else:
            emb = outputs.pooler_output
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().tolist()


def index_text_chunks_to_milvus(cfg: Dict[str, Any]) -> None:
    base_dir = os.path.abspath(cfg["processed_data_dir"])  # ../data/format_data_with_img
    pathmap = read_pathmap(os.path.abspath(cfg["pathmap_path"]))

    connect_milvus(cfg)
    coll = ensure_milvus_collection(cfg["milvus"]["collection"], cfg["embed_dim"], cfg)

    encoder = GTEEncoder(cfg["embed_model_name"])  # 可替换为更轻模型

    chunk_size = cfg["chunk_size"]
    overlap = cfg["chunk_overlap"]

    to_insert = {"file_path": [], "dir": [], "know_path": [], "embedding": []}

    for txt_path in tqdm(iter_txt_files(base_dir), desc="Index Milvus chunks"):
        try:
            content = open(txt_path, "r", encoding="utf-8").read()
        except Exception:
            content = open(txt_path, "r", encoding="gb2312").read()
        file_path_rel = os.path.relpath(txt_path, base_dir)
        parts = file_path_rel.split(os.sep)
        dir_name = parts[0] if parts else ""
        know_path_list = pathmap.get("/".join([dir_name, file_path_rel.replace(os.sep, "/")]), [])
        know_path_text = "/".join(know_path_list) if know_path_list else ""

        chunks = chunk_text(content, chunk_size, overlap)
        if not chunks:
            continue
        embs = encoder.encode(chunks)
        for emb in embs:
            to_insert["file_path"].append(file_path_rel.replace(os.sep, "/"))
            to_insert["dir"].append(dir_name)
            to_insert["know_path"].append(know_path_text)
            to_insert["embedding"].append(emb)

        if len(to_insert["file_path"]) >= 512:
            coll.insert([to_insert["file_path"], to_insert["dir"], to_insert["know_path"], to_insert["embedding"]])
            to_insert = {"file_path": [], "dir": [], "know_path": [], "embedding": []}

    if to_insert["file_path"]:
        coll.insert([to_insert["file_path"], to_insert["dir"], to_insert["know_path"], to_insert["embedding"]])

    coll.flush()


# ======== 检索 ========

def es_bm25_text_search(cfg: Dict[str, Any], query: str, dir_filter: str | None = None, topk: int = 192) -> List[Dict[str, Any]]:
    es = build_es_client(cfg)
    index_name = cfg["elasticsearch"]["index_text"]
    must: List[Dict[str, Any]] = [
        {"multi_match": {"query": query, "fields": ["content^1", "tokens^2"]}}
    ]
    if dir_filter:
        must.append({"term": {"dir": dir_filter}})
    body = {"query": {"bool": {"must": must}}}
    res = es.search(index=index_name, body=body, size=topk)
    hits = res.get("hits", {}).get("hits", [])
    return [h["_source"] | {"_score": h.get("_score", 0.0)} for h in hits if h.get("_source")]


def es_bm25_path_search(cfg: Dict[str, Any], query: str, dir_filter: str | None = None, topk: int = 6) -> List[Dict[str, Any]]:
    es = build_es_client(cfg)
    index_name = cfg["elasticsearch"]["index_path"]
    must: List[Dict[str, Any]] = [{"match": {"know_path": query}}]
    if dir_filter:
        must.append({"term": {"dir": dir_filter}})
    body = {"query": {"bool": {"must": must}}}
    res = es.search(index=index_name, body=body, size=topk)
    hits = res.get("hits", {}).get("hits", [])
    items = []
    for h in hits:
        src = h.get("_source", {})
        src["_score"] = h.get("_score", 0.0)
        items.append(src)
    return items


def milvus_dense_search(cfg: Dict[str, Any], query: str, dir_filter: str | None = None, topk: int = 288) -> List[Dict[str, Any]]:
    connect_milvus(cfg)
    coll = Collection(cfg["milvus"]["collection"])
    encoder = GTEEncoder(cfg["embed_model_name"])  # 简化：每次构造，可优化为复用
    qvec = encoder.encode([query])[0]

    # Milvus 查询表达式
    expr = None
    if dir_filter:
        expr = f"dir == '{dir_filter}'"

    res = coll.search(
        data=[qvec],
        anns_field="embedding",
        param={"metric_type": cfg["milvus"]["metric_type"], "params": {"nprobe": 16}},
        limit=topk,
        expr=expr,
        output_fields=["file_path", "dir", "know_path"],
    )
    out: List[Dict[str, Any]] = []
    for hit in res[0]:
        out.append({
            "file_path": hit.entity.get("file_path"),
            "dir": hit.entity.get("dir"),
            "know_path": hit.entity.get("know_path"),
            "_score": float(hit.distance),
        })
    return out


# ======== 融合（simple / RRF） ========

def rrf_fuse(results_lists: List[List[Dict[str, Any]]], k: float = 60.0) -> List[Tuple[str, float, Dict[str, Any]]]:
    rank_map: Dict[str, float] = {}
    meta_map: Dict[str, Dict[str, Any]] = {}
    for results in results_lists:
        for rank, item in enumerate(results, start=1):
            key = item.get("file_path", item.get("know_path", ""))
            if not key:
                continue
            score = 1.0 / (k + rank)
            rank_map[key] = rank_map.get(key, 0.0) + score
            if key not in meta_map:
                meta_map[key] = item
    fused = [(k, v, meta_map[k]) for k, v in rank_map.items()]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


def simple_merge(results_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    seen = set()
    merged: List[Dict[str, Any]] = []
    for results in results_lists:
        for item in results:
            key = item.get("file_path", item.get("know_path", ""))
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
    return merged


# ======== 精排（bge reranker 简化示例：使用点积/余弦） ========
class SimpleReranker:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def score(self, query: str, docs: List[str]) -> List[float]:
        tok = self.tokenizer([query] + docs, padding=True, truncation=True, return_tensors="pt")
        out = self.model(**tok)
        emb = out.last_hidden_state.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        q = emb[0:1]
        d = emb[1:]
        scores = (q @ d.T).squeeze(0).cpu().tolist()
        return scores


# ======== 问答 ========

def build_context(docs: List[str]) -> str:
    lines = []
    for i, ch in enumerate(docs):
        lines.append(f"### 文档{i}: {ch}")
    return "\n".join(lines)


def qa_prompt(context: str, query: str) -> str:
    return (
        "上下文信息如下：\n----------\n"
        + context
        + "\n----------\n"
        + "请你基于上下文信息而不是自己的知识，回答以下问题，可以分点作答，如果上下文信息没有相关知识，可以回答不确定，不要复述上下文信息：\n"
        + query
        + "\n回答："
    )


def generate_answer(cfg: Dict[str, Any], prompt: str) -> str:
    gen_conf = cfg.get("generation", {}) or {}
    backend = gen_conf.get("backend", "none")
    temperature = float(gen_conf.get("temperature", 0.2))
    max_tokens = int(gen_conf.get("max_tokens", 1024))
    top_p = float(gen_conf.get("top_p", 0.9))

    if backend == "vllm":
        try:
            from openai import OpenAI
            vconf = cfg.get("vllm", {}) or {}
            base_url = vconf.get("base_url", "http://localhost:8000/v1")
            api_key = vconf.get("api_key", "EMPTY")
            model = vconf.get("model", cfg.get("llm_name", ""))
            client = OpenAI(base_url=base_url, api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"vLLM 调用失败: {e}")
            return ""
    elif backend == "ollama":
        try:
            import requests
            oconf = cfg.get("ollama", {}) or {}
            host = oconf.get("host", "http://localhost:11434")
            model = oconf.get("model", "qwen2.5:7b-instruct")
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
            }
            r = requests.post(f"{host}/api/chat", json=data, timeout=180)
            r.raise_for_status()
            j = r.json()
            return (j.get("message", {}) or {}).get("content") or j.get("response", "") or ""
        except Exception as e:
            print(f"Ollama 调用失败: {e}")
            return ""
    else:
        # 不生成，返回空字符串
        return ""


# ======== 教学主流程（索引与查询） ========

def build_indices(config_path: str = "configs/es_milvus.yaml"):
    cfg = load_yaml(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))
    # ES 两个索引
    index_text_chunks_to_es(cfg)
    index_paths_to_es(cfg)
    # Milvus 向量索引
    index_text_chunks_to_milvus(cfg)


def retrieve(query_obj: Dict[str, Any], config_path: str = "configs/es_milvus.yaml") -> Dict[str, Any]:
    cfg = load_yaml(os.path.abspath(os.path.join(os.path.dirname(__file__), config_path)))
    query = query_obj["query"]
    dir_filter = query_obj.get("document")  # 比赛元数据的来源过滤

    # 稀疏：文本块检索 + 路径检索
    sparse_text = es_bm25_text_search(cfg, query, dir_filter, topk=cfg["bm25_topk_text"])
    sparse_path = es_bm25_path_search(cfg, query, dir_filter, topk=cfg["bm25_topk_path"])

    # 密集
    dense = milvus_dense_search(cfg, query, dir_filter, topk=288)

    # 粗排融合
    if cfg.get("coarse_fusion", "simple") == "rrf":
        fused_keys = rrf_fuse([sparse_text, sparse_path, dense])
        # 这里仅返回键与分数，真实系统需回填内容；教学版后续用精排再拉取内容
        # 简化：先用 simple 合并保留内容
        candidates = simple_merge([sparse_text, dense])
    else:
        candidates = simple_merge([sparse_text, dense])

    # 精排：取前若干做 rerank
    rerank_topk = cfg["rerank_topk"]
    # 取候选的文本内容字段（ES 的 content；Milvus 只带路径，需要二次加载文本，教学简化：忽略仅 Milvus 但未在 ES 里的条目）
    contents = [c["content"] for c in candidates if "content" in c]
    contents = contents[:max(rerank_topk * 4, rerank_topk)]

    if not contents:
        return {"answer": "不确定", "contexts": [], "nodes": [], "references": []}

    reranker = SimpleReranker(cfg["reranker_name"]) if os.path.exists(cfg["reranker_name"]) else SimpleReranker("BAAI/bge-small-zh-v1.5")
    scores = reranker.score(query, contents)
    pairs = list(zip(contents, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    selected_texts = [t for t, _ in pairs[:cfg["final_context_topk"]]]

    # 依据选中文本回填引用信息（file_path、know_path）
    selected_items: List[Dict[str, Any]] = []
    used_idx = set()
    for txt in selected_texts:
        found = False
        for idx, item in enumerate(candidates):
            if idx in used_idx:
                continue
            if item.get("content") == txt:
                selected_items.append(item)
                used_idx.add(idx)
                found = True
                break
        if not found:
            # 找不到来源时，构造一个占位
            selected_items.append({"file_path": "", "know_path": "", "dir": "", "content": txt})

    context = build_context([it.get("content", "") for it in selected_items])
    prompt = qa_prompt(context, query)

    # 生成答案（支持 vLLM / Ollama）
    answer_text = generate_answer(cfg, prompt)

    # 组织参考信息
    references = []
    seen_paths = set()
    for it in selected_items:
        fp = it.get("file_path", "")
        kp = it.get("know_path", "")
        if fp or kp:
            key = fp or kp
            if key in seen_paths:
                continue
            seen_paths.add(key)
        references.append({
            "file_path": fp,
            "know_path": kp,
            "dir": it.get("dir", ""),
            "snippet": (it.get("content", "") or "")[:300],
        })

    return {
        "prompt": prompt,
        "contexts": [it.get("content", "") for it in selected_items],
        "nodes": [],
        "references": references,
        "answer": answer_text or "",  # 若生成失败则为空串
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["index", "query"])  # index -> 构建索引； query -> 单条查询
    parser.add_argument("--config", default="configs/es_milvus.yaml")
    parser.add_argument("--query_json", default=None)
    args = parser.parse_args()

    if args.action == "index":
        build_indices(args.config)
    else:
        if not args.query_json:
            raise SystemExit("--query_json required for query action")
        q = json.loads(open(args.query_json, "r", encoding="utf-8").read())
        res = retrieve(q, args.config)
        print(json.dumps(res, ensure_ascii=False, indent=2))