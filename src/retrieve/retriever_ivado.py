from typing import List, Dict, Tuple, Optional
import os
import time
import tqdm
import uuid #for unique document IDs 
import numpy as np
import torch
import faiss # for vector search
import logging
import pandas as pd
import pickle
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
from pathlib import Path


import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "beir"))
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search
from beir.retrieval.search.lexical.elastic_search import ElasticSearch

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

def get_random_doc_id():
    return f'_{uuid.uuid4()}'

class BM25:
    def __init__(
        self,
        tokenizer: AutoTokenizer = None,
        index_name: str = None,
        engine: str = 'elasticsearch',
        **search_engine_kwargs,
    ):
        self.tokenizer = tokenizer
        # load index
        assert engine in {'elasticsearch', 'bing'}
        if engine == 'elasticsearch':
            self.max_ret_topk = 1000
            self.retriever = EvaluateRetrieval(
                PatchedBM25Search(index_name=index_name, hostname='http://localhost:9200', initialize=False, number_of_shards=1),
                k_values=[self.max_ret_topk])

    def retrieve(
        self,
        queries: List[str],  # (bs,)
        topk: int = 1,
        max_query_length: int = None,
        return_scores: bool = False,
    ):
        assert topk <= self.max_ret_topk
        device = None
        bs = len(queries)

        # truncate queries
        if max_query_length:
            ori_ps = self.tokenizer.padding_side
            ori_ts = self.tokenizer.truncation_side
            # truncate/pad on the left side
            self.tokenizer.padding_side = 'left'
            self.tokenizer.truncation_side = 'left'
            tokenized = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                max_length=max_query_length,
                add_special_tokens=False,
                return_tensors='pt')['input_ids']
            self.tokenizer.padding_side = ori_ps
            self.tokenizer.truncation_side = ori_ts
            queries = self.tokenizer.batch_decode(tokenized, skip_special_tokens=True)

        # retrieve
        results: Dict[str, Dict[str, Tuple[float, str]]] = self.retriever.retrieve(
            None, dict(zip(range(len(queries)), queries)), disable_tqdm=True)

        # prepare outputs
        docids: List[str] = []
        docs: List[str] = []
        scores: List[float] = []
        for qid, query in enumerate(queries):
            _docids: List[str] = []
            _docs: List[str] = []
            _scores: List[float] = []
            if qid in results:
                for did, (score, text) in results[qid].items():
                    _docids.append(did)
                    _docs.append(text)
                    _scores.append(float(score))
                    if len(_docids) >= topk:
                        break
            if len(_docids) < topk:  # add dummy docs
                _docids += [get_random_doc_id() for _ in range(topk - len(_docids))]
                _docs += [''] * (topk - len(_docs))
                _scores += [0.0] * (topk - len(_scores))
            docids.extend(_docids)
            docs.extend(_docs)
            scores.extend(_scores)

        docids = np.array(docids).reshape(bs, topk)  # (bs, topk)
        docs = np.array(docs).reshape(bs, topk)  # (bs, topk)
        if return_scores:
            scores = np.array(scores).reshape(bs, topk)  # (bs, topk)
            return docids, docs, scores
        return docids, docs


def bm25search_search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
    # Index the corpus within elastic-search
    # False, if the corpus has been already indexed
    if self.initialize:
        self.index(corpus)
        # Sleep for few seconds so that elastic-search indexes the docs properly
        time.sleep(self.sleep_for)

    #retrieve results from BM25
    query_ids = list(queries.keys())
    queries = [queries[qid] for qid in query_ids]

    final_results: Dict[str, Dict[str, Tuple[float, str]]] = {}
    for start_idx in tqdm.trange(0, len(queries), self.batch_size, desc='que', disable=kwargs.get('disable_tqdm', False)):
        query_ids_batch = query_ids[start_idx:start_idx+self.batch_size]
        results = self.es.lexical_multisearch(
            texts=queries[start_idx:start_idx+self.batch_size],
            top_hits=top_k)
        for (query_id, hit) in zip(query_ids_batch, results):
            scores = {}
            for corpus_id, score, text in hit['hits']:
                scores[corpus_id] = (score, text)
                final_results[query_id] = scores

    return final_results

BM25Search.search = bm25search_search

def bm25search_encode(self, texts, *args, **kwargs):
    """Fallback encode implementation to satisfy abstract base requirements.
    Returns the input texts unchanged which is sufficient for lexical BM25 retriever usage."""
    return texts


def bm25search_search_from_files(self, corpus, queries, top_k, *args, **kwargs):
    """Fallback search_from_files implementation that delegates to `search`.

    Accepts either a corpus dict or a path-like object; if a dict is provided it is used directly,
    otherwise delegation happens with an empty corpus (caller should ensure corpus is indexed).
    """
    try:
        if isinstance(corpus, dict):
            return self.search(corpus, queries, top_k, *args, **kwargs)
    except Exception:
        pass
    return self.search({}, queries, top_k, *args, **kwargs)


BM25Search.encode = bm25search_encode
BM25Search.search_from_files = bm25search_search_from_files


class PatchedBM25Search(BM25Search):
    """Concrete subclass of BEIR's BM25Search that implements abstract methods.

    This avoids instantiation errors from the ABC machinery in some BEIR versions.
    """
    def encode(self, texts, *args, **kwargs):
        return bm25search_encode(self, texts, *args, **kwargs)

    def search_from_files(self, corpus, queries, top_k, *args, **kwargs):
        return bm25search_search_from_files(self, corpus, queries, top_k, *args, **kwargs)


def elasticsearch_lexical_multisearch(self, texts: List[str], top_hits: int, skip: int = 0) -> Dict[str, object]:
    """Multiple Query search in Elasticsearch

    Args:
        texts (List[str]): Multiple query texts
        top_hits (int): top k hits to be retrieved
        skip (int, optional): top hits to be skipped. Defaults to 0.

    Returns:
        Dict[str, object]: Hit results
    """
    request = []

    assert skip + top_hits <= 10000, "Elastic-Search Window too large, Max-Size = 10000"

    for text in texts:
        req_head = {"index" : self.index_name, "search_type": "dfs_query_then_fetch"}
        req_body = {
            "_source": True, # No need to return source objects
            "query": {
                "multi_match": {
                    "query": text, # matching query with both text and title fields
                    "type": "best_fields",
                    "fields": [self.title_key, self.text_key],
                    "tie_breaker": 0.5
                    }
                },
            "size": skip + top_hits, # The same paragraph will occur in results
            }
        request.extend([req_head, req_body])

    res = self.es.msearch(body = request)

    result = []
    for resp in res["responses"]:
        responses = resp["hits"]["hits"][skip:] if 'hits' in resp else []

        hits = []
        for hit in responses:
            hits.append((hit["_id"], hit['_score'], hit['_source']['txt']))

        result.append(self.hit_template(es_res=resp, hits=hits))
    return result

ElasticSearch.lexical_multisearch = elasticsearch_lexical_multisearch


def elasticsearch_hit_template(self, es_res: Dict[str, object], hits: List[Tuple[str, float]]) -> Dict[str, object]:
    """Hit output results template

    Args:
        es_res (Dict[str, object]): Elasticsearch response
        hits (List[Tuple[str, float]]): Hits from Elasticsearch

    Returns:
        Dict[str, object]: Hit results
    """
    result = {
        'meta': {
            'total': es_res['hits']['total']['value'] if 'hits' in es_res else None,
            'took': es_res['took'] if 'took' in es_res else None,
            'num_hits': len(hits)
        },
        'hits': hits,
    }
    return result

ElasticSearch.hit_template = elasticsearch_hit_template


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
bm25_retriever = BM25(
    tokenizer = tokenizer, 
    index_name = "wiki", 
    engine = "elasticsearch",
)

def bm25_retrieve(question, topk):
    docs_ids, docs = bm25_retriever.retrieve(
        [question], 
        topk=topk, 
        max_query_length=256
    )
    return docs[0].tolist()


class SGPT:
    cannot_encode_id = [6799132, 6799133, 6799134, 6799135, 6799136, 6799137, 6799138, 6799139, 8374206, 8374223, 9411956, 
        9885952, 11795988, 11893344, 12988125, 14919659, 16890347, 16898508]
    # 这些向量是 SGPT 不能 encode 的，设置为全 0 向量，点积为 0，不会影响检索

    def __init__(
        self, 
        model_name_or_path,
        sgpt_encode_file_path,
        passage_file
    ):
        logger.info("Loading SGPT model from %s", model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, device_map="auto")
        self.model.eval()
        self.SPECB_QUE_BOS = self.tokenizer.encode("[", add_special_tokens=False)[0]
        self.SPECB_QUE_EOS = self.tokenizer.encode("]", add_special_tokens=False)[0]
        self.SPECB_DOC_BOS = self.tokenizer.encode("{", add_special_tokens=False)[0]
        self.SPECB_DOC_EOS = self.tokenizer.encode("}", add_special_tokens=False)[0]

        logger.info(f"Building SGPT indexes")

        self.p_reps = []

        encode_file_path = sgpt_encode_file_path
        dir_names = sorted(os.listdir(encode_file_path))
        dir_point = 0
        pbar = tqdm.tqdm(total=len(dir_names))
        split_parts = 0
        while True:
            split_parts += 1
            flag = False
            for d in dir_names:
                if d.startswith(f'{split_parts}_'):
                    flag = True
                    break
            if flag == False:
                break

        for i in range(split_parts):
            start_point = dir_point
            while dir_point < len(dir_names) and dir_names[dir_point].startswith(f'{i}_'):
                # filename = dir_names[dir_point]
                dir_point += 1
            cnt = dir_point - start_point
            for j in range(cnt):
                filename = f"{i}_{j}.pt"
                pbar.update(1)
                tp = torch.load(os.path.join(encode_file_path, filename))

                def get_norm(matrix):
                    norm = matrix.norm(dim=1)
                    if 0 in norm:
                        norm = torch.where(norm == 0, torch.tensor(1.0), norm)
                    return norm.view(-1, 1)

                sz = tp.shape[0] // 2
                tp1 = tp[:sz, :]
                tp2 = tp[sz:, :]
                self.p_reps.append((tp1.cuda(i), get_norm(tp1).cuda(i)))
                self.p_reps.append((tp2.cuda(i), get_norm(tp2).cuda(i)))
            
        docs_file = passage_file
        df = pd.read_csv(docs_file, delimiter='\t')
        self.docs = list(df['text'])


    def tokenize_with_specb(self, texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)   
        # Add special brackets & pay attention to them
        for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
            if is_query:
                seq.insert(0, self.SPECB_QUE_BOS)
                seq.append(self.SPECB_QUE_EOS)
            else:
                seq.insert(0, self.SPECB_DOC_BOS)
                seq.append(self.SPECB_DOC_EOS)
            att.insert(0, 1)
            att.append(1)
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        return batch_tokens

    def get_weightedmean_embedding(self, batch_tokens):
        # Get the embeddings
        with torch.no_grad():
            # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state = self.model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings

    def retrieve(
        self, 
        queries: List[str], 
        topk: int = 1,
        return_scores: bool = False,
    ):
        q_reps = self.get_weightedmean_embedding(
            self.tokenize_with_specb(queries, is_query=True)
        )
        q_reps.requires_grad_(False)
        q_reps_trans = torch.transpose(q_reps, 0, 1)

        topk_values_list = []
        topk_indices_list = []
        prev_count = 0
        for p_rep, p_rep_norm in self.p_reps:
            sim = p_rep @ q_reps_trans.to(p_rep.device)
            sim = sim / p_rep_norm
            # print(sim.shape)
            topk_values, topk_indices = torch.topk(sim, k=topk, dim=0)
            # print(torch.transpose(topk_values, 0, 1)[0])
            # print(torch.transpose(topk_indices, 0, 1)[0] + prev_count)
            topk_values_list.append(topk_values.to('cpu'))
            topk_indices_list.append(topk_indices.to('cpu') + prev_count)
            prev_count += p_rep.shape[0]

        all_topk_values = torch.cat(topk_values_list, dim=0)
        global_topk_values, global_topk_indices = torch.topk(all_topk_values, k=topk, dim=0)

        psgs = []
        scores = []
        for qid in range(q_reps.shape[0]):
            ret = []
            ret_scores = []
            for j in range(topk):
                idx = global_topk_indices[j][qid].item()
                fid, rk = idx // topk, idx % topk
                psg = self.docs[topk_indices_list[fid][rk][qid]]
                ret.append(psg)
                ret_scores.append(float(global_topk_values[j][qid].item()))
            psgs.append(ret)
            scores.append(ret_scores)
        if return_scores:
            return psgs, scores
        return psgs
#############################################
## for ivado retrieval
##############################################

class FaissRetriever:
    def __init__(
        self,
        index_path: str,
        pkl_path: str,
        text_key: str = "content",
        id_key: str = "chunk_id",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        normalize_embeddings: bool = True,
        id_map_path: Optional[str] = None,
    ):
        index_paths = self._ensure_list(index_path)
        pkl_paths = self._ensure_list(pkl_path)
        id_map_paths = self._ensure_list(id_map_path) if id_map_path else []

        if len(index_paths) == 0:
            raise ValueError("At least one FAISS index path is required")
        if len(pkl_paths) not in {1, len(index_paths)}:
            raise ValueError("faiss_pkl_path must be a single path or match the number of indexes")
        if id_map_paths and len(id_map_paths) not in {1, len(index_paths)}:
            raise ValueError("faiss_id_map_path must be a single path or match the number of indexes")

        if len(pkl_paths) == 1 and len(index_paths) > 1:
            pkl_paths = pkl_paths * len(index_paths)
        if id_map_paths and len(id_map_paths) == 1 and len(index_paths) > 1:
            id_map_paths = id_map_paths * len(index_paths)

        self.text_key = text_key
        self.id_key = id_key
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(embedding_model)

        self.bundles = []
        for i, idx_path in enumerate(index_paths):
            bundle = self._load_bundle(
                index_path=idx_path,
                pkl_path=pkl_paths[i],
                id_map_path=id_map_paths[i] if id_map_paths else None,
            )
            self.bundles.append(bundle)

        # NEW: tag each bundle with its language, inferred from its pkl path
        from lang_utils import infer_bundle_language
        self.bundle_languages = [infer_bundle_language(p) for p in pkl_paths]

        self.bundle_paths = list(pkl_paths)
        logger.info("=" * 60)
        logger.info("FaissRetriever (DENSE) loaded %d bundle(s):", len(self.bundles))
        for path, lang, bundle in zip(self.bundle_paths, self.bundle_languages, self.bundles):
            logger.info("  [lang=%s] %s  (%d vectors)", lang, path, bundle["index"].ntotal)
        logger.info("=" * 60)




        # Keep single-index aliases for backward compatibility
        self.index = self.bundles[0]["index"]
        self.docs = self.bundles[0]["docs"]
        self.doc_ids = self.bundles[0]["doc_ids"]
        self.doc_id_to_idx = self.bundles[0]["doc_id_to_idx"]
        self.id_map = self.bundles[0]["id_map"]

    # def _normalize_records(self, raw):
    #     if isinstance(raw, list):
    #         return raw
    #     if isinstance(raw, dict):
    #         if "data" in raw and isinstance(raw["data"], list):
    #             return raw["data"]
    #         return list(raw.values())
    #     raise ValueError("Unsupported PKL data structure for chunks")
    def _normalize_records(self, raw):
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if "chunks" in raw and isinstance(raw["chunks"], list):
                return raw["chunks"]
            if "data" in raw and isinstance(raw["data"], list):
                return raw["data"]
            return list(raw.values())
        raise ValueError("Unsupported PKL data structure for chunks")

    def _record_text(self, record):
        if isinstance(record, dict):
            return str(record.get(self.text_key, ""))
        if isinstance(record, (list, tuple)):
            if not record:
                return ""
            last = record[-1]
            if isinstance(last, dict):
                return str(last.get(self.text_key, ""))
            return str(last)
        return str(record)

    def _record_id(self, record, fallback_id):
        if isinstance(record, dict):
            return str(record.get(self.id_key, fallback_id))
        if isinstance(record, (list, tuple)) and record:
            first = record[0]
            if isinstance(first, dict):
                return str(first.get(self.id_key, fallback_id))
            return str(first)
        return str(fallback_id)

    def _load_records_from_path(self, path: str):
        if path.endswith((".pkl", ".pickle")):
            with open(path, "rb") as f:
                raw = pickle.load(f)
            return self._normalize_records(raw)
        if path.endswith(".json"):
            with open(path, "r") as f:
                raw = json.load(f)
            return self._normalize_records(raw)
        if path.endswith(".jsonl"):
            records = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            return self._normalize_records(records)
        raise ValueError(f"Unsupported chunk file format: {path}")

    def _map_faiss_id(self, faiss_id: int, id_map=None, doc_id_to_idx=None) -> Optional[int]:
        if faiss_id < 0:
            return None
        id_map = self.id_map if id_map is None else id_map
        doc_id_to_idx = self.doc_id_to_idx if doc_id_to_idx is None else doc_id_to_idx
        if id_map is None:
            return int(faiss_id)
        if isinstance(id_map, dict):
            mapped = id_map.get(str(faiss_id), id_map.get(int(faiss_id)))
            if mapped is None:
                return None
            if isinstance(mapped, str):
                return doc_id_to_idx.get(mapped)
            return int(mapped)
        if isinstance(id_map, list):
            if 0 <= faiss_id < len(id_map):
                mapped = id_map[faiss_id]
                if isinstance(mapped, str):
                    return doc_id_to_idx.get(mapped)
                return int(mapped)
        return None

    def _ensure_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _is_higher_better(self, index) -> bool:
        metric_type = getattr(index, "metric_type", faiss.METRIC_INNER_PRODUCT)
        return metric_type != faiss.METRIC_L2

    def _load_bundle(self, index_path: str, pkl_path: str, id_map_path: Optional[str]):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Chunk data not found: {pkl_path}")

        logger.info("Loading FAISS index from %s", index_path)

        index = faiss.read_index(index_path)

        logger.info("Loading chunks from %s", pkl_path)
        records = self._load_records_from_path(pkl_path)
        docs = [self._record_text(r) for r in records]
        doc_ids = [self._record_id(r, i) for i, r in enumerate(records)]
        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

        id_map = None
        if id_map_path:
            if not os.path.exists(id_map_path):
                raise FileNotFoundError(f"FAISS id map not found: {id_map_path}")
            logger.info("Loading FAISS id map from %s", id_map_path)
            if id_map_path.endswith((".pkl", ".pickle")):
                with open(id_map_path, "rb") as f:
                    id_map = pickle.load(f)
            else:
                with open(id_map_path, "r") as f:
                    id_map = json.load(f)

        return {
            "index": index,
            "docs": docs,
            "doc_ids": doc_ids,
            "doc_id_to_idx": doc_id_to_idx,
            "id_map": id_map,
            "higher_is_better": self._is_higher_better(index),
        }


    def retrieve(
        self,
        queries,
        topk: int = 1,
        return_scores: bool = False,
        over_fetch_k=None,
        langs=None,
    ):
        if over_fetch_k is None:
            over_fetch_k = max(topk * 20, 50)
    
        q_emb = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype(np.float32)
    
        all_candidates = [[] for _ in range(len(queries))]
        for bundle, bundle_lang in zip(self.bundles, self.bundle_languages):
            for qid in range(len(queries)):
                # language filter: skip bundles that don't match this query's
                # language, unless langs is None (search everything) or the
                # bundle's language is unknown (safety fallback)
                if langs is not None and bundle_lang not in (langs[qid], "unknown", None):
                    continue
    
                k = min(over_fetch_k, bundle["index"].ntotal)
                if k == 0:
                    continue
                distances, indices = bundle["index"].search(q_emb[qid:qid + 1], k)
                for i in range(k):
                    faiss_id = int(indices[0][i])
                    mapped_idx = self._map_faiss_id(
                        faiss_id,
                        id_map=bundle["id_map"],
                        doc_id_to_idx=bundle["doc_id_to_idx"],
                    )
                    if mapped_idx is None or mapped_idx >= len(bundle["docs"]):
                        continue
                    score = float(distances[0][i])
                    rank_score = score if bundle["higher_is_better"] else -score
                    all_candidates[qid].append((rank_score, score, bundle["docs"][mapped_idx]))
    
        all_docs = []
        all_scores = []
        for qid in range(len(queries)):
            candidates = sorted(all_candidates[qid], key=lambda x: x[0], reverse=True)
            docs, scores, seen_texts = [], [], set()
            for rank_score, score, text in candidates:
                if not text or not text.strip() or text in seen_texts:
                    continue
                seen_texts.add(text)
                docs.append(text)
                scores.append(score)
                if len(docs) == topk:
                    break
            if len(docs) < topk:
                missing = topk - len(docs)
                docs.extend([""] * missing)
                scores.extend([0.0] * missing)
            all_docs.append(docs)
            all_scores.append(scores)
            
    
        if return_scores:
            return all_docs, all_scores
        return all_docs

    # def retrieve(
    #     self,
    #     queries: List[str],
    #     topk: int = 1,
    #     return_scores: bool = False,
    # ):
    #     q_emb = self.model.encode(
    #         queries,
    #         convert_to_numpy=True,
    #         normalize_embeddings=self.normalize_embeddings,
    #     ).astype(np.float32)

    #     all_candidates = [[] for _ in range(len(queries))]
    #     for bundle in self.bundles:
    #         distances, indices = bundle["index"].search(q_emb, topk)
    #         for qid in range(len(queries)):
    #             for i in range(topk):
    #                 faiss_id = int(indices[qid][i])
    #                 mapped_idx = self._map_faiss_id(
    #                     faiss_id,
    #                     id_map=bundle["id_map"],
    #                     doc_id_to_idx=bundle["doc_id_to_idx"],
    #                 )
    #                 if mapped_idx is None or mapped_idx >= len(bundle["docs"]):
    #                     continue
    #                 score = float(distances[qid][i])
    #                 rank_score = score if bundle["higher_is_better"] else -score
    #                 all_candidates[qid].append((rank_score, score, bundle["docs"][mapped_idx]))

    #     all_docs = []
    #     all_scores = []
    #     for qid in range(len(queries)):
    #         candidates = sorted(all_candidates[qid], key=lambda x: x[0], reverse=True)
    #         docs = [c[2] for c in candidates[:topk]]
    #         scores = [c[1] for c in candidates[:topk]]
    #         if len(docs) < topk:
    #             missing = topk - len(docs)
    #             docs.extend([""] * missing)
    #             scores.extend([0.0] * missing)
    #         all_docs.append(docs)
    #         all_scores.append(scores)

    #     if return_scores:
    #         return all_docs, all_scores
    #     return all_docs


"""
It mirrors FaissRetriever's interface (retrieve(queries, topk, return_scores))
so BasicRAG/SingleRAG can use either retriever_type interchangeably.
"""



class SparseRetrieverIvado:
    """
    Loads one or more `sparse_index.pkl` files (as produced by
    src/sparse_index.py's SparseIndexer.save()) and exposes a
    retrieve() interface matching FaissRetriever, including:
      - multi-bundle support (e.g. macommunaute + 211)
      - over-fetch + exact-text dedup when merging across bundles
      - optional language filtering via `langs=` (see bundle_languages)
    """

    def __init__(
        self,
        sparse_index_path,
        text_key: str = "content",
        id_key: str = "chunk_id",
    ):
        self.text_key = text_key
        self.id_key = id_key

        paths = self._ensure_list(sparse_index_path)
        if len(paths) == 0:
            raise ValueError("At least one sparse_index_path is required")
        print("#################### selecting sparse method ########################")
        print("#################### sparse_index_path ########################", sparse_index_path )
        logger.info("#################### sparse_index_path: %s ########################", sparse_index_path)

        self.bundles = []
        self.bundle_languages = []
        for p in paths:
            bundle = self._load_bundle(p)
            self.bundles.append(bundle)
            self.bundle_languages.append(bundle["language"] or self._infer_lang_from_path(p))

        # Backward-compatible single-bundle aliases
        self.method = self.bundles[0]["method"]
        self.language = self.bundles[0]["language"]
        self.chunks = self.bundles[0]["chunks"]
        self.index = self.bundles[0]["index"]

    def _ensure_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _infer_lang_from_path(self, path: str) -> str:
        parts = Path(path).parts
        if "en" in parts:
            return "en"
        if "fr" in parts:
            return "fr"
        return "unknown"

    def _load_bundle(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sparse index not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)

        method = data.get("method", "bm25")
        language = data.get("language")
        chunks = data.get("chunks", [])
        index = data.get("index")
        vectorizer = data.get("vectorizer")  # only present for tfidf

        # Rebuild the tokenizer the same way SparseIndexer does, since it's
        # not pickled (it's a lambda / spacy pipeline object).
        tokenizer = self._build_tokenizer(language)

        return {
            "method": method,
            "language": language,
            "chunks": chunks,
            "index": index,
            "vectorizer": vectorizer,
            "tokenizer": tokenizer,
        }

    def _build_tokenizer(self, language: str):
        # Mirrors SparseIndexer._get_tokenizer() in src/sparse_index.py
        if language == "zh":
            try:
                import jieba
                return lambda text: jieba.lcut(text)
            except ImportError:
                return lambda text: list(text)
        elif language == "fr":
            try:
                import spacy
                try:
                    nlp = spacy.load("fr_core_news_sm")
                    return lambda text: [
                        token.text.lower() for token in nlp(text)
                        if not token.is_punct and not token.is_space
                    ]
                except OSError:
                    return lambda text: (
                        text.lower().replace(",", " ").replace(".", " ")
                        .replace(";", " ").split()
                    )
            except ImportError:
                return lambda text: (
                    text.lower().replace(",", " ").replace(".", " ")
                    .replace(";", " ").replace(":", " ").split()
                )
        else:
            return lambda text: text.lower().split()

    def retrieve(
        self,
        queries: List[str],
        topk: int = 1,
        return_scores: bool = False,
        over_fetch_k: Optional[int] = None,
        langs: Optional[List[str]] = None,
    ):
        """
        Args:
            queries: list of query strings
            topk: final number of results to return per query
            return_scores: if True, also return per-result scores
            over_fetch_k: how many candidates to pull from each bundle
                before merging (default: max(topk*20, 50), mirrors the
                dense retriever's over-fetch fix)
            langs: optional per-query language filter (e.g. ["fr"]) --
                only bundles whose language matches are searched for that
                query. Bundles with unknown language are always searched
                as a fallback.
        """
        if over_fetch_k is None:
            over_fetch_k = max(topk * 20, 50)

        all_candidates = [[] for _ in range(len(queries))]

        for bundle in self.bundles:
            bundle_lang = bundle["language"]
            tokenizer = bundle["tokenizer"]
            chunks = bundle["chunks"]
            index = bundle["index"]
            method = bundle["method"]
            vectorizer = bundle.get("vectorizer")

            for qid, query in enumerate(queries):
                if langs is not None and bundle_lang not in (langs[qid], "unknown", None):
                    continue

                query_tokens = tokenizer(query)

                if method == "bm25":
                    scores = index.get_scores(query_tokens)
                elif method == "tfidf":
                    query_vector = vectorizer.transform([" ".join(query_tokens)])
                    scores = (index * query_vector.T).toarray().flatten()
                else:
                    raise ValueError(f"Unsupported sparse method: {method}")

                k = min(over_fetch_k, len(chunks))
                if k == 0:
                    continue
                top_indices = scores.argsort()[::-1][:k]

                for idx in top_indices:
                    score = float(scores[idx])
                    if score <= 0:
                        continue
                    text = str(chunks[idx].get(self.text_key, ""))
                    all_candidates[qid].append((score, score, text))

        all_docs, all_scores = [], []
        for qid in range(len(queries)):
            candidates = sorted(all_candidates[qid], key=lambda x: x[0], reverse=True)
            docs, scores_out, seen_texts = [], [], set()
            for rank_score, score, text in candidates:
                if not text or not text.strip() or text in seen_texts:
                    continue
                seen_texts.add(text)
                docs.append(text)
                scores_out.append(score)
                if len(docs) == topk:
                    break
            if len(docs) < topk:
                missing = topk - len(docs)
                docs.extend([""] * missing)
                scores_out.extend([0.0] * missing)
            all_docs.append(docs)
            all_scores.append(scores_out)

        if return_scores:
            return all_docs, all_scores
        return all_docs

    def retrieve_auto_lang(self, queries, topk=1, return_scores=False, over_fetch_k=None):
        """Convenience wrapper: detect each query's language automatically."""
        from lang_utils import detect_query_language  # see lang_utils.py snippet
        langs = [detect_query_language(q) for q in queries]
        result = self.retrieve(
            queries, topk=topk, return_scores=return_scores,
            over_fetch_k=over_fetch_k, langs=langs,
        )
        return result, langs
