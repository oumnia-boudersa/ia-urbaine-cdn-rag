import os
import re
import tqdm
import json
import spacy
import torch
import random
import hashlib
import logging
import numpy as np
import pytrec_eval
from math import exp
from peft import PeftModel
from scipy.special import softmax
from collections import defaultdict
from complexity_judge import ComplexityJudge
from lang_utils import detect_query_language
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from retrieve.retriever_ivado import BM25, SGPT, FaissRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig









logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

# spaCy NLP pipeline for sentence/entity/POS tagging
nlp = spacy.load("en_core_web_sm")

############################################
def stable_seed(text, base_seed):
    """Process-independent 32-bit seed from (base_seed, text)."""
    digest = hashlib.sha256(f"{base_seed}|{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


class SeededGenerationMixin:
    generation_seed = None      # None -> disabled -> original behaviour

    def _reseed(self, prompt_text):
        base = getattr(self, "generation_seed", None)
        if base is None:
            return None
        seed = stable_seed(prompt_text, base)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed
############################################
class BasicGenerator(SeededGenerationMixin):
    def __init__(self, model_name_or_path):
        # Load tokenizer/model once per run; trust_remote_code only for Falcon models.
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                    trust_remote_code = "falcon" in model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", 
                    trust_remote_code = "falcon" in model_name_or_path)
        self.model.generation_config.use_cache = False
        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_text, max_length, return_logprobs=False):
        #self.embed_device = self.model.get_input_embeddings().weight.device

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        input_ids = input_ids.to(self.model.device)
        #input_ids = input_ids.to(self.embed_device)

        print("")
        print("self.model.device", self.model.device)
        input_length = input_ids.shape[1]

        #attention_mask = torch.ones_like(input_ids).to(self.embed_device)
        attention_mask = torch.ones_like(input_ids)
        self._reseed(input_text) 
        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                output_scores = True,
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs
        
        else:
            outputs = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                attention_mask = attention_mask,
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            return text, None, None
    
    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)
        self._reseed(input_text)
        outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_length, 
            return_dict_in_generate = True, 
            output_scores = True,     
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        # attention
        atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0] #last layer first batch
        if solver == "max": 
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        # mean_atten = mean_atten[tl:tr]
            
        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq) 
        else:
            seqentropies = None 

        return text, seqlist, attns, seqlogprobs, seqentropies
    




class Counter:
    """Tracks metrics for inference: retrieval count, generation count, hallucination count, tokens, sentences."""
    
    def __init__(self):
        """Initialize counters to zero."""
        self.retrieve = 0      # Number of retrieval calls
        self.generate = 0      # Number of generation calls
        self.hallucinated = 0  # Number of hallucinations detected and corrected
        self.token = 0         # Total tokens generated
        self.sentence = 0      # Total sentences generated

    def add_generate(self, text, tokenizer):
        """Update counters after text generation.
        
        Args:
            text: Generated text.
            tokenizer: Tokenizer to count tokens and extract sentences.
        """
        self.generate += 1
        # Count tokens via tokenizer
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        # Count sentences via spaCy
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        """Compute delta (difference) between this counter and another.
        
        Useful for logging per-example metrics without double-counting across examples.
        
        Args:
            other_counter: Counter to subtract from this one.
            
        Returns:
            Dictionary of metric deltas.
        """
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
         

class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        self.generator.generation_seed = getattr(self, "generation_seed", None)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer, 
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, 
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path, 
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )

            ###########################################################
            elif self.retriever_type == "SPARSE":
                print(" ##################### using SPARSE retriever type ##################:", self.retriever_type)
                from retrieve.retriever_ivado import SparseRetrieverIvado
                self.retriever = SparseRetrieverIvado(
                    sparse_index_path = self.sparse_index_path,
                    text_key = getattr(self, "sparse_text_key", "content"),
                    id_key = getattr(self, "sparse_id_key", "chunk_id"),
                )
            ###########################################################3
            
            ## for ivado data
            elif self.retriever_type == "FAISS":
                self.retriever = FaissRetriever(
                    index_path = self.faiss_index_path,
                    pkl_path = self.faiss_pkl_path,
                    text_key = getattr(self, "faiss_text_key", "content"),
                    id_key = getattr(self, "faiss_id_key", "chunk_id"),
                    embedding_model = getattr(
                        self,
                        "faiss_embedding_model",
                        "paraphrase-multilingual-MiniLM-L12-v2",
                    ),
                    normalize_embeddings = getattr(self, "faiss_normalize", True),
                    id_map_path = getattr(self, "faiss_id_map_path", None),
                )

                print("\n========== FAISS DEBUG ==========")
                print("index_path:", self.faiss_index_path)
                print("pkl_path:", self.faiss_pkl_path)
                print("text_key:", getattr(self, "faiss_text_key", "content"))
                print("id_key:", getattr(self, "faiss_id_key", "chunk_id"))

                print("Retriever:", self.retriever)
                print("Retriever attributes:", self.retriever.__dict__.keys())

                for key, value in self.retriever.__dict__.items():
                    print(f"\nATTRIBUTE: {key}")
                    print("TYPE:", type(value))

                    if isinstance(value, (list, tuple)):
                        print("LENGTH:", len(value))
                        print("FIRST:", value[:2])

                    elif isinstance(value, dict):
                        print("LENGTH:", len(value))
                        print("FIRST:", list(value.items())[:2])

                print("========== END FAISS DEBUG ==========\n")

            
            ## end
            else:
                raise NotImplementedError
        
        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
      #
        # fo ivado data
                  queries = [query], 
                topk = topk,
            )
            return docs[0] # les documents associés à la première requête, le nombre de documents dépend de topk
        # fo ivado data
        # elif self.retriever_type == "FAISS":
        #     docs = self.retriever.retrieve(
        #         queries = [query],
        #         topk = topk,
        #     )
        #     return docs[0]
        elif self.retriever_type in {"FAISS", "SPARSE"}:
            print("====================================================================")
            print("====================== query ================================", query)
            print("====================================================================")
            lang = detect_query_language(query)
            self.last_detected_lang = lang  # stash for use in inference()/prompt building
            print("====================================================================")
            print("================== language detected =========================", lang)
            docs = self.retriever.retrieve(
                queries = [query],
                topk = topk,
                langs = [lang],
            )
            return docs[0]
    
        # end
        else:
            raise NotImplementedError
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else "" # retourne la 1ère phrase.

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" # retourne la dernière phrase.
    
    def inference(self, question, demo, case):
        # non-retrieval
        #La query formulation décrit comment on transforme la question de l’utilisateur avant de l’envoyer au modèle ou au moteur de recherche. C’est une étape en amont de l’inférence. direct veut dire la question est utilisée comme elle est, sans aucune modification.
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        print("================================================ Answering with no retrieval ====================================================")
        print("================================================ RAW GENERATED TEXT ====================================================")
        print(text)
        print("================================================ END RAW GENERATED TEXT ====================================================")

        print("text before cleaning:", text)
        logger.info(f"Generated text before cleaning: {text}")
        # Remove special tokens
        text = text.replace("<|eot_id|>", "")
        text = text.replace("<|end_of_text|>", "")

        text = re.split(
            r"Question\s*:",
            text,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0]

        # Clean whitespace
        text = text.strip()
        print("text after cleaning:", text)
        logger.info(f"Generated text: {text}")



        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text
    

class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case):
        print("====================== SingleRAG class =======================")
        logger.info("SingleRAG inference started")
        self.retrieved_docs = []
        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)  # extraire les top n documents avan la génération. 
        self.retrieved_docs.append({"query": question, "docs": docs})
        print(f"Retrieved docs: {docs}")
        print(f"Retrieved docs length: {len(docs)}")
        print("self.retrieved_docs", self.retrieved_docs)

        # 对 topk 个 passage 生成 prompt
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += "Context:\n"

        for i, doc in enumerate(docs):
            prompt += f"[{i+1}] {doc}\n"

        prompt += (
            "Answer the question above with ONLY the answer. "
            "Do not generate another question, another Answer section, "
            "or any additional examples.\n"
        )

        #prompt += "Answer in the same format as before.\n" # force le LLM à respecter le format des demos
        prompt += case
        print(f"Final prompt: {prompt}")

        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                # STOP when the model starts generating the next example
        # Stop at the first occurrence of "Question:"
        text = re.split(
            r"Question\s*:",
            text,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0]

        # Remove special tokens
        text = text.replace("<|eot_id|>", "")
        text = text.replace("<|end_of_text|>", "")

        # match = re.search(
        #     #r"\s*Question\s*:",
        #     r"Question\s*:",
        #     text,
        #     flags=re.IGNORECASE
        # )
        # if match:
        #     text = text[:match.start()]
        
        # stop_patterns = [
        #     r"\nQuestion:",
        #     r"\nQuestion\s*:",
        #     r"\n###",
        # ]
        # for pattern in stop_patterns:
        #     match = re.search(pattern, text)
        #     if match:
        #         text = text[:match.start()]
        #         break

        # Clean whitespace
        text = text.strip()

        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        text = ""
        retrieve_question = question
        while True:
            old_len = len(text)
            docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i+1}] {doc}\n"
            prompt += "Answer in t he same format as before.\n"
            prompt += case + " " + text
            if self.method == "fix-length-retrieval":
                new_text, _, _ = self.generator.generate(prompt, self.fix_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                text = text.strip() + " " + new_text.strip()
                retrieve_question = new_text.strip()
            else:
                # fix sentence
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                new_text = new_text.strip()
                sentences = list(nlp(new_text).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                text = text.strip() + " " + str(sentences[0])
                retrieve_question = sentences[0]
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                print("Hallucination detected:", sentences[sid])
                # keep sentences before hallucination 

                prev = "" if sid == 0 else " ".join(sentences[:sid])
                print("Previous sentences:", prev)
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                print("Current sentence before modification:", curr)
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1
        
        # No hallucination
        print("No hallucination detected.")
        return text, None, False
    
    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case + " " + text
            print("_________________prompt________________________")
            print(prompt)
            new_text, tokens, logprobs = self.generator.generate(
                prompt, 
                self.generate_max_length, 
                return_logprobs=True
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                if self.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text
    

class EntityRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)
        
        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)
        
        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        return super().inference(question, demo, case)


class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, attentions, weight):
        # Entrées: texte généré par le LLM, tokens du texte, scores d'attention du modèle, importance du token (logprob ou entropie)
        
        # découper la génération en phrases. l'hallucination est détectée phrase par phrase.
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0 # index global des tokens ¸
        # sid index de la phrase.
        # [tl:tr] correspondant à la phrase 
        for sid, sent in enumerate(sentences):
            # le code cherche quels tokens correspondent à chaque phrase.
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
            # value = attenion * (-log prob)
            print("________All attentions___________",attentions)

            # Extraire les attentions de cette phrase
            attns = attentions[tl:tr]
            print("___________sentence attns:___________", attns)

            # Normalisation pour que la somme = 1
            attns = np.array(attns) / sum(attns)
            print("___________normalized attns:_________", attns)

            # Calculer un score d’hallucination par token
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)] # value = attn * weight * longueur_phrase

            thres = [1 if v > self.hallucination_threshold else 0 for v in value] # 1 → token halluciné, 0 → token OK

            # Si une phrase contient des tokens halluciné, On s’arrête immédiatement et on renvoie : return True, prev_text: text sûr, tokens_de_la_phrase: tokens fautifs, mask_hallucination: 1 qui indiqe faux token 
            if 1 in thres:
                # hallucinated
                if "check_real_words" in self.__dict__ and self.check_real_words: # Vérifie si l’objet (self) a un attribut check_real_words et s’il est activé (True).
                    
                    # Utilise spaCy (nlp) pour analyser la phrase grammaticalement, Ne garde que les mots “importants” : noms, verbes, adjectifs, noms propres, nombres
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in 
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    # On parcourt tous les tokens de la phrase, Si le token ne contient aucun vrai mot, on le marque comme non-halluciné (thres[i]=0) Si le token contient un vrai mot, on garde sa valeur initiale (1 si hallucination détectée)
                    # seuls les tokens hallucinés et contenant des vrais mots restent marqués.
                    for i in range(len(thres)):
                        if not match(tokens[tl+i]):
                            thres[i] = 0                
                
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # curr = " ".join(
                #     [tokens[i] if thres[i] == 0 else "[xxx]" for i in range(len(thres))]
                # )
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None
        

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        # EXTRAIRE DES MOTS POUR RETRIEVE
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_ids = input_ids.to(self.generator.model.device)
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]
        
        #Les modèles comme BERT/LLM peuvent diviser un mot en plusieurs tokens (subword). Ici, le code recompose les tokens pour former les mots complets.
        # ex: tokens_tmp = ["run", "##ning", "fast"], tokens = ["running", "fast"]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
               
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        # 获取幻觉词对应的 attention
        
        curr_st = len(tokens) - len(curr_tokens)
        atten_tmp = torch.mean(atten_tmp, dim=0)
        attns = []
        for r in range_:
            # att = torch.zeros(atten_tmp.shape[0], input_length)
            att = torch.zeros(input_length)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i-1][:r[0]] # 上一位的
                v = v / v.sum()
                t = torch.zeros(input_length)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            # merge token for att
            att = torch.tensor([att[rr[0]:rr[1]+1].sum() for rr in range_]) 
            #att = torch.tensor([att[rr[0]:rr[1]+1].sum().detach().cpu().item() for rr in range_]) # i added detach
            attns.append(att)
            
        # 计算每个超过阈值的 token 在前文的 attentions
        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 分析词性，保留实词对应的 attns
        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in 
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        
        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False
        
        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if i >= curr_st and curr_hit[i - curr_st]:
                continue
            if match(tok):
                real_pairs.append((att, tok, i))
        
        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)
        
        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])


    
        
    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        # print(question)
        # print("#" * 20)
        # Evaluate query complexity at the beginning and store the result
        self.last_complexity_result = getattr(self, 'evaluate_query_complexity', lambda q: {})(question)
        
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            # print('####', prompt)
            # prompt += case + " " + text
            # changed att to attn middle for dragin wit middle layer
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt, 
                self.generate_max_length, 
                # self.attention_solver, 
                use_entropy = self.method == "dragin", 
                use_logprob = self.method == "attn_prob"
            )
            weight = entropies if self.method == "dragin" else [-v for v in logprobs]

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)
            
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                forward_all = [question, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                def fetch_last_n_tokens(text, num, tokenizer = self.generator.tokenizer):
                    tokens = tokenizer.tokenize(text)
                    if num >= len(tokens):
                        return text
                    last_n_tokens = tokens[-num:]
                    last_n_sentence = ' '.join(last_n_tokens)
                    return last_n_sentence

                if self.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)

                elif self.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(
                        list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                    )

                elif self.query_formulation == "forward_all":
                    retrieve_question = forward_all
                
                elif self.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)
                
                elif self.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.__dict__
                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)
                
                elif self.query_formulation == "real_words": 
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext, 
                        curr_tokens = curr_tokens, 
                        curr_hit = curr_hit,
                    ) 
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                # print('#####', prompt)
                # prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text)
                tmp_li = [text.strip(), ptext.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
                # text = text.strip() + " " + ptext.strip() + " " + new_text.strip()

                # print("### retrieve_question ###")
                # print(retrieve_question)
                # context = "### Context: ###\n"
                # for i, doc in enumerate(docs):
                #     context += f"[{i+1}] {doc}\n" 
                # print(context)
                # print(text)
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        # print("#" * 20)
        return text
