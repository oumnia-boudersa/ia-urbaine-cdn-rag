import os
import json
import argparse
import tqdm as tqdm_module
from copy import copy
import logging
from data import IvadoCDN
from generate_ivado import *

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.config_path = config_path
    if "shuffle" not in args:
        args.shuffle = False 
    if "use_counter" not in args:
        args.use_counter = True 
    return args

def main():
    args = get_args()
    logger.info(f"{args}")

    # Create a unique run directory for this invocation (0,1,2,...) under args.output_dir.
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    dir_name = os.listdir(args.output_dir)
    for i in range(10000):
        if str(i) not in dir_name:
            args.output_dir = os.path.join(args.output_dir, str(i))
            os.makedirs(args.output_dir)
            break
    logger.info(f"output dir: {args.output_dir}")
    # save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    # create output file
    output_file = open(os.path.join(args.output_dir, "output.txt"), "w")

    # Load dataset
    if args.dataset == "ivado_cdn":
        data = IvadoCDN(args.data_path, getattr(args, "queries_file", "cdn_rag_queries.json"))
    else:
        raise NotImplementedError
    
    data.format(fewshot=args.fewshot)
    data = data.dataset
    if args.shuffle:
        data = data.shuffle()
    if args.sample != -1:
        samples = min(len(data), args.sample)
        data = data.select(range(samples))
   
    # Select generation strategy according to method flag.
    if args.method == "non-retrieval":
        model = BasicRAG(args)
    elif args.method == "single-retrieval":
        model = SingleRAG(args)
    elif args.method == "fix-length-retrieval" or args.method == "fix-sentence-retrieval":
        model = FixLengthRAG(args)
    elif args.method == "token":
        model = TokenRAG(args)
    elif args.method == "entity":
        model = EntityRAG(args)
    elif args.method == "attn_prob" or args.method == "dragin":
        model = AttnWeightRAG(args)

    else:
        raise NotImplementedError


    logger.info("start inference")
    for i in tqdm_module.tqdm(range(len(data))):
        print(f"-------------------------Processing {i} / {len(data)}-------------")
        last_counter = copy(model.counter)
        batch = data[i]
        print(f"______QUESTION: {batch['question']}_________")
        print(f"-------------DEMO: {batch['demo']}--------------")
        print(f"------------CASE: {batch['case']}---------------")
        pred = model.inference(batch["question"], batch["demo"], batch["case"])
        print(f"________pred {pred}___________")
        pred = pred.strip()
        ret = {
            "qid": batch["qid"], 
            "prediction": pred,
        }

        print(f"----------{ret}--------------------")
        if args.use_counter:
            ret.update(model.counter.calc(last_counter))
    
        
        output_file.write(json.dumps(ret)+"\n")

        # Force save to disk
        output_file.flush()
        os.fsync(output_file.fileno())
        print(f"Saved: {ret}")
        print(f"model counter {model.counter.calc(last_counter)}" )
        print(f"answer saved in {output_file} in {dir_name}")
    output_file.close()
    
if __name__ == "__main__":
    main()