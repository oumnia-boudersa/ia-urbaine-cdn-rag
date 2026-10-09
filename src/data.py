from typing import Dict, List, Callable, Tuple, Union, Callable
import logging
import os
import json
import re
import glob
import string
import spacy
from collections import Counter
from tqdm import tqdm
import numpy as np
from datasets import Dataset

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class BaseDataset:
    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        return {}

    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
            
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    def format(self, fewshot: int = 0):
        def _format(
            example: Dict,
            use_answer: bool = False,
            input_template_func: Callable = None,
        ):
            q = example['question']
            if 'cot' in example:
                cot = example['cot'] if type(example['cot']) is str else ''.join(example['cot'])
            else:
                cot = None
            a = example['answer']

            query = input_template_func(q)
            if use_answer:
                query += ('' if query[-1] in {'\n', ' '} else ' ') + self.output_template(cot, a)
            return query

        # demo
        demo = [{
            'question': self.examplars[i]['question'],
            'case': _format(self.examplars[i], use_answer=True, input_template_func=self.demo_input_template),
            'ctxs': self.examplars[i]['ctxs'] if 'ctxs' in self.examplars[i] else []
        } for i in range(fewshot)] if fewshot else []

        def _format_for_dataset(example):
            # case
            case = _format(example, use_answer=False, input_template_func=self.test_input_template)
            # ctx
            example['demo'] = demo
            example['case'] = case
            return example
        self.dataset = self.dataset.map(_format_for_dataset)
    
    def get_real_prediction(self, pred):
        return pred


class IvadoCDN(BaseDataset):
    examplars: List[Dict] = [
        {
            "question": "Where can I get food assistance in NDG?",
            "cot": (
                "The Depot Community Food Center offers food assistance in NDG. "
                "Extract the address, hours, and food-assistance-specific schedule "
                "from the Context."
            ),
            "ctxs": [
                "Organization 1: The Depot Community Food Center URL:\n"
                "https://macommunaute.ca/bottin-des-agences/depotalimentairendg/ Website:\n"
                "http://www.depotmtl.org/ Description: ABOUT THE ORGANIZATION Contact details 6505 av.\n"
                "Somerled, Montreal QC H4V 1S7 5144834680 Opening hours SCHEDULE Our office is open\n"
                "Monday to Friday from 9 a.m. at 4 p.m. at 6505 ave. Somerled. Food assistance is offered on\n"
                "Tuesdays from 2 p.m. to 7 p.m. and on Fridays from 10 a.m. to 2 p.m. at 6450 ave. Somerled.\n"
                "Mission Le Dépôt center Community Food Depot, formerly NDG Food Depot, is an\n"
                "organization non-profit community organization that works in collaboration with other community\n"
                "partners community to tackle food security issues and reduce the difficulties of poverty."
            ],
            "answer": (
                "Yes. The Depot Community Food Center (formerly NDG Food Depot), located at "
                "6505 avenue Somerled, Montreal, offers food assistance. Its office is open "
                "Monday to Friday from 9 a.m. to 4 p.m. Food assistance specifically is offered "
                "on Tuesdays from 2 p.m. to 7 p.m. and on Fridays from 10 a.m. to 2 p.m., at "
                "6450 avenue Somerled. You can contact them at 514-483-4680."
            )
        },
 
        {
            "question": "Is there an organization that helps low-income households find affordable housing?",
            "cot": (
                "HAPOPEX manages affordable rental housing for low-income households. "
                "Extract address, hours, and the scale of housing they manage from the Context."
            ),
            "ctxs": [
                "Organization 9: Popular housing in Parc-Extension - HAPOPEX URL:\n"
                "https://macommunaute.ca/bottin-des-agences/habitationspopulairesdeparcextensionhapopex/\n"
                "Website: https://www.hapopex.org/ Description: ABOUT THE ORGANIZATION Contact details 445,\n"
                "rue Jean Talon ouest, Montréal QC H3N1R1 514-583-1378 Opening hours SCHEDULE Monday:\n"
                "8:30 a.m. to 12 p.m. and 1 p.m. to 4:30 p.m. Tuesday: 8:30 a.m. to 12 p.m. and 1 p.m. to 4:30 p.m.\n"
                "Wednesday: 1 p.m. to 4:30 p.m. Thursday: 8:30 a.m. to 12 p.m. and 1 p.m. to 4:30 p.m. Friday: 8:30\n"
                "a.m. to 12 p.m. and 1 p.m. to 4:30 p.m. Mission HAPOPEX (Les Habitations populaire de\n"
                "Parc-Extension) wishes to help low-income households in Montreal in order to improve their living\n"
                "conditions accommodation. Description Hapopex acquires and manages buildings with the aim of "
                "offering rental of decent and affordable housing to people with low or modest incomes. We manage "
                "a real estate portfolio of 375 housing units in 16 buildings spread across the neighborhoods "
                "North-Central Montreal."
            ],
            "answer": (
                "Yes. HAPOPEX (Les Habitations populaires de Parc-Extension), located at 445 "
                "rue Jean Talon Ouest, Montréal, helps low-income households find decent, "
                "affordable rental housing. It manages 375 housing units across 16 buildings in "
                "North-Central Montreal. Office hours are Monday, Tuesday, Thursday and Friday "
                "from 8h30 to 12h00 and 13h00 to 16h30, and Wednesday from 13h00 to 16h30. You "
                "can reach them at 514-583-1378."
            )
        },
 
        {
            "question": "Is there a service that helps elderly people stay at home in Côte-des-Neiges?",
            "cot": (
                "SARPAD Côte-des-Neiges supports elderly people who want to stay at home. "
                "Extract address, hours, and the list of services from the Context."
            ),
            "ctxs": [
                "Organization 3: SARPAD Volunteer Center Inc. - SARPAD URL:\n"
                "https://macommunaute.ca/bottin-des-agences/centredebenevolatsarpad/ Website:\n"
                "https://www.sarpad.com/ Description: ABOUT THE ORGANIZATION Contact details 6555 Chemin\n"
                "de la Côte-des-Neiges, Suite 442, Montréal QC H3S 2A6 514-737-2454 514-737-4376 Opening\n"
                "hours SCHEDULE Monday to Thursday - 9:00 a.m. to 5:00 p.m. Friday 9:00 a.m. to 4:30 p.m.\n"
                "Mission The mission of SARPAD volunteer center (Support and Respite Services for\n"
                "Elderly People in Domicile) is to support, by a team composed mainly of volunteers, the people with\n"
                "a loss of autonomy who wish to stay at home, in addition to supporting caregivers. Description Our\n"
                "services for the elderly: Transport support medical Group grocery shopping Respite Friendly visits\n"
                "Shopping Walks Phone calls Tax clinic Computer course To make a first service request, simply\n"
                "call either the Côte-des-Neiges office or the Outremont office."
            ],
            "answer": (
                "Yes. The SARPAD Volunteer Center – Côte-des-Neiges, located at 6555 Chemin de "
                "la Côte-des-Neiges, Suite 442, Montréal, supports elderly people who wish to "
                "stay at home, including services like medical transport support, grocery "
                "shopping, respite care, and friendly visits. It is open Monday to Thursday from "
                "9h00 to 17h00 and Friday from 9h00 to 16h30. You can contact them at "
                "514-737-2454."
            )
        },
 
        {
            "question": "Is there a place where young adults in Côte-des-Neiges can get free help finding a job?",
            "cot": (
                "Carrefour Jeunesse-Emploi CDN/VMR/Outremont offers free employment services "
                "for young adults. Extract address, hours, and the services offered from the "
                "Context."
            ),
            "ctxs": [
                "Organization 1: Carrefour Jeunesse-Emploi de Côte-des-Neiges/ Ville Mont-Royal/ Outremont -\n"
                "CJE-CDN / VMR / Outremont URL:\n"
                "https://macommunaute.ca/bottin-des-agences/carrefourjeunesseemploidecotedesneiges/ Website:\n"
                "http://www.cjecdn.qc.ca/ Description: ABOUT THE ORGANIZATION Contact details 6555,\n"
                "Côte-des-neiges, Montréal QC H3S 2A6 514-342-5678 514-342-6377 Opening hours SCHEDULE\n"
                "Monday to Thursday: 9 a.m. to 5 p.m. Friday: 1:30 p.m. to 5 p.m. Mission Since 1997, the Carrefour "
                "Jeunesse-Emploi offers free activities and services to young adults from Côte-des-Neiges, Ville "
                "Mont-Royal and Outremont. The CJE team of professionals aims to help in their process of "
                "integrating into employment, returning to studies or in their project of business creation. "
                "Description Employability Help in creating a skills assessment Group preparatory training for "
                "employment Individual and group support in job search procedures."
            ],
            "answer": (
                "Yes. The Carrefour Jeunesse-Emploi CDN/VMR/Outremont, located at 6555 "
                "Côte-des-Neiges, Montréal, offers free employment services for young adults, "
                "including skills assessments, job-search support, and group training. It is "
                "open Monday to Thursday from 9h00 to 17h00 and Friday from 13h30 to 17h00. "
                "You can contact them at 514-342-5678."
            )
        },
 
        {
            "question": "Is there a place where immigrants can take free French classes?",
            "cot": (
                "ÉMULCQ offers free part-time French courses for immigrants. Extract address, "
                "hours, and how to register from the Context."
            ),
            "ctxs": [
                "Organization 18: Multiethnic School of Languages and Cultures of Quebec URL:\n"
                "https://macommunaute.ca/bottin-des-agences/ecole-multiethnique-de-langues-et-de-cultu\n"
                "res-du-quebec/ Website: https://coursdefrancais.ca/ Description: ABOUT THE ORGANIZATION\n"
                "Contact details 3480 boulevard Décarie, 2nd floor, Montreal QC H4A 3J3 514 484-8899 Opening\n"
                "hours SCHEDULE Monday to Friday 9:00 a.m. to 9:00 p.m., Saturday and Sunday 2:00 p.m. to\n"
                "6:00 p.m. Mission Our mission at the Multiethnic School of languages and cultures of Quebec\n"
                "(ÉMULCQ) is to offer free, quality French courses to immigrants, thus promoting their linguistic and\n"
                "social integration into the community Quebecois. Description Services offered by ÉMULCQ\n"
                "Free, part-time French courses – Our courses are designed to allow immigrants to learn\n"
                "French at their own pace. How to register For you To register for the next session, please visit the\n"
                "website quebec.ca."
            ],
            "answer": (
                "Yes. The Multiethnic School of Languages and Cultures of Quebec (ÉMULCQ), "
                "located at 3480 boulevard Décarie (2nd floor), Montréal, offers free, "
                "part-time French courses for immigrants. It is open Monday to Friday from "
                "9h00 to 21h00, and weekends from 14h00 to 18h00. You can contact them at "
                "514-484-8899, or register for the next session at quebec.ca."
            )
        }
 
        ]

    demo_input_template = lambda self, ques: f"Question: {ques}\nAnswer:"
    test_input_template = lambda self, ques: f"Question: {ques}\nAnswer:"
    output_template = lambda self, cot, ans: f"{ans}" if ans else ""
 
    def __init__(self, data_path: str, queries_file: str = "cdn_rag_queries.json"):
        logger.info(f"Loading Ivado CDN queries from {data_path}")
        query_path = os.path.join(data_path, queries_file)
        if not os.path.exists(query_path):
            raise FileNotFoundError(f"Ivado query file not found: {query_path}")
 
        with open(query_path, "r") as fin:
            raw = json.load(fin)
 
        dataset = []
        for persona, difficulties in raw.items():
            if not isinstance(difficulties, dict):
                continue
            for difficulty, questions in difficulties.items():
                if not isinstance(questions, list):
                    continue
                for idx, question in enumerate(questions):
                    qid = f"{persona}_{difficulty}_{idx}".replace(" ", "_").lower()
                    dataset.append({
                        "qid": qid,
                        "question": question,
                        "answer": "",
                        "persona": persona,
                        "difficulty": difficulty,
                    })
 
        self.dataset = Dataset.from_list(dataset)
    