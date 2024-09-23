from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

from lmms_eval.tasks.mmsearch.retrieve_content.tokenization.tokenizers import LexicalAnalyzer
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    chunk_length: int = 200
    slidew: bool = False
    sentb: bool = False
    TopK: int = 8


class Content_Retriever:
    def __init__(self):
        # define tokenizer
        self.tokenizer = LexicalAnalyzer()
        self.tokenizer_offsets = LexicalAnalyzer(do_char_positions=True)

        self.config = Config()

        self.tokenizer_offsets.settings['do_sliding_window_passages'] = self.config.slidew
        self.tokenizer_offsets.settings['respect_sent_boundaries'] = self.config.sentb
        # define retrieval model
        self.model = BGEM3FlagModel(
            'BAAI/bge-m3',  
            use_fp16=True
        ) # Setting use_fp16 to True speeds up computation with a slight performance degradation

    def split_doc_into_passages(self, doc):
        text = doc
        passages = []

        passages_tokens = self.tokenizer.analyze_excerpts(text)
        for _, passage_tokens in enumerate(passages_tokens):
            if self.tokenizer.settings['respect_sent_boundaries']:

                tokens = []
                for psg in passage_tokens:
                    tokens.extend(psg)
                passage_tokens = tokens
            if len(passage_tokens) == 0:
                continue

            passage_text = " ".join(passage_tokens)
            passages.append(passage_text)
        
        return passages

    
    def get_retrieved_content(self, requery, content):
        docs = [content]
        all_chucks = self.split_doc_into_passages(content)
        # encode
        output_1 = self.model.encode([requery], return_dense=True, return_sparse=True, return_colbert_vecs=True, batch_size=12, max_length=self.config.chunk_length)
        output_2 = self.model.encode(all_chucks, return_dense=True, return_sparse=True, return_colbert_vecs=True, batch_size=12, max_length=self.config.chunk_length)
        scores = []
        for i in range(len(output_2['colbert_vecs'])):
            scores.append(
                self.model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][i]).item()
            )

        sorted_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        sorted_values, original_indices = zip(*sorted_pairs)
        return '\n'.join([all_chucks[idx] for idx in sorted_values[:self.config.TopK]])

