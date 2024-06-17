from pathlib import Path

from transformers import BertForSequenceClassification, AutoTokenizer
from random import randint
import torch
from torch import cuda

MODEL_PATH = Path('models/')


class TSA_Pipeline:
    _max_len = 512

    def __init__(self, model_name: str):
        model_path = Path(MODEL_PATH, Path(model_name))
        self._device = 'cuda' if cuda.is_available() else 'cpu'
        self._tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased', local_files_only=True)
        model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased',
                                                              num_labels=5, local_files_only=True)
        model.load_state_dict(torch.load(model_path))
        model.to(self._device)
        model.eval()
        self._model = model

    def _make_prompt(self, text: str, name: str) -> list:
        prompt = ['[CLS]']
        prompt.extend(self._tokenizer.tokenize(name))
        prompt.append('[SEP]')
        prompt.extend(self._tokenizer.tokenize(text))
        return prompt

    def _align_sequence(self, tokenized_text: list) -> list:
        if (len(tokenized_text) > self._max_len):
            tokenized_text = tokenized_text[:self._max_len]
        else:
            tokenized_text = tokenized_text + ['[PAD]' for _ in range(self._max_len - len(tokenized_text))]

        return tokenized_text

    def get_sentiment(self, text: str, company_name: str) -> int:
        tokenized_text = self._make_prompt(text.lower(), company_name)
        al_tokenized_text = self._align_sequence(tokenized_text)

        attn_mask = [1 if tok != '[PAD]' else 0 for tok in al_tokenized_text]
        attn_mask = torch.tensor(attn_mask, dtype=torch.long).to(self._device)
        ids = self._tokenizer.convert_tokens_to_ids(al_tokenized_text)
        ids = torch.tensor(ids, dtype=torch.long).to(self._device)

        ids = ids[None, :]
        attn_mask = attn_mask[None, :]

        outputs = self._model(input_ids=ids, attention_mask=attn_mask)
        loss, tr_logits = outputs.loss, outputs.logits
        prediction = tr_logits.argmax(dim=1).detach().cpu().numpy()[0] + 1

        return prediction
