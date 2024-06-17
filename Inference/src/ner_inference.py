from pathlib import Path

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from torch import cuda

MODEL_PATH = Path('models/')


class NER_Pipeline:
    _max_len = 512

    def __init__(self, model_name: str):
        model_path = Path(MODEL_PATH, Path(model_name))
        self._device = 'cuda' if cuda.is_available() else 'cpu'
        self._tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/xlm-roberta-large-en-ru')
        model = AutoModelForTokenClassification.from_pretrained('DeepPavlov/xlm-roberta-large-en-ru',
                                                                num_labels=2)
        model.load_state_dict(torch.load(model_path))
        model.to(self._device)
        model.eval()
        self._model = model

    def _align_sequence(self, tokenized_text: list) -> list:
        if (len(tokenized_text) > self._max_len):
            tokenized_text = tokenized_text[:self._max_len]
        else:
            tokenized_text = tokenized_text + ['[PAD]' for _ in range(self._max_len - len(tokenized_text))]

        return tokenized_text

    def get_companies(self, text: str) -> list:
        tokenized_text = self._tokenizer.tokenize(text)
        al_tokenized_text = self._align_sequence(tokenized_text)

        attn_mask = [1 if tok != '[PAD]' else 0 for tok in al_tokenized_text]
        attn_mask = torch.tensor(attn_mask, dtype=torch.long).to(self._device)
        ids = self._tokenizer.convert_tokens_to_ids(al_tokenized_text)
        ids = torch.tensor(ids, dtype=torch.long).to(self._device)

        ids = ids[None, :]
        attn_mask = attn_mask[None, :]

        outputs = self._model(input_ids=ids, attention_mask=attn_mask)
        loss, tr_logits = outputs.loss, outputs.logits
        active_logits = tr_logits.view(-1, self._model.num_labels)
        print('active_logits', active_logits)
        flattened_predictions = torch.argmax(active_logits, axis=1)

        active_accuracy = attn_mask.view(-1) == 1
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        predictions = predictions.cpu().detach().numpy()

        print('predictions', predictions)

        names = []
        name = []
        for i in range(predictions.shape[0]):
            if predictions[i] == 1:
                name.append(tokenized_text[i])
            if predictions[i] == 0 and name:
                names.append(self._tokenizer.decode(self._tokenizer.convert_tokens_to_ids(name)))
                name = []
        if name:
            names.append(self._tokenizer.decode(self._tokenizer.convert_tokens_to_ids(name)))

        return list(set(names))
