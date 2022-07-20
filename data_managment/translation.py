import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TranslationPipeline
from tqdm.auto import tqdm
from langdetect import detect, LangDetectException


def detect_language(document, min_len=5):
    try:
        if len(document.split()) > min_len:
            return detect(document)
        else:
            return "short"
    except LangDetectException:
        return "unk"


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


class Translator:
    def __init__(self, model="Helsinki-NLP/opus-mt-mul-en", device=0, batch_size=32):
        self.translation_tokenizer = AutoTokenizer.from_pretrained(model)
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.pipeline = TranslationPipeline(
            self.translation_model,
            self.translation_tokenizer,
            device=device,
            batch_size=batch_size
        )

    def translate_sentences(self, sentences):
        dataset = TranslationDataset(sentences)
        translation_result = []
        for out in tqdm(self.pipeline(dataset), total=len(sentences)):
            translation_result.extend([x['translation_text'] for x in out])
        return translation_result


multilanguage_translator = Translator("Helsinki-NLP/opus-mt-mul-en")
korean_translator = Translator("Helsinki-NLP/opus-mt-ko-en")
