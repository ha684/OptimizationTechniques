from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages

bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")


from style_bert_vits2.tts_model import TTSModel

assets_root = Path("model_assets")

model = TTSModel(
    model_path=assets_root / model_file,
    config_path=assets_root / config_file,
    style_vec_path=assets_root / style_file,
    device="cuda",
)

model.infer

model.device