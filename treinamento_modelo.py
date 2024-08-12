import os

import evaluate
import pandas as pd
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, default_data_collator

from captcha_dataset import CaptchaDataset

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Definição do DataSet de Treino
train_path = "dataset/train/"
train_dir_list = os.listdir(train_path)
train_dataframe_dataset = pd.DataFrame(train_dir_list, columns=['file_name'])
train_dataframe_dataset['text'] = train_dataframe_dataset['file_name'].map(lambda x: x.split(".")[0])
print(train_dataframe_dataset)

# Definição do DataSet de Teste
test_path = "dataset/test/"
test_dir_list = os.listdir(test_path)
test_dataframe_dataset = pd.DataFrame(test_dir_list, columns=['file_name'])
test_dataframe_dataset['text'] = test_dataframe_dataset['file_name'].map(lambda x: x.split(".")[0])
print(test_dataframe_dataset)

# VALORES BASICOS E CONSTANTES
MODEL_CKPT = "microsoft/trocr-small-printed"
MODEL_NAME = MODEL_CKPT.split("/")[-1].replace("printed", "captcha")
NUM_OF_EPOCHS = 9

# USANDO GPU NVIDIA DEDICADA SE DISPONÍVEL
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.memory_summary(device=None, abbreviated=False)
else:
    device = torch.device('cpu')

# Instanciando o processor, Criando o treino e testando as instancias do dataset
processor = TrOCRProcessor.from_pretrained(MODEL_CKPT, clean_up_tokenization_spaces=True)
train_dataset = CaptchaDataset(root_dir="dataset/train/", df=train_dataframe_dataset, processor=processor)
test_dataset = CaptchaDataset(root_dir="dataset/test/", df=test_dataframe_dataset, processor=processor)

print(f"O dataset de treinamento contém {len(train_dataset)} amostras.")
print(f"O dataset de teste contém {len(test_dataset)} amostras.")

encoding = train_dataset[0]

# Show Label for Above Example
labels = encoding['labels']
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)
print(label_str)

# Instantiate Model
model = VisionEncoderDecoderModel.from_pretrained(MODEL_CKPT).to(device)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

model.config.vocab_size = model.config.decoder.vocab_size

model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

#Define Metricas de Avaliação
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    label_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


# Define Training Arguments
args = Seq2SeqTrainingArguments(
    output_dir=MODEL_NAME,
    num_train_epochs=NUM_OF_EPOCHS,
    predict_with_generate=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    logging_first_step=True,
    hub_private_repo=False,
    push_to_hub=False
)

# Define Trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor,
    args=args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=default_data_collator
)

# Fit/Train Model
trainer.train()

# Save Model & Model State
trainer.save_model()
processor.save_pretrained(f'./{MODEL_NAME}')
trainer.save_state()

# Evaluate Model
trainer.evaluate()

kwargs = {
    "finetuned_from": model.config._name_or_path,
    "tasks": "image-to-text",
    "tags": ["image-to-text"],
}

trainer.create_model_card(**kwargs)
