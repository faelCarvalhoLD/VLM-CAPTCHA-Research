import os

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load image from the IAM database (actually this model is meant to be used on printed text)
# url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = Image.open(
    "dataset/sample/2b827.png").convert("RGB")

path = 'trocr-base-printed-captcha-ocr'
processor = TrOCRProcessor.from_pretrained(path, clean_up_tokenization_spaces=True)
model = VisionEncoderDecoderModel.from_pretrained(path)
model.to(device)
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values.to(device))
solution_captcha = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].replace(' ', '')
print('solution_captcha: ' + solution_captcha)
