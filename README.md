# TrOCR Solver Captcha

## Descrição do modelo
O modelo transformer utilizado é o resultado de uma serie de refinamentos do modelo [TrOCR (Transformer-based Optical 
Character Recognition with Pre-trained Models)](https://arxiv.org/abs/2109.10282). 
Inicialmente a Microsoft refinou o modelo, produzindo modelos trocr-*-printed como [trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed). Para esse refinamento, ela utilizou o dataset [SROIE](https://rrc.cvc.uab.es/?ch=13), treinados com imagens baseadas em
recibos digitalizados.

## Descrição do projeto
Tomando como base o [projeto](https://huggingface.co/DunnBC22/trocr-base-printed_captcha_ocr) de Brian Dunn [DunnBC22] e
usando um [dataset de catpchas respondidos](https://www.kaggle.com/datasets/alizahidraja/captcha-data), o modelo final pode ser treinado, usando como base trocr-base-printed

No refinamento feito por Dunn, foram usados 856 captchas respondidos e 214 capctchas usados para testar e levatar métricas.

## Instruções de Execução:

Inicialmente execute o arquivo treinamento_modelo.py para a construção do modelo, onde será salvo na pasta trocr-base-printed-captcha-ocr

Após a conclusão do treino, execute o execucao_modelo_pretreinado.py, para o uso do modelo em questão

## Referências:

RADFORD, Alec et al. Learning Transferable Visual Models From Natural Language Supervision. arXiv preprint arXiv:2109.10282, 2021. Disponível em: https://arxiv.org/abs/2109.10282. Acesso em: 7 de Agosto de 2024.
