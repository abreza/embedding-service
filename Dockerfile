FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git wget

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model
RUN wget -O /app/model/config.json https://huggingface.co/intfloat/multilingual-e5-large/resolve/main/config.json
RUN wget -O /app/model/pytorch_model.bin https://huggingface.co/intfloat/multilingual-e5-large/resolve/main/pytorch_model.bin
RUN wget -O /app/model/tokenizer_config.json https://huggingface.co/intfloat/multilingual-e5-large/resolve/main/tokenizer_config.json
RUN wget -O /app/model/tokenizer.json https://huggingface.co/intfloat/multilingual-e5-large/resolve/main/tokenizer.json
RUN wget -O /app/model/special_tokens_map.json https://huggingface.co/intfloat/multilingual-e5-large/resolve/main/special_tokens_map.json
RUN wget -O /app/model/sentencepiece.bpe.model https://huggingface.co/intfloat/multilingual-e5-large/resolve/main/sentencepiece.bpe.model

COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
