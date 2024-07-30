import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import logging
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

model_path = '/app/model'

logger.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
logger.info("Tokenizer and model loaded successfully")


class EmbeddingRequest(BaseModel):
    text: str


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@app.post("/embed")
async def generate_embedding(request: EmbeddingRequest):
    start_time = time.time()
    logger.info(f"Received embedding request for text: {request.text[:50]}...")

    try:
        logger.debug("Tokenizing input...")
        inputs = tokenizer(f"passage: {request.text}", max_length=512,
                           padding=True, truncation=True, return_tensors='pt')
        logger.debug("Tokenization completed")

        logger.debug("Generating embeddings...")
        with torch.no_grad():
            outputs = model(**inputs)
        logger.debug("Embeddings generated")

        logger.debug("Post-processing embeddings...")
        embeddings = average_pool(
            outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        logger.debug("Post-processing completed")

        result = {"embedding": embeddings.squeeze().tolist()}

        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(
            f"Embedding generated successfully. Processing time: {processing_time:.2f} seconds")

        return result
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting embedding microservice...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
