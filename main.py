import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
import logging
import time
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_PATH = '/app/model'
logger.info("Loading tokenizer and ONNX model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    ort_session = ort.InferenceSession("/app/model/model.onnx")
    logger.info("Tokenizer and ONNX model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise


class EmbeddingRequest(BaseModel):
    text: str


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states * attention_mask[..., None]
    return last_hidden.sum(axis=1) / attention_mask.sum(axis=1)[..., None]


@app.post("/embed")
async def generate_embedding(request: EmbeddingRequest):
    start_time = time.time()
    logger.info(f"Received embedding request for text: {request.text[:50]}...")
    try:
        logger.debug("Tokenizing input...")
        inputs = tokenizer(f"passage: {request.text}", max_length=512,
                           padding=True, truncation=True, return_tensors='np')
        logger.debug("Tokenization completed")

        logger.debug("Generating embeddings...")
        ort_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        last_hidden_state = ort_outputs[0]
        logger.debug("Embeddings generated")

        logger.debug("Post-processing embeddings...")
        embeddings = average_pool(last_hidden_state, inputs['attention_mask'])
        embeddings = embeddings / \
            np.linalg.norm(embeddings, axis=1, keepdims=True)
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


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting embedding microservice...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
