import grpc
from concurrent import futures
import time
import logging
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

import embedding_service_pb2
import embedding_service_pb2_grpc

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = '/app/model'


class EmbeddingService(embedding_service_pb2_grpc.EmbeddingServiceServicer):
    def __init__(self):
        logger.info("Loading tokenizer and ONNX model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.ort_session = ort.InferenceSession("/app/model/model.onnx")
        logger.info("Tokenizer and ONNX model loaded successfully")

    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states * attention_mask[..., None]
        return last_hidden.sum(axis=1) / attention_mask.sum(axis=1)[..., None]

    def GenerateEmbedding(self, request, context):
        start_time = time.time()
        logger.info(
            f"Received embedding request for text: {request.text[:50]}...")
        try:
            inputs = self.tokenizer(f"passage: {request.text}", max_length=512,
                                    padding=True, truncation=True, return_tensors='np')

            ort_inputs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
            ort_outputs = self.ort_session.run(None, ort_inputs)
            last_hidden_state = ort_outputs[0]

            embeddings = self.average_pool(
                last_hidden_state, inputs['attention_mask'])
            embeddings = embeddings / \
                np.linalg.norm(embeddings, axis=1, keepdims=True)

            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(
                f"Embedding generated successfully. Processing time: {processing_time:.2f} seconds")

            return embedding_service_pb2.EmbeddingResponse(embedding=embeddings.squeeze().tolist())
        except Exception as e:
            logger.error(
                f"Error generating embedding: {str(e)}", exc_info=True)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return embedding_service_pb2.EmbeddingResponse()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    embedding_service_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("gRPC server started on port 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
