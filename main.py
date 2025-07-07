import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import uvicorn

import embedding_service_pb2 as pb2
import embedding_service_pb2_grpc as pb2_grpc

MODEL_PATH = "/app/model"
GRPC_PORT = 50051
HTTP_PORT = 8000


class EmbeddingBackend:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.session = ort.InferenceSession(f"{MODEL_PATH}/model.onnx")

    def _average_pool(self, hidden, mask):
        hidden = hidden * mask[..., None]
        return hidden.sum(1) / mask.sum(1)[..., None]

    def generate(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            f"passage: {text}",
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        outputs = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )[0]
        emb = self._average_pool(outputs, inputs["attention_mask"])
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.squeeze()


class EmbeddingGRPCServicer(pb2_grpc.EmbeddingServiceServicer):
    def __init__(self, backend: EmbeddingBackend):
        self.backend = backend

    def GenerateEmbedding(self, request, context):
        start = time.time()
        try:
            vec = self.backend.generate(request.text).tolist()
            return pb2.EmbeddingResponse(embedding=vec)
        finally:
            duration = time.time() - start
            print(f"[gRPC] processed in {duration:.2f}s")


def start_grpc(backend: EmbeddingBackend):
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingGRPCServicer(backend), server
    )
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    print(f"gRPC server ready on :{GRPC_PORT}")
    return server


class HTTPEmbeddingRequest(BaseModel):
    text: str


def create_http_app(backend: EmbeddingBackend) -> FastAPI:
    app = FastAPI(title="Embedding Service")

    @app.post("/embed")
    async def embed(req: HTTPEmbeddingRequest):
        vec = backend.generate(req.text).tolist()
        return {"embedding": vec}

    return app


def start_http(app: FastAPI):
    thread = threading.Thread(
        target=uvicorn.run,
        kwargs=dict(app=app, host="0.0.0.0", port=HTTP_PORT, log_level="info"),
        daemon=True,
    )
    thread.start()
    print(f"HTTP server ready on :{HTTP_PORT}")


if __name__ == "__main__":
    backend = EmbeddingBackend()
    grpc_server = start_grpc(backend)
    start_http(create_http_app(backend))

    grpc_server.wait_for_termination()
