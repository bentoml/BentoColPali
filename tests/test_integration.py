import subprocess
from typing import List

import bentoml
import numpy as np
import pytest

from bentocolpali.service import ImagePayload
from tests.conftest import COLPALI_EMBEDDING_DIM

BENTOML_PORT = "50001"
LOCAL_BENTOML_URL = f"http://localhost:{BENTOML_PORT}"


class TestColPaliServiceIntegration:
    """
    Integration tests for the ColPali.

    To run these tests, first you need to build the BentoService:

    ```bash
    bentoml build -f bentofile.yaml
    ```
    """

    @pytest.mark.slow
    def test_embed_images_route(self, image_payloads: List[ImagePayload]):
        with subprocess.Popen(
            ["bentoml", "serve", "bentocolpali.service:ColPaliService", "-p", BENTOML_PORT]
        ) as server_proc:
            try:
                with bentoml.SyncHTTPClient(LOCAL_BENTOML_URL) as client:
                    embeddings = client.embed_images(image_payloads)

                assert embeddings.ndim == 3
                assert embeddings.shape[0] == len(image_payloads)
                assert embeddings.shape[2] == COLPALI_EMBEDDING_DIM

            finally:
                server_proc.terminate()

    @pytest.mark.slow
    def test_embed_queries_route(self, queries: List[str]):
        with subprocess.Popen(
            ["bentoml", "serve", "bentocolpali.service:ColPaliService", "-p", BENTOML_PORT]
        ) as server_proc:
            try:
                with bentoml.SyncHTTPClient(LOCAL_BENTOML_URL) as client:
                    embeddings = client.embed_queries(queries)

                assert embeddings.ndim == 3
                assert embeddings.shape[0] == len(queries)
                assert embeddings.shape[2] == COLPALI_EMBEDDING_DIM

            finally:
                server_proc.terminate()

    @pytest.mark.slow
    def test_score_embeddings_route(
        self,
        random_image_embeddings: np.ndarray,
        random_query_embeddings: np.ndarray,
    ):
        with subprocess.Popen(
            ["bentoml", "serve", "bentocolpali.service:ColPaliService", "-p", BENTOML_PORT]
        ) as server_proc:
            try:
                with bentoml.SyncHTTPClient(LOCAL_BENTOML_URL) as client:
                    scores = client.score_embeddings(
                        image_embeddings=random_image_embeddings,
                        query_embeddings=random_query_embeddings,
                    )

                assert scores.ndim == 2
                assert scores.shape[0] == random_query_embeddings.shape[0]
                assert scores.shape[1] == random_image_embeddings.shape[0]

            finally:
                server_proc.terminate()

    @pytest.mark.slow
    def test_score_route(
        self,
        image_payloads: List[ImagePayload],
        queries: List[str],
    ):
        with subprocess.Popen(
            ["bentoml", "serve", "bentocolpali.service:ColPaliService", "-p", BENTOML_PORT]
        ) as server_proc:
            try:
                with bentoml.SyncHTTPClient(LOCAL_BENTOML_URL) as client:
                    scores = client.score(images=image_payloads, queries=queries)

                assert scores.ndim == 2
                assert scores.shape[0] == len(queries)
                assert scores.shape[1] == len(image_payloads)

                # Check if the maximum scores per row are in the diagonal of the matrix score
                assert (scores.argmax(axis=1) == range(len(queries))).all()

            finally:
                server_proc.terminate()
