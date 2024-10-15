import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from bentocolpali.service import ImagePayload

PAYLOAD_TEST_PATH = "./tests/data/query_test.json"
assert Path(PAYLOAD_TEST_PATH).exists(), f"File not found: {PAYLOAD_TEST_PATH}"

COLPALI_EMBEDDING_DIM = 128


def load_queries_and_images(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


@pytest.fixture(scope="module")
def queries() -> List[str]:
    data = load_queries_and_images(PAYLOAD_TEST_PATH)
    return data.get("queries", [])


@pytest.fixture(scope="module")
def images() -> List[Dict[str, Any]]:
    data = load_queries_and_images(PAYLOAD_TEST_PATH)
    return data.get("images", [])


@pytest.fixture(scope="module")
def image_payloads(images: List[Dict[str, Any]]) -> List[ImagePayload]:
    return [ImagePayload(**elt) for elt in images]


@pytest.fixture(scope="module")
def random_image_embeddings() -> np.ndarray:
    return np.random.rand(5, 32, COLPALI_EMBEDDING_DIM)


@pytest.fixture(scope="module")
def random_query_embeddings() -> np.ndarray:
    return np.random.rand(3, 16, COLPALI_EMBEDDING_DIM)
