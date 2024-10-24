"""
Test the `ColPaliService` BentoML service.
"""

from typing import List
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch

from bentocolpali.service import ColPaliService, ImagePayload
from tests.conftest import COLPALI_EMBEDDING_DIM


@pytest.mark.asyncio
async def test_embed_images(image_payloads: List[ImagePayload]):
    service = ColPaliService()

    mock_output = torch.rand(len(image_payloads), 32, COLPALI_EMBEDDING_DIM, dtype=torch.float32, device="cpu")

    with patch.object(service.model, "forward", new=Mock(return_value=mock_output)) as mock_service_method:
        image_embeddings = await service.embed_images(image_payloads)
        mock_service_method.assert_called_once()

    assert isinstance(image_embeddings, np.ndarray)
    assert image_embeddings.ndim == 3
    assert image_embeddings.shape == mock_output.shape


@pytest.mark.asyncio
async def test_embed_queries(queries: List[str]):
    service = ColPaliService()

    mock_output = torch.rand(len(queries), 32, COLPALI_EMBEDDING_DIM, dtype=torch.float32, device="cpu")

    with patch.object(service.model, "forward", new=Mock(return_value=mock_output)) as mock_service_method:
        query_embeddings = await service.embed_queries(queries)
        mock_service_method.assert_called_once()

    assert isinstance(query_embeddings, np.ndarray)
    assert query_embeddings.ndim == 3
    assert query_embeddings.shape == mock_output.shape


@pytest.mark.asyncio
async def test_score_embeddings(
    random_image_embeddings: np.ndarray,
    random_query_embeddings: np.ndarray,
):
    service = ColPaliService()

    scores = await service.score_embeddings(
        image_embeddings=random_image_embeddings,
        query_embeddings=random_query_embeddings,
    )

    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 2
    assert scores.shape == (len(random_query_embeddings), len(random_image_embeddings))


@pytest.mark.asyncio
async def test_score(
    image_payloads: List[ImagePayload],
    queries: List[str],
):
    service = ColPaliService()

    mock_embed_images = AsyncMock(return_value=np.random.rand(len(image_payloads), 32, COLPALI_EMBEDDING_DIM))
    mock_embed_queries = AsyncMock(return_value=np.random.rand(len(queries), 16, COLPALI_EMBEDDING_DIM))

    with patch.object(service, "embed_images", new=mock_embed_images):
        with patch.object(service, "embed_queries", new=mock_embed_queries):
            scores = await service.score(images=image_payloads, queries=queries)

    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 2
