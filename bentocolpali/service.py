import asyncio
import logging
from typing import Annotated, List, cast

import bentoml
import numpy as np
import torch
from annotated_types import MinLen
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from PIL import Image

from .interfaces import ImagePayload
from .utils import convert_b64_to_pil_image, is_url

logger = logging.getLogger("bentoml")


def create_model_pipeline(path: str) -> ColPali:
    return cast(
        ColPali,
        ColPali.from_pretrained(
            pretrained_model_name_or_path=path,
            torch_dtype=torch.bfloat16,
            device_map=get_torch_device("auto"),
            local_files_only=True,
            low_cpu_mem_usage=True,
        ),
    ).eval()


def create_processor_pipeline(path: str) -> ColPaliProcessor:
    return cast(
        ColPaliProcessor,
        ColPaliProcessor.from_pretrained(
            pretrained_model_name_or_path=path,
            local_files_only=True,
        ),
    )


@bentoml.service(
    name="colpali_batch",
    workers=1,
    traffic={"concurrency": 64},
)
class ColPaliBatchService:
    """
    ColPali service (batch intermediate service) for embedding images and queries, and scoring them.
    Provides batch processing capabilities.

    NOTE: You need to build the model using `bentoml.models.create(name="colpali_model")` before using this service.
    """

    _model_ref: bentoml.Model = bentoml.models.get("colpali_model")

    def __init__(self) -> None:
        self.model: ColPali = create_model_pipeline(path=self._model_ref.path)
        self.processor: ColPaliProcessor = create_processor_pipeline(path=self._model_ref.path)
        logger.info(f"ColPali loaded on device: {self.model.device}")

    @bentoml.api(
        batchable=True,
        batch_dim=(0, 0),
        max_batch_size=64,
        max_latency_ms=30_000,
    )
    async def embed_images(
        self,
        items: List[ImagePayload],
    ) -> np.ndarray:
        """
        Generate image embeddings of shape (batch_size, sequence_length, embedding_dim).
        """

        images: List[Image.Image] = []

        for item in items:
            if is_url(item.url):
                raise NotImplementedError("URLs are not supported.")
            images += [convert_b64_to_pil_image(item.url)]

        batch_images = self.processor.process_images(images).to(self.model.device)

        with torch.inference_mode():
            image_embeddings = self.model(**batch_images)

        return image_embeddings.cpu().to(torch.float32).detach().numpy()

    @bentoml.api(
        batchable=True,
        batch_dim=(0, 0),
        max_batch_size=64,
        max_latency_ms=30_000,
    )
    async def embed_queries(
        self,
        items: List[str],
    ) -> np.ndarray:
        """
        Generate query embeddings of shape (batch_size, sequence_length, embedding_dim).
        """
        batch_queries = self.processor.process_queries(items).to(self.model.device)

        with torch.inference_mode():
            query_embeddings = self.model(**batch_queries)

        return query_embeddings.cpu().to(torch.float32).detach().numpy()

    @bentoml.api
    async def score_embeddings(
        self,
        image_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Returns the late-interaction/MaxSim scores of shape (num_queries, num_images).

        Args:
            image_embeddings: The image embeddings of shape (num_images, sequence_length, embedding_dim).
            query_embeddings: The query embeddings of shape (num_queries, sequence_length, embedding_dim).
        """

        image_embeddings_torch: List[torch.Tensor] = [torch.Tensor(x) for x in image_embeddings]
        query_embeddings_torch: List[torch.Tensor] = [torch.Tensor(x) for x in query_embeddings]

        return (
            self.processor.score(
                qs=query_embeddings_torch,
                ps=image_embeddings_torch,
            )
            .cpu()
            .to(torch.float32)
            .numpy()
        )

    @bentoml.api
    async def score(
        self,
        images: Annotated[List[ImagePayload], MinLen(1)],
        queries: Annotated[List[str], MinLen(1)],
    ) -> np.ndarray:
        """
        Returns the late-interaction/MaxSim scores of the queries against the images.
        """

        image_embeddings, query_embeddings = await asyncio.gather(
            self.embed_images(images),
            self.embed_queries(queries),
        )

        return await self.score_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
        )


@bentoml.service(name="colpali")
class ColPaliService:
    """
    ColPali service for embedding images and queries, and scoring them.
    Provides batch processing capabilities.

    NOTE: You need to build the model using `bentoml.models.create(name="colpali_model")` before using this service.
    """

    _colpali_batch = bentoml.depends(ColPaliBatchService)

    @bentoml.api
    async def embed_images(self, images: List[ImagePayload]) -> np.ndarray:
        """
        Generate image embeddings of shape (batch_size, sequence_length, embedding_dim).
        """
        return await self._colpali_batch.to_async.embed_images(images)

    @bentoml.api
    async def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Generate query embeddings of shape (batch_size, sequence_length, embedding_dim).
        """
        return await self._colpali_batch.to_async.embed_queries(queries)

    @bentoml.api
    async def score_embeddings(
        self,
        image_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Returns the late-interaction/MaxSim scores of shape (num_queries, num_images).
        """
        return await self._colpali_batch.to_async.score_embeddings(image_embeddings, query_embeddings)

    @bentoml.api
    async def score(
        self,
        images: Annotated[List[ImagePayload], MinLen(1)],
        queries: Annotated[List[str], MinLen(1)],
    ) -> np.ndarray:
        """
        Returns the late-interaction/MaxSim scores of the queries against the images.
        """

        image_embeddings, query_embeddings = await asyncio.gather(
            self._colpali_batch.to_async.embed_images(images),
            self._colpali_batch.to_async.embed_queries(queries),
        )

        return await self._colpali_batch.to_async.score_embeddings(image_embeddings, query_embeddings)
