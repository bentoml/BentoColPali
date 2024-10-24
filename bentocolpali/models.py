"""
Use this script to build the models for the ColPali BentoService.
"""

import argparse
import os
from typing import Optional, cast

import bentoml
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from peft import LoraConfig, PeftModel


def build_models(model_name: str, hf_token: Optional[str] = None) -> None:
    """
    Build the model and the preprocessor for the ColPali BentoService.
    """
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token is None:
            raise ValueError("HF token is required.")

    with bentoml.models.create(name="colpali_model") as model_ref:
        lora_config = LoraConfig.from_pretrained(model_name)

        cast(
            PeftModel,
            PeftModel.from_pretrained(
                ColPali.from_pretrained(
                    lora_config.base_model_name_or_path,
                    torch_dtype=torch.bfloat16,
                    device_map=get_torch_device("auto"),
                    token=hf_token,
                ),
                model_name,
            ),
        ).merge_and_unload().save_pretrained(model_ref.path)

        print("Model successfully built.")

        cast(
            ColPaliProcessor,
            ColPaliProcessor.from_pretrained(
                pretrained_model_name_or_path=model_name,
                token=hf_token,
            ),
        ).save_pretrained(model_ref.path)

        print("Preprocessor successfully built.")

    return


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the models for the ColPali BentoService.")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name or path of the model to build.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        required=False,
        help="Hugging Face token for model access. Defaults to the HF_TOKEN environment variable.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    build_models(model_name=args.model_name, hf_token=args.hf_token)
    print("ColPali models were successfully built.")


if __name__ == "__main__":
    main()
