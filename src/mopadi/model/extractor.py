import torch
from typing import Union
from transformers import AutoModel
from torchvision import transforms

from dotenv import load_dotenv
import timm
from timm.models._factory import create_model
from timm.layers.mlp import SwiGLUPacked
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform  # type: ignore[import-not-found]
import os

from mopadi.configs.choices import *  # type: ignore[reportWildcardImportFromLibrary]

# ignore the warning coming from conch
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers",
    category=FutureWarning,
)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


class FeatureExtractorConch:
    def __init__(self, device: Union[str, torch.device] = 'cpu'):
        from conch.open_clip_custom import create_model_from_pretrained

        model_result = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
        self.model = model_result[0] if isinstance(model_result, tuple) else model_result
        self.model.eval().to(device)
        self.device = device

        for p in self.model.parameters():
            p.requires_grad = False

        transform_list = [transforms.Resize(size=448, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                          transforms.CenterCrop(size=(448,448)),
                          transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ]
        self.transform = transforms.Compose(transform_list)

        print(f"CONCH model successfully initialised on device {self.device}...\n")

    def extract_feats(self, imgs_batch: torch.Tensor, need_grad: bool = True):
        """
        Extracts features from WSI tiles using CONCH tile encoder.
        """
        assert imgs_batch.min() >= -1e-3 and imgs_batch.max() <= 1+1e-3, "Expect [0,1] input"
        batch = torch.stack([self.transform(x) for x in imgs_batch]).to(self.device)

        if need_grad:
            features = self.model.encode_image(batch, proj_contrast=False, normalize=False)   # TODO: try with float32, but need to reextract feats
        else:
            with torch.inference_mode():
                features = self.model.encode_image(batch, proj_contrast=False, normalize=False)   # TODO: try with float32, but need to reextract feats
        return features


class FeatureExtractorConch15:
    def __init__(self, device: Union[str, torch.device] = 'cpu'):

        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        self.model, self.eval_transform = titan.return_conch()
        print(self.eval_transform)
        self.model.eval().to(device)
        self.device = device

        transform_list = [transforms.Resize(size=448, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
        self.transform = transforms.Compose(transform_list)

        for p in self.model.parameters():
            p.requires_grad = False

        print(f"CONCH model successfully initialised on device {self.device}...\n")

    def extract_feats(self, imgs_batch: torch.Tensor,  need_grad: bool = True):
        """
        Extracts features from WSI tiles using CONCH tile encoder.
        """
        assert imgs_batch.min() >= -1e-3 and imgs_batch.max() <= 1+1e-3, "Expect [0,1] input"
        batch = torch.stack([self.transform(x) for x in imgs_batch]).to(self.device)

        if need_grad:
            features = self.model(batch)
        else:
            with torch.inference_mode():
                features = self.model(batch)
        return features

    
class FeatureExtractorVirchow2:
    def __init__(self, device: Union[str, torch.device] = "cpu"):

        self.model = create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        ).eval().to(device)
        self.device = device

        for p in self.model.parameters():
            p.requires_grad = False

        print(f"Virchow2 model successfully initialised on device {self.device}...\n")

        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))  # type: ignore[assignment]  # type: ignore[assignment]

    def extract_feats(self, imgs_batch: torch.Tensor, need_grad: bool = True):
        """
        Extract features from input images using Virchow tile encoder.

        Args:
            images (torch.Tensor): A batch of images, shape (B, C, H, W).

        Returns:
            torch.Tensor: Embeddings for the batch, shape (B, 2560).
        """
        assert imgs_batch.min() >= -1e-3 and imgs_batch.max() <= 1+1e-3, "Expect [0,1] input"
        batch = torch.stack([self.transform(x) for x in imgs_batch]).to(self.device)  # type: ignore[arg-type]

        if need_grad:
            output = self.model(batch) # shape: (B, 261, 1280)
        else:
            with torch.inference_mode():
                output = self.model(batch)

        class_token = output[:, 0]  # shape: (B, 1280)
        #patch_tokens = output[:, 5:]  # Ignore tokens 1-4 (register tokens)
        #features = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)  # shape: (B, 2560)
        return class_token.half()

    
class FeatureExtractorGenomic:
    """
    Pseudo-extractor for pre-computed genomic feature vectors.

    Unlike image-based extractors (CONCH, Virchow2, UNI2), this class does not
    encode images.  Genomic features are loaded from pre-computed .h5 files by
    the dataset and passed directly as conditioning features to the diffusion
    model.

    This class exists to maintain interface compatibility with the rest of the
    pipeline and to store metadata about the genomic feature space.
    """

    def __init__(self, feat_dim: int, device: Union[str, torch.device] = 'cpu'):
        self.feat_dim = feat_dim
        self.device = device
        # No neural-network model — nothing to put on a device.
        self.model = None
        print(f"Genomic feature extractor initialized (feat_dim={feat_dim}, device={device}).")
        print("Note: Genomic features are pre-computed; no image encoder is used.\n")

    # ------------------------------------------------------------------
    # Interface helpers
    # ------------------------------------------------------------------
    @property
    def supports_image_extraction(self) -> bool:
        """Genomic features cannot be extracted from images."""
        return False

    def extract_feats(self, imgs_batch: torch.Tensor, need_grad: bool = True) -> torch.Tensor:
        """
        Not supported for genomic features.

        Genomic features are patient-level and cannot be derived from tile
        images.  They must be pre-computed and provided via the dataset.
        """
        raise NotImplementedError(
            "FeatureExtractorGenomic: genomic features cannot be extracted "
            "from images.  They must be pre-computed and loaded via the "
            "dataset (e.g. WDSTilesWithGenomicFeatures)."
        )


class FeatureExtractorUNI2:
    def __init__(self, device: Union[str, torch.device] = "cpu"):

        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }

        self.model = create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)

        transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        print(transform) # just for doublechecking
        self.model.eval().to(device)
        self.device = device

        for p in self.model.parameters():
            p.requires_grad = False

        print(f"UNI2 model successfully initialised on device {device}...\n")

        transform_list = [transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                          transforms.Normalize(mean=(0.4850, 0.4560, 0.4060), std=(0.2290, 0.2240, 0.2250))
        ]
        self.transform = transforms.Compose(transform_list)

    def extract_feats(self, imgs_batch, need_grad: bool = True):
        assert imgs_batch.min() >= -1e-3 and imgs_batch.max() <= 1+1e-3, "Expect [0,1] input"
        batch = torch.stack([self.transform(x) for x in imgs_batch]).to(self.device)
    
        if need_grad:
            features = self.model(batch)
        else:
            with torch.inference_mode():
                features = self.model(batch)
        return features


