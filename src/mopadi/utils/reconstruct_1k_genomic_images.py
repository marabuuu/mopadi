# reconstruct_1k_genomic_images.py
import os
import sys
import json
import numpy as np
import h5py
import torch
import cv2
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity, mean_squared_error
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchvision import transforms
from pathlib import Path
from collections import OrderedDict
import logging
import shutil

from mopadi.configs.templates import *  # type: ignore
from mopadi.utils.encode import ImageEncoder
from mopadi.dataset import WDSTilesWithGenomicFeatures, ZipTilesWithGenomicFeatures
from torch.utils.data import DataLoader
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


class GenomicFeatureCache:
    """Simple cache for genomic .h5 files (same as in dataset.py)."""
    def __init__(self, max_open=32, feat_key="feats"):
        self.max_open = max_open
        self.feat_key = feat_key
        self.cache = {}

    def get(self, h5_path):
        if h5_path in self.cache:
            return self.cache[h5_path]
        
        with h5py.File(h5_path, "r") as f:
            # h5py returns a dataset-like object; indexing is untyped so pylance complains.
            # using [()] to read entire array and ignoring type issues keeps lint happy.
            feat = np.asarray(f[self.feat_key][()])  # type: ignore
            feat = feat.squeeze()  # shape: (D,)
        
        self.cache[h5_path] = feat
        
        # Clean old entries if cache exceeds limit
        if len(self.cache) > self.max_open:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        return feat


def find_genomic_h5(patient_id, genomic_feature_dirs):
    """
    Find the .h5 file for a patient in genomic feature directories.
    
    patient_id: e.g., "TCGA-A1-A0SE-DX1" (str or tensor)
    genomic_feature_dirs: list of directories containing .h5 files
    """
    # Convert to string if necessary (might be a tensor from dataloader)
    if isinstance(patient_id, torch.Tensor):
        patient_id = patient_id.item()
    patient_id = str(patient_id).strip()
    
    if isinstance(genomic_feature_dirs, str):
        genomic_feature_dirs = [genomic_feature_dirs]
    
    for feature_dir in genomic_feature_dirs:
        if not os.path.isdir(feature_dir):
            continue
        h5_path = os.path.join(feature_dir, f"{patient_id}.h5")
        if os.path.exists(h5_path):
            return h5_path
    
    return None


def sanitize_filename(patient_id):
    """
    Convert patient ID to safe filename by replacing problematic characters.
    """
    if isinstance(patient_id, torch.Tensor):
        patient_id = patient_id.item()
    patient_id = str(patient_id).strip()
    return patient_id.replace('/', '_').replace('\\', '_').replace(' ', '_')


def compute_structural_similarity(reconstructed_image, image_original, ms_ssim):
    """Compute SSIM, MS-SSIM, and MSE metrics."""
    # Transform from [-1, 1] to [0, 1]
    img_ori = np.array((image_original.cpu().detach().numpy() + 1) / 2)
    flipped = np.swapaxes(img_ori, 0, 2)
    
    manip_img = np.array(reconstructed_image[0].permute(2, 1, 0).cpu())
    before_gray = cv2.cvtColor(flipped.astype(np.float32), cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(manip_img.astype(np.float32), cv2.COLOR_BGR2GRAY)

    # SSIM (return tuple (score, diff) when ``full=True``)
    # Pylance sometimes warns about a possible 3-tuple, so we extract first element
    ssim_ret = structural_similarity(
        before_gray, after_gray,
        full=True,
        data_range=before_gray.max() - before_gray.min()
    )
    if isinstance(ssim_ret, tuple):
        score = ssim_ret[0]
    else:
        score = ssim_ret  # type: ignore

    # MS-SSIM
    adjusted_image = (image_original.unsqueeze(0).cpu() + 1) / 2 
    ms_ssim_res = ms_ssim(reconstructed_image.cpu(), adjusted_image)

    # MSE
    mse = mean_squared_error(before_gray, after_gray)
    
    return score, mse, ms_ssim_res.numpy()


def main(encoder_path, data_loader, conf, save_path, num_samples=1000, samples_to_reconstruct=None, zip_results=True):
    """
    Evaluate autoencoder by reconstructing images conditioned on genomic features.
    
    The data_loader must yield batches with at least 'img' and 'feat' keys.
    ZipTilesWithGenomicFeatures already pairs each tile with its patient's
    genomic feature vector, so no manual .h5 lookup is needed here.
    
    Parameters
    ----------
    encoder_path : str
        Path to the trained autoencoder checkpoint
    data_loader : DataLoader
        DataLoader providing {img, feat, filename, ...} samples
    conf : TrainConfig
        Configuration object
    save_path : str
        Directory to save results
    num_samples : int
        Maximum number of samples to evaluate
    samples_to_reconstruct : list or None
        List of patient IDs to reconstruct (e.g., ['TCGA-AR-A2LK', 'TCGA-OL-A66I']).
        If None, reconstruct all samples. Use 'all' as string to reconstruct all.
    zip_results : bool
        If True, zip each patient's reconstructed tiles folder at the end (saves space).
    """
    
    # Load encoder with genomic feature extractor
    encoder = ImageEncoder(
        conf, 
        autoenc_path=encoder_path, 
        feat_extractor='genomic',  # CRITICAL: specify genomic features
        device="cuda:0"
    )
    
    # Initialize metrics
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
    
    logger.info(f"Loaded encoder from {encoder_path}")
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Parse samples_to_reconstruct
    if samples_to_reconstruct is None or samples_to_reconstruct == 'all':
        filter_patients = None  # reconstruct all
    else:
        filter_patients = set([str(p).upper() for p in samples_to_reconstruct])
    
    results = []
    skipped_count = 0
    processed_count = 0
    patient_tile_counts = {}  # track tiles per patient
    
    # Process samples
    pbar = tqdm(data_loader, desc="Processing samples")
    for batch_idx, batch in enumerate(pbar):
        if processed_count >= num_samples:
            break
        
        # Handle batch data
        if not isinstance(batch, dict):
            logger.error(f"Batch {batch_idx} is not a dict. Skipping.")
            continue
        
        if 'img' not in batch or 'feat' not in batch:
            logger.error(f"Batch {batch_idx} missing 'img' or 'feat' keys. Available: {batch.keys()}")
            continue
        
        imgs = batch['img']           # [B, C, H, W]
        feats = batch['feat']         # [B, D]  — genomic features already matched per patient
        filenames = batch.get('filename', [f"tile_{batch_idx}_{i}" for i in range(len(imgs))])
        
        for i, (img, feat) in enumerate(zip(imgs, feats)):
            if processed_count >= num_samples:
                break
            
            # Extract a label for this tile from its filename
            fname = filenames[i] if i < len(filenames) else f"tile_{processed_count}"
            # Extract patient ID from zip path like ".../TCGA-XX-XXXX-DX1.uuid.zip:tile.png"
            # Patient key format: "TCGA-XX-XXXX" or with full DX info
            try:
                # Try to extract patient ID from the zip filename
                if ".zip:" in str(fname):
                    zip_part = str(fname).split(".zip:")[0]
                    basename = os.path.basename(zip_part)
                    # Extract base TCGA ID (first 3 hyphens): TCGA-XX-XXXX
                    parts = basename.split('-')
                    patient_id = '-'.join(parts[:3])  # e.g., "TCGA-AR-A2LK"
                else:
                    # Fallback: just use the sample_id as patient_id
                    patient_id = f"unknown_{processed_count}"
            except Exception as e:
                logger.warning(f"Could not extract patient ID from {fname}: {e}")
                patient_id = f"unknown_{processed_count}"
            
            # Check if we should process this patient
            if filter_patients is not None and patient_id.upper() not in filter_patients:
                skipped_count += 1
                continue
            
            img = img.unsqueeze(0).to(encoder.device)        # [1, C, H, W]
            genomic_feat = feat.unsqueeze(0).to(encoder.device)  # [1, D]
            
            with torch.no_grad():
                try:
                    # Encode image to latent noise using genomic features
                    x_T = encoder.encode_to_noise(img, genomic_feat, T=250)
                    
                    # Decode back to image using same genomic features
                    reconstructed = encoder.decode_image(x_T, genomic_feat, T=20)
                    
                    # Compute metrics
                    ssim, mse, ms_ssim_score = compute_structural_similarity(
                        reconstructed, img.squeeze(0), ms_ssim
                    )
                    
                    # Create patient subdirectory
                    patient_dir = os.path.join(save_path, sanitize_filename(patient_id))
                    os.makedirs(patient_dir, exist_ok=True)
                    
                    # Save reconstructed image in patient-specific directory
                    tile_label = os.path.basename(str(fname)).replace('.png', '').replace('.jpg', '')
                    safe_label = sanitize_filename(tile_label)
                    filename = f"reconstructed_{processed_count:05d}_{safe_label}.png"
                    encoder.save_image(reconstructed.squeeze(0), filename, patient_dir)
                    
                    # Record results
                    result_entry = {
                        'sample_id': processed_count,
                        'patient_id': patient_id,
                        'tile': tile_label,
                        'SSIM': float(ssim),
                        'MS-SSIM': float(ms_ssim_score),
                        'MSE': float(mse),
                        'filename': filename,
                        'tile_path': str(fname)
                    }
                    results.append(result_entry)
                    
                    # Track tiles per patient
                    if patient_id not in patient_tile_counts:
                        patient_tile_counts[patient_id] = 0
                    patient_tile_counts[patient_id] += 1
                    
                    processed_count += 1
                    pbar.set_postfix({'processed': processed_count, 'skipped': skipped_count})
                    
                except Exception as e:
                    logger.error(f"Reconstruction failed for {fname}: {e}", exc_info=False)
                    skipped_count += 1
                    continue
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(save_path, 'autoencoder_evaluation_genomic.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"\n✅ Evaluation complete!")
        logger.info(f"   Processed: {processed_count}")
        logger.info(f"   Skipped:   {skipped_count}")
        logger.info(f"   Results saved to: {csv_path}")
        logger.info(f"\n   SSIM:    {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
        logger.info(f"   MS-SSIM: {df['MS-SSIM'].mean():.4f} ± {df['MS-SSIM'].std():.4f}")
        logger.info(f"   MSE:     {df['MSE'].mean():.4f} ± {df['MSE'].std():.4f}")
        
        # Save per-patient summary
        if patient_tile_counts:
            logger.info(f"\n   Tiles reconstructed per patient:")
            for pid, count in sorted(patient_tile_counts.items()):
                logger.info(f"      {pid}: {count} tiles")
        
        # Optional: zip results by patient
        if zip_results and results:
            logger.info(f"\n   Zipping results by patient...")
            for patient_id in patient_tile_counts.keys():
                patient_dir = os.path.join(save_path, sanitize_filename(patient_id))
                if os.path.isdir(patient_dir):
                    zip_path = f"{patient_dir}.zip"
                    try:
                        shutil.make_archive(patient_dir, 'zip', patient_dir)
                        logger.info(f"      Zipped {patient_id} → {os.path.basename(zip_path)}")
                        # Optional: remove directory after zipping
                        # shutil.rmtree(patient_dir)
                    except Exception as e:
                        logger.error(f"Failed to zip {patient_dir}: {e}")
    else:
        logger.warning(f"No results to save. Processed: {processed_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    # === CONFIGURATION ===
    # Update these paths for YOUR specific setup
    
    # Path to trained autoencoder checkpoint
    # Example: /mnt/bulk-saturn/maralampert/genhist/experiments/20260306_mopadi_training_attempt_from_scratch/autoenc/last.ckpt
    encoder_path = "/mnt/bulk-saturn/maralampert/genhist/experiments/20260306_mopadi_training_attempt_from_scratch/autoenc/last.ckpt"
    
    # Path to save reconstructed images and results
    save_path = "/mnt/bulk-saturn/maralampert/genhist/experiments/20260306_mopadi_training_attempt_from_scratch/reconstruct_1k_genomic_images"
    
    # Directories containing genomic .h5 files (one .h5 per patient)
    # These should contain files like: TCGA-A1-A0SE-DX1.h5, TCGA-A1-A0SF-DX1.h5, etc.
    genomic_feature_dirs = [
        "/mnt/bulk-saturn/maralampert/genhist/experiments/20260126_trying_to_get_the_whole_picture_with_existing_diffusion_model_checkpoint_but_without_properly_handling_genomics/full_train_512/mopadi_features/test"
    ]
    
    # Configuration object matching your training setup.
    # We normally need this to know image size, normalization and model structure.
    # Instead of hard‑coding a factory function, we try to reconstruct it from the
    # checkpoint hparams that were saved during training.  This avoids the
    # Pylance undefined-variable warning and ensures the config matches the run.
    from mopadi.configs.config import TrainConfig
    from mopadi.configs.choices import ModelName

    def load_conf_from_ckpt(ckpt_path: str) -> TrainConfig:
        state = torch.load(ckpt_path, map_location="cpu")
        if "hyper_parameters" in state:
            conf = TrainConfig()
            conf.from_dict(state["hyper_parameters"])
            # Set model_name if not present (required for diffusion autoencoder)
            if conf.model_name is None:
                conf.model_name = ModelName.beatgans_autoenc
            return conf
        else:
            raise ValueError("checkpoint contains no hyper_parameters to reconstruct config")

    try:
        conf = load_conf_from_ckpt(encoder_path)
        logger.info("Loaded configuration from checkpoint hyperparameters")
    except Exception as e:
        logger.warning(f"Could not load config from checkpoint ({e}); falling back to pancancer_autoenc()")
        try:
            conf = pancancer_autoenc()
        except Exception as ee:
            logger.error(f"Fallback config factory failed: {ee}")
            sys.exit(1)
    
    # === LOAD TEST DATA ===
    # For reconstruction evaluation we NEED the original images, because:
    #   1. Encode: original image (x₀) + genomic features (z)  →  latent noise (xT)
    #   2. Decode: latent noise (xT) + genomic features (z)    →  reconstructed image (x̂₀)
    #   3. Compare: x₀  vs  x̂₀   using SSIM / MS-SSIM / MSE
    # Without originals there is nothing to compare the reconstruction against.
    #
    # Since you have zip files with PNGs (not tar shards), we use
    # ZipTilesWithGenomicFeatures which pairs each tile with its patient's
    # genomic .h5 file automatically.
    
    # Directory (or list) containing your zip files with tile images
    # e.g. each zip is named TCGA-XX-XXXX-DX1.zip and contains .png tiles
    tile_zip_dirs = [
        "/mnt/bulk-saturn/maralampert/genhist/data/BRCA-tumor-tiles-all"
    ]
    
    try:
        logger.info(f"Loading tile images from: {tile_zip_dirs}")
        logger.info(f"Loading genomic features from: {genomic_feature_dirs}")
        
        test_dataset = ZipTilesWithGenomicFeatures(
            root_dirs=tile_zip_dirs,
            feature_dirs=genomic_feature_dirs,
            process_only_zips=True,   # CRITICAL: without this, zip files are skipped!
            skip_zip_validation=True, # Skip slow CRC check on every zip file
            do_normalize=conf.do_normalize,
            do_resize=conf.do_resize,
            img_size=conf.img_size,
        )
        
        logger.info(f"Test dataset contains {len(test_dataset)} tiles with genomic features")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            num_workers=2,
            shuffle=True,   # shuffle so we sample diverse patients
        )
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Check your shard paths and configuration.")
        sys.exit(1)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    
    # === OPTIONAL: Specify which samples to reconstruct ===
    # Set to None or 'all' to reconstruct all samples
    # Or provide a list of patient IDs to reconstruct only those:
    samples_to_reconstruct = ['TCGA-5L-AAT0', 'TCGA-5T-A9QA', 'TCGA-A1-A0SG', 'TCGA-A1-A0SK', 'TCGA-A1-A0SM']  # Change to e.g. ['TCGA-AR-A2LK', 'TCGA-OL-A66I'] to reconstruct only specific patients
    
    main(
        encoder_path=encoder_path,
        data_loader=test_loader,
        conf=conf,
        save_path=save_path,
        num_samples=1000,  # Adjust based on your test set size
        samples_to_reconstruct=samples_to_reconstruct,
        zip_results=True,  # Set to False if you don't want to zip results
    )