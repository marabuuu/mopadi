#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation study: Reconstruction WITHOUT genomic features.

This script reconstructs the same tiles as `reconstruct_1k_genomic_images.py`
but uses ZERO genomic feature vectors during both encoding and decoding.

By comparing the results of this script with the genomic-conditioned version,
we can assess whether the genomic information actually helps reconstruction quality.

Expected outcome:
  - If reconstruction quality drops significantly (↓ SSIM, ↑ MSE):
    → Genomic features are informative and helpful
  - If quality is similar:
    → Features are not being used effectively (may indicate training issue)
  - If quality improves (?!)
    → Features may be confusing the model (training issue)

Usage:
    python reconstruct_ablation_no_genomics.py

Results are saved to a separate directory for easy comparison with genomic-conditioned run.
"""

import os
import sys
import json
import numpy as np
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
from mopadi.dataset import ZipTilesWithGenomicFeatures  # type: ignore[reportAttributeAccessIssue]
from torch.utils.data import DataLoader
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def sanitize_filename(patient_id):
    """Convert patient ID to safe filename by replacing problematic characters."""
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

    # SSIM
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


def main(
    encoder_path,
    data_loader,
    conf,
    save_path,
    genomic_feature_dim=512,
    num_samples=1000,
    samples_to_reconstruct=None,
    zip_results=True
):
    """
    Evaluate autoencoder WITHOUT genomic features (ablation study).
    
    All parameters same as reconstruct_1k_genomic_images.py except:
    - genomic_feature_dim: dimension of the feature vector (needed for zero-vector)
    """
    
    # Load encoder with genomic feature extractor
    encoder = ImageEncoder(
        conf, 
        autoenc_path=encoder_path, 
        feat_extractor='genomic',  # Still use genomic extractor architecture
        device="cuda:0"
    )
    
    # Initialize metrics
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
    
    logger.info(f"Loaded encoder from {encoder_path}")
    logger.info(f"⚠️  ABLATION MODE: Using ZERO genomic feature vectors (shape: [1, {genomic_feature_dim}])")
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Parse samples_to_reconstruct
    if samples_to_reconstruct is None or samples_to_reconstruct == 'all':
        filter_patients = None
    else:
        filter_patients = set([str(p).upper() for p in samples_to_reconstruct])
    
    results = []
    skipped_count = 0
    processed_count = 0
    patient_tile_counts = {}
    
    # Process samples
    pbar = tqdm(data_loader, desc="Processing samples (NO genomics)")
    for batch_idx, batch in enumerate(pbar):
        if processed_count >= num_samples:
            break
        
        if not isinstance(batch, dict):
            logger.error(f"Batch {batch_idx} is not a dict. Skipping.")
            continue
        
        if 'img' not in batch or 'feat' not in batch:
            logger.error(f"Batch {batch_idx} missing 'img' or 'feat' keys. Available: {batch.keys()}")
            continue
        
        imgs = batch['img']
        feats = batch['feat']  # We load but don't use them (ablation)
        filenames = batch.get('filename', [f"tile_{batch_idx}_{i}" for i in range(len(imgs))])
        
        for i, (img, feat) in enumerate(zip(imgs, feats)):
            if processed_count >= num_samples:
                break
            
            fname = filenames[i] if i < len(filenames) else f"tile_{processed_count}"
            try:
                if ".zip:" in str(fname):
                    zip_part = str(fname).split(".zip:")[0]
                    basename = os.path.basename(zip_part)
                    parts = basename.split('-')
                    patient_id = '-'.join(parts[:3])
                else:
                    patient_id = f"unknown_{processed_count}"
            except Exception as e:
                logger.warning(f"Could not extract patient ID from {fname}: {e}")
                patient_id = f"unknown_{processed_count}"
            
            if filter_patients is not None and patient_id.upper() not in filter_patients:
                skipped_count += 1
                continue
            
            img = img.unsqueeze(0).to(encoder.device)
            
            # === ABLATION: Use ZERO genomic features ===
            zero_feat = torch.zeros((1, genomic_feature_dim), device=encoder.device, dtype=torch.float32)
            
            with torch.no_grad():
                try:
                    # Encode with zero features
                    x_T = encoder.encode_to_noise(img, zero_feat, T=250)
                    
                    # Decode with zero features
                    reconstructed = encoder.decode_image(x_T, zero_feat, T=20)
                    
                    # Compute metrics
                    ssim, mse, ms_ssim_score = compute_structural_similarity(
                        reconstructed, img.squeeze(0), ms_ssim
                    )
                    
                    # Create patient subdirectory
                    patient_dir = os.path.join(save_path, sanitize_filename(patient_id))
                    os.makedirs(patient_dir, exist_ok=True)
                    
                    # Save reconstructed image
                    tile_label = os.path.basename(str(fname)).replace('.png', '').replace('.jpg', '')
                    safe_label = sanitize_filename(tile_label)
                    filename = f"reconstructed_ablation_{processed_count:05d}_{safe_label}.png"
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
                        'tile_path': str(fname),
                        'ablation_mode': 'NO_GENOMICS'  # Mark for easy filtering
                    }
                    results.append(result_entry)
                    
                    if patient_id not in patient_tile_counts:
                        patient_tile_counts[patient_id] = 0
                    patient_tile_counts[patient_id] += 1
                    
                    processed_count += 1
                    pbar.set_postfix({'processed': processed_count, 'skipped': skipped_count})
                    
                except Exception as e:
                    logger.error(f"Reconstruction failed for {fname}: {e}", exc_info=False)
                    skipped_count += 1
                    continue
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(save_path, 'autoencoder_evaluation_NO_GENOMICS.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"\n✅ Ablation evaluation complete!")
        logger.info(f"   Processed: {processed_count}")
        logger.info(f"   Skipped:   {skipped_count}")
        logger.info(f"   Results saved to: {csv_path}")
        logger.info(f"\n   SSIM:    {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
        logger.info(f"   MS-SSIM: {df['MS-SSIM'].mean():.4f} ± {df['MS-SSIM'].std():.4f}")
        logger.info(f"   MSE:     {df['MSE'].mean():.4f} ± {df['MSE'].std():.4f}")
        
        logger.info(f"\n   📊 Tiles reconstructed per patient:")
        for pid, count in sorted(patient_tile_counts.items()):
            logger.info(f"      {pid}: {count} tiles")
        
        # Zip results
        if zip_results and results:
            logger.info(f"\n   Zipping results by patient...")
            for patient_id in patient_tile_counts.keys():
                patient_dir = os.path.join(save_path, sanitize_filename(patient_id))
                if os.path.isdir(patient_dir):
                    zip_path = f"{patient_dir}.zip"
                    try:
                        shutil.make_archive(patient_dir, 'zip', patient_dir)
                        logger.info(f"      Zipped {patient_id} → {os.path.basename(zip_path)}")
                    except Exception as e:
                        logger.error(f"Failed to zip {patient_dir}: {e}")
        
        logger.info(f"\n💡 To compare with genomic-conditioned results:")
        logger.info(f"   conda activate Master-Thesis-Repository/.venv")
        logger.info(f"   python run_pipeline.py --config src/config.yaml --stage evaluation")
    else:
        logger.warning(f"No results to save. Processed: {processed_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    # === CONFIGURATION ===
    encoder_path = "/mnt/bulk-saturn/maralampert/genhist/experiments/20260306_mopadi_training_attempt_from_scratch/autoenc/last.ckpt"
    
    # Save to a different directory so results are easy to compare
    save_path = "/mnt/bulk-saturn/maralampert/genhist/experiments/20260306_mopadi_training_attempt_from_scratch/reconstruct_1k_no_genomics_ablation"
    
    genomic_feature_dirs = [
        "/mnt/bulk-saturn/maralampert/genhist/experiments/20260126_trying_to_get_the_whole_picture_with_existing_diffusion_model_checkpoint_but_without_properly_handling_genomics/full_train_512/mopadi_features/test"
    ]
    
    tile_zip_dirs = [
        "/mnt/bulk-saturn/maralampert/genhist/data/BRCA-tumor-tiles-all"
    ]
    
    # Load config from checkpoint
    from mopadi.configs.config import TrainConfig
    from mopadi.configs.choices import ModelName

    def load_conf_from_ckpt(ckpt_path: str) -> TrainConfig:
        state = torch.load(ckpt_path, map_location="cpu")
        if "hyper_parameters" in state:
            conf = TrainConfig()
            conf.from_dict(state["hyper_parameters"])
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
    
    # Load dataset (same as with-genomics version)
    try:
        logger.info(f"Loading tile images from: {tile_zip_dirs}")
        logger.info(f"Genomic feature directories ignored in ablation mode")
        
        test_dataset = ZipTilesWithGenomicFeatures(
            root_dirs=tile_zip_dirs,
            feature_dirs=genomic_feature_dirs,
            process_only_zips=True,
            skip_zip_validation=True,
            do_normalize=conf.do_normalize,
            do_resize=conf.do_resize,
            img_size=conf.img_size,
        )
        
        logger.info(f"Test dataset contains {len(test_dataset)} tiles")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            num_workers=2,
            shuffle=True,
        )
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    logger.info("\n" + "="*70)
    logger.info("ABLATION STUDY: Reconstruction WITHOUT Genomic Features")
    logger.info("="*70 + "\n")
    
    # Use same patients as main run for fair comparison
    samples_to_reconstruct = ['TCGA-5L-AAT0', 'TCGA-5T-A9QA', 'TCGA-A1-A0SG', 'TCGA-A1-A0SK', 'TCGA-A1-A0SM']
    
    main(
        encoder_path=encoder_path,
        data_loader=test_loader,
        conf=conf,
        save_path=save_path,
        genomic_feature_dim=512,  # Match your training config
        num_samples=1000,
        samples_to_reconstruct=samples_to_reconstruct,
        zip_results=True,
    )
    
    logger.info("\n" + "="*70)
    logger.info("Ablation complete! Compare the two CSVs to assess genomic impact.")
    logger.info("="*70)
