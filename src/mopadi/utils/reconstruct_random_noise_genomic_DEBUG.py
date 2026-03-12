#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEBUG VERSION: Reconstruct from random noise with detailed shape logging.
Processes only 5 samples to quickly identify issues.
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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


def sanitize_filename(patient_id):
    """Convert patient ID to safe filename by replacing problematic characters."""
    if isinstance(patient_id, torch.Tensor):
        patient_id = patient_id.item()
    patient_id = str(patient_id).strip()
    return patient_id.replace('/', '_').replace('\\', '_').replace(' ', '_')


def compute_structural_similarity_debug(reconstructed_image, image_original, ms_ssim):
    """Compute SSIM, MS-SSIM, and MSE metrics with shape debugging."""
    
    logger.info(f"  [SHAPES] reconstructed_image: {reconstructed_image.shape}")
    logger.info(f"  [SHAPES] image_original input: {image_original.shape}")
    
    # Transform from [-1, 1] to [0, 1]
    img_ori = np.array((image_original.cpu().detach().numpy() + 1) / 2)
    logger.info(f"  [SHAPES] img_ori after numpy: {img_ori.shape}")
    
    flipped = np.swapaxes(img_ori, 0, 2)
    logger.info(f"  [SHAPES] flipped (swapped axes): {flipped.shape}")
    
    # Get the first element if batch dimension exists
    if len(reconstructed_image.shape) == 4:  # [B, C, H, W]
        manip_img = np.array(reconstructed_image[0].permute(2, 1, 0).cpu())
    else:  # [C, H, W]
        manip_img = np.array(reconstructed_image.permute(2, 1, 0).cpu())
    
    logger.info(f"  [SHAPES] manip_img: {manip_img.shape}")
    
    before_gray = cv2.cvtColor(flipped.astype(np.float32), cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(manip_img.astype(np.float32), cv2.COLOR_BGR2GRAY)
    
    logger.info(f"  [SHAPES] before_gray: {before_gray.shape}")
    logger.info(f"  [SHAPES] after_gray: {after_gray.shape}")
    
    if before_gray.shape != after_gray.shape:
        logger.error(f"  [ERROR] Shape mismatch! {before_gray.shape} vs {after_gray.shape}")
        raise ValueError(f"Image shapes don't match: {before_gray.shape} vs {after_gray.shape}")

    # SSIM
    ssim_ret = structural_similarity(
        before_gray, after_gray,
        full=True,
        data_range=before_gray.max() - before_gray.min()
    )
    if isinstance(ssim_ret, tuple):
        score = ssim_ret[0]
    else:
        score = ssim_ret

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
    img_shape=(3, 256, 256),
    genomic_feature_dim=512,
    num_samples=5,  # DEBUG: Only 5 samples
    samples_to_reconstruct=None,
    decode_steps=50,
    zip_results=False
):
    """
    DEBUG VERSION: Generate images from RANDOM NOISE with shape logging.
    """
    
    logger.info("\n" + "="*70)
    logger.info("DEBUG MODE: Random Noise Generation with Shape Logging")
    logger.info("="*70)
    logger.info(f"Expected img_shape: {img_shape}")
    logger.info(f"Num samples to process: {num_samples}")
    logger.info(f"Decode steps (T): {decode_steps}")
    logger.info("="*70 + "\n")
    
    # Load encoder with genomic feature extractor
    encoder = ImageEncoder(
        conf, 
        autoenc_path=encoder_path, 
        feat_extractor='genomic',
        device="cuda:0"
    )
    
    # Initialize metrics
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
    
    logger.info(f"✅ Loaded encoder from {encoder_path}")
    
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
    pbar = tqdm(data_loader, desc="Processing samples (DEBUG)")
    for batch_idx, batch in enumerate(pbar):
        if processed_count >= num_samples:
            logger.info(f"\n✅ Reached {num_samples} samples. Stopping.")
            break
        
        if not isinstance(batch, dict):
            logger.error(f"Batch {batch_idx} is not a dict. Skipping.")
            continue
        
        if 'img' not in batch or 'feat' not in batch:
            logger.error(f"Batch {batch_idx} missing 'img' or 'feat' keys. Available: {batch.keys()}")
            continue
        
        imgs = batch['img']
        feats = batch['feat']
        filenames = batch.get('filename', [f"tile_{batch_idx}_{i}" for i in range(len(imgs))])
        
        for i, (img, feat) in enumerate(zip(imgs, feats)):
            if processed_count >= num_samples:
                break
            
            fname = filenames[i] if i < len(filenames) else f"tile_{processed_count}"
            
            logger.info(f"\n[Sample {processed_count + 1}] Processing: {fname}")
            logger.info(f"  [ORIGINAL] img shape: {img.shape}")
            
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
                logger.info(f"  ⏭️  Skipped (not in target patients)")
                skipped_count += 1
                continue
            
            logger.info(f"  Patient: {patient_id}")
            
            # Keep original for comparison
            img_original = img.unsqueeze(0).to(encoder.device)
            logger.info(f"  [SHAPES] img_original (with batch): {img_original.shape}")
            
            # === ABLATION: Start from RANDOM GAUSSIAN NOISE ===
            logger.info(f"  [GENERATION] Creating random noise with shape: {img_shape}")
            random_noise = torch.randn(img_shape, device=encoder.device, dtype=torch.float32)
            random_noise = random_noise.unsqueeze(0)  # [1, C, H, W]
            logger.info(f"  [GENERATION] random_noise (with batch): {random_noise.shape}")
            
            # Use real genomic features
            genomic_feat = feat.unsqueeze(0).to(encoder.device)
            logger.info(f"  [GENERATION] genomic_feat shape: {genomic_feat.shape}")
            
            with torch.no_grad():
                try:
                    # Decode random noise using genomic features as guidance
                    logger.info(f"  [DECODE] Starting decode_image()...")
                    generated = encoder.decode_image(random_noise, genomic_feat, T=decode_steps)
                    logger.info(f"  [DECODE] generated shape: {generated.shape}")
                    
                    # Compute metrics against original image
                    logger.info(f"  [METRICS] Computing structural similarity...")
                    ssim, mse, ms_ssim_score = compute_structural_similarity_debug(
                        generated, img.unsqueeze(0).squeeze(0), ms_ssim
                    )
                    
                    logger.info(f"  ✅ Success! SSIM={ssim:.4f}, MSE={mse:.4f}, MS-SSIM={ms_ssim_score:.4f}")
                    
                    # Create patient subdirectory
                    patient_dir = os.path.join(save_path, sanitize_filename(patient_id))
                    os.makedirs(patient_dir, exist_ok=True)
                    
                    # Save generated image
                    tile_label = os.path.basename(str(fname)).replace('.png', '').replace('.jpg', '')
                    safe_label = sanitize_filename(tile_label)
                    filename = f"generated_random_{processed_count:05d}_{safe_label}.png"
                    encoder.save_image(generated.squeeze(0), filename, patient_dir)
                    logger.info(f"  💾 Saved to: {os.path.join(patient_dir, filename)}")
                    
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
                        'ablation_mode': 'RANDOM_NOISE_GENOMIC_DEBUG',
                        'decode_steps': decode_steps
                    }
                    results.append(result_entry)
                    
                    if patient_id not in patient_tile_counts:
                        patient_tile_counts[patient_id] = 0
                    patient_tile_counts[patient_id] += 1
                    
                    processed_count += 1
                    pbar.set_postfix({'processed': processed_count, 'skipped': skipped_count})
                    
                except Exception as e:
                    logger.error(f"  ❌ Generation failed: {e}", exc_info=True)
                    skipped_count += 1
                    continue
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(save_path, 'autoencoder_evaluation_DEBUG.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"\n" + "="*70)
        logger.info(f"✅ DEBUG RUN COMPLETE!")
        logger.info(f"="*70)
        logger.info(f"   Processed: {processed_count}")
        logger.info(f"   Skipped:   {skipped_count}")
        logger.info(f"   Results: {csv_path}")
        logger.info(f"\n   Metrics:")
        logger.info(f"   SSIM:    {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
        logger.info(f"   MS-SSIM: {df['MS-SSIM'].mean():.4f} ± {df['MS-SSIM'].std():.4f}")
        logger.info(f"   MSE:     {df['MSE'].mean():.4f} ± {df['MSE'].std():.4f}")
        
        logger.info(f"\n   📊 Tiles per patient:")
        for pid, count in sorted(patient_tile_counts.items()):
            logger.info(f"      {pid}: {count} tiles")
        logger.info("="*70)
    else:
        logger.warning(f"❌ No results to save. Processed: {processed_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    # === CONFIGURATION ===
    encoder_path = "/mnt/bulk-saturn/maralampert/genhist/experiments/20260306_mopadi_training_attempt_from_scratch/autoenc/last.ckpt"
    
    save_path = "/mnt/bulk-saturn/maralampert/genhist/experiments/20260306_mopadi_training_attempt_from_scratch/reconstruct_DEBUG_5_tiles"
    
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
        logger.info("✅ Loaded configuration from checkpoint hyperparameters")
        logger.info(f"   channels: {getattr(conf, 'channels', 'N/A')}")
        logger.info(f"   img_size: {getattr(conf, 'img_size', 'N/A')}")
    except Exception as e:
        logger.warning(f"⚠️  Could not load config from checkpoint ({e}); falling back to pancancer_autoenc()")
        try:
            conf = pancancer_autoenc()
            logger.info(f"   channels: {getattr(conf, 'channels', 'N/A')}")
            logger.info(f"   img_size: {getattr(conf, 'img_size', 'N/A')}")
        except Exception as ee:
            logger.error(f"Fallback config factory failed: {ee}")
            sys.exit(1)
    
    # Load dataset
    try:
        logger.info(f"Loading tile images from: {tile_zip_dirs}")
        logger.info(f"Genomic features from: {genomic_feature_dirs}")
        
        test_dataset = ZipTilesWithGenomicFeatures(
            root_dirs=tile_zip_dirs,
            feature_dirs=genomic_feature_dirs,
            process_only_zips=True,
            skip_zip_validation=True,
            do_normalize=conf.do_normalize,
            do_resize=conf.do_resize,
            img_size=conf.img_size,
        )
        
        logger.info(f"✅ Test dataset contains {len(test_dataset)} tiles")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # DEBUG: Single sample at a time for clarity
            num_workers=0,  # DEBUG: No workers for easier debugging
            shuffle=True,
        )
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Use same patients for testing
    samples_to_reconstruct = ['TCGA-5L-AAT0', 'TCGA-5T-A9QA', 'TCGA-A1-A0SG', 'TCGA-A1-A0SK', 'TCGA-A1-A0SM']
    
    # Image shape from config
    img_shape = (conf.channels, conf.img_size, conf.img_size) if hasattr(conf, 'channels') and hasattr(conf, 'img_size') else (3, 512, 512)
    
    logger.info(f"\n🔍 Using img_shape: {img_shape}")
    
    main(
        encoder_path=encoder_path,
        data_loader=test_loader,
        conf=conf,
        save_path=save_path,
        img_shape=img_shape,
        genomic_feature_dim=512,
        num_samples=5,  # DEBUG: Only 5 samples
        samples_to_reconstruct=samples_to_reconstruct,
        decode_steps=50,
        zip_results=False,
    )
