# Zip File Support for Genomic Conditioning

The `ZipTilesWithGenomicFeatures` class enables training with genomic features using **zip files instead of tar shards**. This avoids the overhead of converting large datasets to WebDataset format.

## Usage

### Configuration

In your config YAML:

```yaml
# Key settings:
feat_extractor: 'genomic'           # Enable genomic conditioning
use_web_dataset: false              # Use zip files instead of tar shards

# Data paths:
data_dirs:                          # List of dirs containing .zip files with tiles
  - /path/to/tiles_dir1
  - /path/to/tiles_dir2

feature_dirs:                       # Dict or list of dirs with .h5 genomic features
  COHORT_A: /path/to/features_cohortA
  COHORT_B: /path/to/features_cohortB
  # OR:
  # - /path/to/features/
```

### Expected File Structure

**Tiles (in zip files):**
```
/path/to/tiles_dir/
├── PATIENT.COHORT.zip           # Each zip contains tiles for one patient
│   ├── tile_0_0.png
│   ├── tile_0_1.png
│   └── ...
└── PATIENT2.COHORT.zip
```

**Genomic features (HDF5):**
```
/path/to/features_cohort/
├── PATIENT.h5                    # One H5 file per patient
├── PATIENT2.h5
└── ...
```

Each `.h5` file should contain a `"feats"` dataset with shape `(D,)` where `D` is the feature dimension (default 512 for VAE-encoded gene expression).

### Metadata Extraction

The dataset automatically extracts:
- **Patient ID**: First two components of filename (e.g., `TCGA.AB` from `TCGA.AB.XXXX.zip`)
- **Cohort**: Directory name of the zip file (e.g., `BRCA` from `BRCA/TCGA.AB.XXXX.zip`)

### Training

```python
from mopadi.configs.config import TrainConfig

config = TrainConfig.load_from_yaml('config.yaml')
dataset = config.make_dataset(use_web_dataset=False)  # ← Forces ZipTilesWithGenomicFeatures
loader = config.make_loader(dataset)

# Iterate:
for batch in loader:
    images = batch['img']           # (B, C, H, W)
    genomic_feats = batch['feat']   # (B, D)
    coords = batch['coords']        # (B, 2) - tile coordinates
    # Train as usual
```

┌─────────────────────────────────────────────────────────────────┐
│                     RECONSTRUCTION PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

LOADING PHASE:
═════════════
                    ┌──────────────────┐
                    │  BRCA-tumor-     │
                    │  tiles-all/      │
                    │  (1,112 zips)    │
                    └────────┬─────────┘
                             │
                   ┌─────────┴──────────┐
                   ↓                    ↓
          Extract PNG tile      Extract Patient ID
          512×512, RGB          TCGA-AR-A2LK
                   │                    │
                   ↓                    ↓
        Normalize to [-1,1]    Look up in features/
        [1,3,512,512]           TCGA-AR-A2LK.h5
                   │                    │
                   ↓                    ↓
          Image Tensor            Genomic Features
          img [1,3,512,512]       feat [1,512]
                   │                    │
                   └─────────┬──────────┘
                             ↓
                    ┌────────────────┐
                    │  BATCH [4]     │  ← DataLoader batches
                    │  imgs [4,3,...]│
                    │  feats [4,512] │
                    └────────┬───────┘

INFERENCE PHASE:
════════════════
                        YOUR CHECKPOINT
                        ¦ last.ckpt
                        ↓
    ┌───────────────────────────────────────────┐
    │   Diffusion Autoencoder (beatgans_autoenc)│
    │   ─ U-Net Denoiser (128 base channels)    │
    │   ─ Encoder Network                       │
    │   ─ Decoder Network                       │
    │   ─ Genomic Feature Conditioning          │
    └────────┬──────────────────────────────────┘
             │
    ┌────────┴──────────┐
    ↓                   ↓
ENCODE PHASE        DECODE PHASE
    │                   │
For each tile:      Using SAME genomic feat:
 
img [1,3,512,512]          x_T [noise]
feat [1,512] ──────→ U-Net denoiser (z-conditioned)
    │                    ↓ (20 DDIM steps)
    ↓                    ├─ predict noise
[Encoder Net]          ├─ remove noise
    │                    ├─ condition on z
[250 diffusion       repeat
 steps with z]          └─ output: x̂₀
    ↓
x_T [noise]
    
    x̂₀ [reconstructed]
    ↓
    ├─ SSIM
    ├─ MS-SSIM
    └─ MSE
    
    RESULTS:
    ├─ Reconstructed PNG
    ├─ Metrics (CSV)
    └─ Per-patient organization

## Comparison: Tar Shards vs. Zip Files

| Aspect | Tar Shards (WebDataset) | Zip Files |
|--------|-------------------------|-----------|
| **Speed** | Fastest (streaming) | Fast (local I/O) |
| **Memory** | Minimal (streaming) | Higher (batch loading) |
| **Scalability** | Multi-node training | Single/multi-GPU |
| **Setup** | Requires conversion | Direct use |
| **Disk Space** | Efficient (tar) | ~10% overhead (zip) |

**Use tar shards for:**
- Large datasets (>100K tiles)
- Multi-node / distributed training
- Maximum streaming efficiency

**Use zip files for:**
- Rapid experimentation
- Medium datasets (<100K tiles)
- Avoiding conversion overhead
- Local development

## Performance Notes

- `H5GenomicCache` keeps up to 32 `.h5` files open (configurable via `h5_cache_items`)
- Zip file tile access uses Python's built-in `zipfile` module
- Patient ID → H5 path mapping is cached to avoid repeated filesystem lookups
- Genomic features are cached in memory after first access to same patient

## Troubleshooting

**"No genomic H5 for cohort=X patient=Y"**
- Check `.h5` filename matches patient ID extracted from zip filename
- Ensure feature directory structure matches `feature_dirs` config
- Verify `.h5` contains `"feats"` dataset (or specify `feat_key` parameter)

**Slow tile loading**
- Zip files are sequential; consider distributing zips across disks
- Increase `num_workers` in loader for parallel zip reading

**Memory issues**
- Reduce `batch_size` or `h5_cache_items`
- Use tar shards for better streaming efficiency

## Implementation Details

- **Parent Class**: `DefaultTilesDataset` (already supports `.zip:internal/path` notation)
- **Feature Cache**: `H5GenomicCache` (LRU with max_open file handles)
- **Config Route**: `make_dataset(use_web_dataset=False, feat_extractor='genomic')` → `ZipTilesWithGenomicFeatures`
