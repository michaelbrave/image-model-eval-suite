# v1-5-pruned-emaonly

Suite: `core-100`
Checkpoint: `/home/mike/model-store/StableDiffusion/v1-5-pruned-emaonly.safetensors`
Cases: 400
Scored cases: 400

## Model Profile

### Domain Speciality

- `aesthetic`: best at `environment_world`
- `aesthetic_classifier`: best at `anime_cartoon`
- `technical_image_stats`: best at `anime_cartoon`

Domain ranking (aesthetic):
  - `environment_world`: 0.5806
  - `illustration_painting`: 0.5605
  - `anime_cartoon`: 0.5529
  - `design_product`: 0.5429
  - `mixed_general`: 0.5419
  - `cgi_render`: 0.5346
  - `photography`: 0.5260

### Contrast Leaning

The model leans **balanced** (avg brightness: 0.427/1.0).

### Optimal Resolution

- `aesthetic`: best at `wide` (1344×768)
- `aesthetic_classifier`: best at `portrait` (832×1216)
- `technical_image_stats`: best at `square` (1024×1024)

### Preferred Prompt Style

- `technical_image_stats`: `comma-separated`


## Score Counts

- `aesthetic`: 640
- `aesthetic_classifier`: 84
- `technical_image_stats`: 400

## Score Families

- `aesthetic_classifier`: 0.7381
- `technical_image_stats`: 0.6556
- `aesthetic`: 0.5470

## Scorer Scores

- `cafe-aesthetic`: 0.7381
- `brightness-contrast`: 0.6556
- `sd-chad`: 0.5770
- `improved-aesthetic-predictor`: 0.5291
- `aesthetics-scorer`: 0.5248

## Prompt Style Scores

### aesthetic

- `comma-separated`: 0.5422

### aesthetic_classifier

- `comma-separated`: 0.7381

### technical_image_stats

- `comma-separated`: 0.6556

## Domain Scores

### aesthetic

- `environment_world`: 0.5806
- `illustration_painting`: 0.5605
- `anime_cartoon`: 0.5529
- `design_product`: 0.5429
- `mixed_general`: 0.5419
- `cgi_render`: 0.5346
- `photography`: 0.5260

### aesthetic_classifier

- `anime_cartoon`: 0.7381

### technical_image_stats

- `anime_cartoon`: 0.6843
- `design_product`: 0.6719
- `illustration_painting`: 0.6656
- `environment_world`: 0.6613
- `photography`: 0.6581
- `cgi_render`: 0.6412
- `mixed_general`: 0.6275

## Top Probe Scores

### aesthetic

- `perspective`: 0.5814
- `text-design`: 0.5625
- `architecture`: 0.5618
- `composition`: 0.5554
- `portrait`: 0.5514
- `character`: 0.5489
- `motion`: 0.5481
- `lighting`: 0.5474
- `landscape`: 0.5449
- `anatomy`: 0.5423
- `materials`: 0.5403
- `text-detail`: 0.5400
- `product`: 0.5384
- `multi-subject`: 0.5211

### aesthetic_classifier

- `landscape`: 0.9347
- `multi-subject`: 0.9142
- `character`: 0.8986
- `composition`: 0.8910
- `motion`: 0.8867
- `materials`: 0.8370
- `anatomy`: 0.8307
- `text-design`: 0.8080
- `portrait`: 0.8062
- `lighting`: 0.7819
- `architecture`: 0.6973
- `text-detail`: 0.6675
- `product`: 0.6339

### technical_image_stats

- `product`: 0.6738
- `text-design`: 0.6643
- `text-detail`: 0.6625
- `materials`: 0.6605
- `character`: 0.6604
- `portrait`: 0.6597
- `lighting`: 0.6553
- `anatomy`: 0.6542
- `architecture`: 0.6505
- `motion`: 0.6472
- `landscape`: 0.6464
- `composition`: 0.6443
- `perspective`: 0.6236
- `multi-subject`: 0.5848

## Notes

This card was generated from recorded scorer outputs. Score families are kept separate because aesthetic, prompt reward, classifier, and technical image-stat scores are not interchangeable.

**Speciality**: the domain(s) where this model scores highest. **Contrast leaning** measures whether images tend toward light, dark, or balanced exposure. **Optimal resolution** is the aspect ratio that yields the best scores. **Preferred prompt style** is determined via a separate style-find test run.
