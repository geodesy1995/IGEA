# IGEA Experiment Run Status

## Ready

- Local Windows Python commands use `venv\Scripts\python`.
- GPU experiment commands use `scripts\run_gpu_docker.ps1`, which runs the project inside `igea-tensorflow-gpu:2.12`.
- Core Python dependencies are installed in `D:\projects\IGEA\venv`.
- `config/pw.txt` exists for local Docker PostGIS password `igea`.
- Docker PostGIS data/config is prepared. Start it with `docker compose up -d postgis` before the next experiment run.
- Ireland/Northern Ireland PBF is present at `data/raw/ireland-and-northern-ireland-latest.osm.pbf`.
- English fastText model is present at `models/fasttext/cc.en.300.bin`.
- Regenerated NCA CSVs exist:
  - `config/osmTagKeyWiki.csv`
  - `config/osmKeyWiki.csv`
- Docker/PostGIS runtime files are in `docker-compose.yml`, `docker/postgis/init.sql`, and `docker/osm2pgsql/ireland_features.lua`.
- `scripts/import_ireland_osm.ps1` imports full OSM node/way/relation features through Docker-based `osm2pgsql`.
- `scripts/build_nca_vocab.py` can regenerate `config/osmTagKeyWiki.csv` and `config/osmKeyWiki.csv` from PBF or RDF.
- `scripts/run_cached_ablation_matrix.py` can reuse DBpedia or Wikidata NCA/KG/candidate artifacts across ablation variants and seeds.
- `scripts/summarize_ablation_results.py` writes aggregate and raw ablation summaries.
- `scripts/cap_candidate_train_pairs.py` keeps all positive pairs and caps false pairs per KG entity for runtime/class balance.

## Current Data State

PostGIS full-feature import:

```text
table = ireland_features
feature_count = 9,967,649
nodes = 2,427,451
ways = 7,432,499
relations = 107,699
wikidata_features = 71,270
wikipedia_features = 11,033
```

Regenerated NCA vocabulary from the full Ireland/Northern Ireland PBF:

```text
linked_entities = 74,760
written_keys = 1,977
written_tags = 22,085
stats = config/osmTagKeyWiki.stats.json
```

DBpedia OSM-linked coverage:

```text
coverage_report = data/ablation_dbpedia_osm_linked_smoke/20260517-230315/_common/it_1/coverage_report.csv
osm_wikipedia_tag_rows = 11,033
normalized_unique_titles = 4,495
queried_titles = 4,495
dbpedia_entities_with_coordinates = 3,925
dbpedia_entities_without_coordinates = 570
```

Candidate audit after positive-preserving negative cap:

```text
candidate_audit = data/ablation_dbpedia_osm_linked_smoke_variants_fast/20260518-034715/_common/it_1/candidate_audit.csv
train_rows = 79,353
train_true = 4,373
train_false = 74,980
estimated_test_true_support = 874.6
bbox_nonzero_rows = 12,154
passed = True
```

The uncapped candidate file is retained at:

```text
data/ablation_dbpedia_osm_linked_smoke/20260517-230315/_common/it_1/train pairs.tsv.uncapped.tsv
```

## Latest Smoke Results

OSM-linked DBpedia smoke variants completed before the leakage guard was added:

```text
root = data/ablation_dbpedia_osm_linked_smoke_variants_fast/20260518-034715
summary = data/ablation_dbpedia_osm_linked_smoke_variants_fast/20260518-034715/ablation_summary.csv
raw_runs = data/ablation_dbpedia_osm_linked_smoke_variants_fast/20260518-034715/ablation_runs.csv
variants = original, all_spatial
seed = 42
epochs = 3
sequence_length_cap = 128
max_vocabulary_size = 50,000
class_weight = True
```

Smoke metrics on held-out train-pair split:

```text
original:
  precision = 0.71
  recall = 1.00
  f1 = 0.83
  support = 875
  predicted_match_count = 471
  candidate_count = 30,481
  runtime = 01:08:30

all_spatial:
  precision = 0.69
  recall = 1.00
  f1 = 0.81
  support = 875
  predicted_match_count = 578
  candidate_count = 30,481
  runtime = 00:44:33
```

Interpretation: this run proved that the enlarged OSM-linked pair pipeline can reach sufficient positive support. However, it is not a valid performance estimate because candidate text still contained `wkid` and the model used row-level splitting at the time. Regenerate candidate pairs and rerun smoke after the leakage fixes below before using metrics.

## Implemented Normalization Changes

- DBpedia entity collection supports `entity_source=osm_linked`.
- OSM `wikipedia` tags are normalized to DBpedia resource titles.
- DBpedia entities without DBpedia coordinates are excluded from the main KG dump.
- Wikidata entity collection now supports `entity_source=osm_linked` using OSM `wikidata=Q...` tags as the seed entity list.
- `readRDFWikidata.py` now preserves OSM type prefixes for node/way/relation IDs and fixes batched `VALUES` queries for QIDs.
- OSM coordinate fallback is not used for KG coordinates.
- `ireland_features` includes nodes, ways, and selected relations with geometry.
- Candidate rows track `osm_uid` so nodes, ways, and relations do not collide on numeric `osm_id`.
- Candidate text now excludes `wkid`, `confidence`, and `iteration` to avoid label/prediction metadata leakage.
- Main scientific configs now use `dist_threshold=2500` and `max_candidates=100`; automatic 25 km retry expansion is disabled.
- Direct OSM `wikipedia`/`wikidata` gold positives are preserved even when they fall outside the 2.5 km ordinary candidate radius or outside the top 100 ordinary candidates.
- Candidate generation writes `candidate_generation_audit.csv`; the cached runner merges its gold-preservation metrics and split-aware positive support into `candidate_audit.csv`.
- The cached runner rejects reused common artifacts when their generation audit is missing or their `dist_threshold` / `max_candidates` do not match the current config.
- `bbox_overlap` is still generated as a diagnostic column but has been removed from model inputs and default ablation variants.
- Distance, bearing, and coordinate offsets use OSM geometry centroid.
- Very large polygons are excluded from normal candidate search unless they are directly linked to the KG title.
- PostGIS candidate queries have per-query timeouts.
- Training false rows are capped per KG entity while all positive rows are retained.
- Attention model now supports deterministic seed, sequence-length cap, vocabulary cap, configurable batch size, early stopping, and class weights.
- Attention model now uses `split_strategy=group` and `split_group_column=wkid` by default, so the same KG entity does not appear in both train and test splits.

## Blocked / Not Final Yet

- Wikidata may still be limited by Wikidata Query Service rate limits, but it now has the same OSM-linked path as DBpedia.
- The previous CPU full DBpedia ablation matrix was manually cancelled after `dbpedia_original_seed42` completed and `dbpedia_original_seed43` had started.
- The later GPU DBpedia run under `data/ablation_dbpedia_osm_linked_gpu` used the exploratory 10 km / 200-candidate setting and was manually cancelled. Do not use it as the paper main result.
- GPU TensorFlow is available through Docker. Smoke test result: TensorFlow 2.12.0, `built_with_cuda=True`, `GPU:0` detected on RTX 4060.
- The current `original`/`all_spatial` smoke scores are leakage-invalid. They should stay in the record only as a pipeline validation run.

## Rebuild Commands

Start PostGIS:

```powershell
Copy-Item config\pw.example.txt config\pw.txt
docker compose up -d postgis
docker compose exec postgis psql -U user -d db -c "\dt"
```

Download Ireland/Northern Ireland PBF and English fastText:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_ireland_assets.ps1
```

Import OSM into `ireland_features`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\import_ireland_osm.ps1
```

Generate NCA vocabulary from Ireland PBF:

```powershell
venv\Scripts\python scripts\build_nca_vocab.py data\raw\ireland-and-northern-ireland-latest.osm.pbf --input-format pbf --link-keys both --tag-output config\osmTagKeyWiki.csv --key-output config\osmKeyWiki.csv
```

Regenerate OSM-linked DBpedia common artifacts:

```powershell
venv\Scripts\python scripts\run_cached_ablation_matrix.py config\config_ireland_dbpedia.ini --variants original --seeds 42 --output-root .\data\ablation_dbpedia_osm_linked
```

Run GPU smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gpu_docker.ps1 -Build
```

Run OSM-linked DBpedia matrix on GPU:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gpu_docker.ps1 -CommandLine "python scripts/run_cached_ablation_matrix.py config/config_ireland_dbpedia_gpu.ini --output-root ./data/ablation_dbpedia_osm_linked_gpu --seeds 42,43,44,45,46"
```

Run OSM-linked Wikidata matrix on GPU:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gpu_docker.ps1 -CommandLine "python scripts/run_cached_ablation_matrix.py config/config_ireland_wikidata_gpu.ini --output-root ./data/ablation_wikidata_osm_linked_gpu --seeds 42,43,44,45,46"
```

Reuse the current common artifacts for a fast smoke:

```powershell
venv\Scripts\python scripts\run_cached_ablation_matrix.py config\config_smoke_ireland_dbpedia.ini --variants original,all_spatial --seeds 42 --output-root .\data\ablation_dbpedia_osm_linked_smoke_variants_fast --reuse-common-dir .\data\ablation_dbpedia_osm_linked_smoke\20260517-230315\_common\it_1
```

Summarize results:

```powershell
venv\Scripts\python scripts\summarize_ablation_results.py .\data\ablation_dbpedia_osm_linked_smoke_variants_fast\20260518-034715 --output .\data\ablation_dbpedia_osm_linked_smoke_variants_fast\20260518-034715\ablation_summary.csv --raw-output .\data\ablation_dbpedia_osm_linked_smoke_variants_fast\20260518-034715\ablation_runs.csv
```
