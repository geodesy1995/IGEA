# Iterative Geographic Entity Alignment with Cross-Attention  
This is the README file for the paper Iterative Geographic Entity Alignment with Cross-Attention.
Aim of this file is to explain the steps necessary to run the code provided.
This code has been tested on Python 3.8 and Python 3.9

## SetUp
For easier management of isntalling libraries we suggest using a virtual environment. This can be created in your current working directory with:
```bash
virtualenv venv
```
After activating the virtual environment (e.g. by executing the `activate` script in the venv folder) install the necessary libraries:
```
pip3 install -r requirements.txt
```
Finally download the fasttext model of your choice (e.g. by using the fasttext python library)
```
python3
```
```python
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')
```
Enter the location of the fasttext model into the config file  

[- import the osm file into postgres -]  
check pbfNodes2postGIS folder for more info

## running the experiment  
The experiment can be run by executing the `runExperiment.py` file
```bash
python3 runExperiment.py
```
If desired, the runExperiment file can be run with an additional argument to specify the config file to be used. If ran without arguments, the `config.ini` file from the `config` folder will be used.  

## config  
The config file contains all configurable information for the experiment. It is structured into parts for the different experiment components. The config contains the following options:

### postGIS  
postGIS contains all information about the database connection.  

| option | use |
| ------ | --- |
| host | hostserver of database |
| user | database user to connect with |
| dbname | name of the database to connect to |
| port | port used for database connection |
| passwordfile | path to text file containing the password for connecting to the database |

### nca  
nca contains information necessary for the schema alignment part of the linking process.

| option | use |
| ------ | --- |
| osm_tag_location | path to a csv file containing wikidata keys matched to osm keys|
|osm_key_location | path to a csv file containing wikidata labels matched to osm tags|
|columns_location | textfile containing the names of all columns into which osm tag information has been moved during migration from osm file to database. **Leave this empty if all Tags are imported into a single tag column** |
|relevance_threshold | minimum amount of entities for a class to be added to the alignment dataset |
|prediction_threshold | minimum confidence for predicting a wikidata osm class match |
|latent_space|dimension of the latent space used in schema alignment|
|num_epochs|number of epochs to train for during schema alignment|
|train_verbose| how much information to display during schema alignment training. Choose from: 0, 1, 2|

### dbpedia scrape  
dbpedia scrape contains information defining how and which information to collect from dbpedia  

| option | use |
| ------ | --- |
|dbpedia_source| dbpedia instance to source data from e.g. fr for `fr.dbpedia.com`|
|country| Country that entities need to be contained in. Usually a uppercase database Relation like `France` check dbpedia for more information|


### wikidata scrape  
wikidata scrape contains information defining which information to scrape from wikidata  

| option | use |
| ------ | --- |
|country_id| Wikidata Qid of the desired country to collect for|
|scrape_values| types of information to scrape. Define as list separated by commas. Example with all possible options: `name, popularity, type labels, full properties`|
|name_language|language to prefer for entity names|

### candidate generation  
candidate generation contains options to adhere to during creation of entity pairs

| option | use |
| ------ | --- |
|method | Method used for candidate generation choose from: distance, name|
|max_candidates| maximum amount of osm candidates to generate per wikidata entry |
|dist_threshold| maximum distance in meters between wikdata entity location and osm entity location to still be considered a possible match |


### fasttext
fasttext contains information about the fasttext model used

| option | use |
| ------ | --- |
|location| path to the fasttext model file to use for encoding |

### entity linking
entity linking contains possible options for entity link prediction

| option | use |
| ------ | --- |
|attention| Use self attention based model|
|attention_dimension| output dimension for self attention|
|linear_dimension| output dimension for linear combination of self attention output|
|epochs|number of epochs to train for|
|train_verbose|how much information to display during entity linking training. Choose from: 0, 1, 2|
|prediction_threshold| confidence threshold to predict two entities to be linked|
|base_table|database table containing osm information (has to be created from osm information previous to the experiment)|
|view_name| name of the view used to store osm information from entities considered for candidate generation (will be created, updated and deleted during the experiment)|
|prediction_table|name for table containing entity match predictions (as well as ground truth marked with iteration = 0)(will be created, updated and deleted during the experiment)|
|index_name|Name of the index created for the temp view|


### experiment
experiment contains metadata used to name outputs in ablation runs.

| option | use |
| ------ | --- |
|name|Human-readable experiment name written to reports and metadata files.|

### spatial context
spatial context controls the Urban AI extension that adds relational spatial cues to the original IGEA distance input.

| option | use |
| ------ | --- |
|variant|Named spatial feature set. Supported values: original, bearing, offset, topology, all_spatial, no_bearing, no_offset, no_topology.|
|features|Explicit comma-separated feature list. If set, this overrides the named variant. Supported features: dist, bearing_sin, bearing_cos, d_lat, d_lon, bbox_overlap.|
|encoder_units|Comma-separated Dense layer sizes for the spatial context encoder. The original IGEA baseline uses only dist and keeps the distance-scalar path.|

### llm verifier
llm verifier controls selective verification during iterative bootstrapping. The dummy provider logs low-margin cases without changing predictions. The OpenAI provider can be used as a real verifier after the main full run completes.

| option | use |
| ------ | --- |
|enabled|Whether to run the selective verifier gate on low-margin predictions before adding links to the next iteration.|
|provider|Verifier provider. Supported: dummy, openai.|
|margin|A prediction is sent to the verifier when abs(probability - prediction_threshold) <= margin.|
|cost_per_1k_calls|Cost proxy used for reporting expected verifier cost.|
|model|OpenAI model used when provider=openai. Default: gpt-4o-mini.|
|timeout_seconds|OpenAI request timeout for one verification call.|
|max_output_tokens|Maximum structured-output tokens for one verification call.|
|max_calls|Maximum verifier calls per run. Use 5 for smoke tests; 0 means unlimited.|
|log_filename|TSV file written in each iteration folder with selected low-margin cases and verifier results.|

### meta  
meta contains options for experiment metadata

| option | use |
| ------ | --- |
|data_folder| folder to write experiment files to, auto generated in working directory if left blank|
|num_iterations|number of nca and el iterations to run|
|kg_source| knowledge graph source to use for geo entities. Choose from wikidata or dbpedia.|

### legacy  
Legacy options control different strategies for benchmark runs.

| option | use |
| ------ | --- |
|use_legacy_embeddings| calculate own embeddings based on the osm2kg project. Only a single iteration will be run. |
|num_epochs| number of epochs to train custom embeddings for|
| embedding_dim | dimension of custom embeddings |
|model| **legacy option not used with attention** model to use for prediction, choose from: r_forest, mlp, log_reg, d_tree|
|do_oversampling| **legacy option not used with attention** Flag whether to use SMOTE oversampling during training|

### misc
misc contains other miscellaneous options

| option | use |
| ------ | --- |
|testrun|If set to True, different options during the experiment will be set to reduce runtime and enable a test run for system validation purposes|
|limit|number of entries for dataset to limit to during testruns NOTE: very small datasets can lead to problems such as not finding valid candidates by chance|

## files of interest  
During the experiment information is written into files, that may be of interest for closer inspection and future work.  

| file | content |
| ------ | --- |
| osm rbf.tsv | triplet representation of osm information|
| nca dataset.tsv | matched entities from osm and wikidata as well as encoded information for class match prediction during schema alignment |
| qid_index.tsv | list of wikidata types and their corresponding QIDs |
| predicted classes.tsv | output prediction for matching classes generated during schema alignment |
| create view.sql | sql command used for view generation, containing osm classes that were predicted to match |
| wikidata classes.txt | list of all QIDs of wikidata classes that will be used during entity linking |
| wikidata dump.parquet | parquet file containing all wikidata entities from the given classes |
| pair something | candidate pairs generated for entity linking prediction |
| unmatch something | pairs with no valid match to generate new predicted linked entities |
| el train data.parquet| fasttext embedded training data for entity linking |
| model_report | report entity linking model performance after training |
| model.sav | pickel dump of trained el prediction model |
|predicted matches something| result of model prediction on unmatched pairs |
|experiment metadata.json| metadata for the run, including experiment name, KG source, spatial variant, feature list, thresholds, and verifier settings |
|llm verifier log.tsv| low-margin candidates selected for selective LLM verification, with dummy verifier results and cost proxy |
|ablation_summary.csv| optional summary generated by scripts/summarize_ablation_results.py, including delta F1 versus the original IGEA baseline |
|keras model| not a file, but the keras model will be saved in this folder for later use (for import instructions look into predictUnmatched)|
|tokenizer.sav|saved tokenizers for attention model. For example implementation look at predictUnmatched|


## scripts  
In this chapter the function of each script will be outlined briefly. The scripts are listed in order of use during the linking pipeline.

| file | task |
| ------ | --- |
|attention|file containing the Attention class for easy availability|
| prepareSchema | generate necessary tables in postgres |
| osm2rdf | generate rdf data of linked osm entities from postgres |
| readRDFWikidata | fetch wikidata information for linked entities generated in osm2rdf |
| schemaMatch | Train model on entity matches an predict class matches |
| reformClasses | use schemaMatch information to generate view in postgres containing osm entities of the predicted class matches, create list of corresponding wikidata qids for entity linking |
| scrapeWikidata | collect all wikidata entities for previous generated QIDs |
| candidateGeneration | generate candidate pairs for wikidata entities from postgres view, split into pairs containing a match and unmatched pairs |
| computeFTEmbeddings | generate fasttext embeddings for entity linking dataset |
| entityLinking | train entity linking predictor |
|entityLinkingAttention| train entity linking predictor with self attention (embeddings will be selfgenerated)|
| predict unmatched | predict possible matches on previously not matchable candidate pairs and write to database |
| experiment_config | shared experiment helpers for spatial feature selection, verifier settings, filenames, and metadata |
| base_llm | verifier interface and dummy verifier used by the selective LLM gate |
| build_nca_vocab | regenerate osmTagKeyWiki.csv and osmKeyWiki.csv from linked OSM entities when the original NCA vocabulary files are unavailable |
| run_ablation_matrix | generate per-variant configs for original IGEA, spatial ablations, leave-one-out ablations, and dummy-gate runs |
| run_cached_ablation_matrix | cached ablation runner for DBpedia or Wikidata that reuses common NCA/KG/candidate artifacts across spatial variants |
| summarize_ablation_results | collect class_report and metadata files into a CSV table with delta F1 versus original IGEA |

### legacy Embeddings
Scripts for comparison to previous work
| file | task |
| ------ | --- |
| transformForKV | Use candidate pair table to generate key value pairs for given tags. Key value pairs will be used during generation of embeddings |
| embeddingKeyValue | Use key value pairs to calculate custom embeddings |
| prepareTrainingFromKV | Combine embeddings and training pairs for use in entity Linking. |

## Urban AI extension experiment setup

This fork adds an experiment layer for the abstract "GeoEA for Urban AI: Extending IGEA with Spatial Context Encoding and Selective LLM Verification". The goal is to keep the original IGEA benchmark setting and cross-attention matcher intact, while testing which added spatial cues improve performance over the original distance-only baseline.

The original IGEA baseline is represented by:

```ini
[spatial context]
variant=original
features=dist

[llm verifier]
enabled=False
```

The main abstract-aligned model is represented by:

```ini
[spatial context]
variant=all_spatial
features=dist,bearing_sin,bearing_cos,d_lat,d_lon,bbox_overlap

[llm verifier]
enabled=False
provider=dummy
```

The current verifier is intentionally dummy-first. It allows the low-margin gate, logging format, and cost proxy to be evaluated reproducibly before connecting a paid or local LLM. Since the dummy verifier returns unsure, it does not change neural predictions.

Spatial numeric inputs are normalized before entering the neural matcher. The `dist` column is transformed with `log1p`, then a `StandardScaler` is fit on the training split and saved as `spatial scaler.sav`. Prediction uses the same saved scaler. This keeps mixed-scale features such as distance, bearing, coordinate offsets, and bbox overlap comparable across ablation variants.

### Local Docker runtime

The recommended local runtime keeps Docker operations in the IGEA folder while storing PostgreSQL data in a Docker named volume.

For the default local setup, create `config/pw.txt` with one line:

```text
igea
```

Or copy the provided example:

```powershell
Copy-Item config\pw.example.txt config\pw.txt
```

Start PostGIS:

```powershell
docker compose up -d postgis
```

Check the database:

```powershell
docker compose exec postgis psql -U user -d db -c "\dt"
```

Download the Ireland/Northern Ireland OSM extract and the English fastText model:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_ireland_assets.ps1
```

The download script writes:

- `data/raw/ireland-and-northern-ireland-latest.osm.pbf`
- `models/fasttext/cc.en.300.bin`

Import OSM features into PostGIS:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\import_ireland_osm.ps1
```

The import script uses the Docker image `iboates/osm2pgsql:latest`, so `osm2pgsql` does not need to be installed on the Windows host. It creates the `ireland_features` table expected by the experiment configs. The table stores nodes, ways, and selected relations with `osm_uid`, `osm_type`, `osm_id`, `tags`, and Web Mercator geometry. Distance, bearing, and offset features use the geometry centroid, while `bbox_overlap` uses `ST_Intersects(OSM geometry, KG point)`.

The default experiment configs use an English-language Ireland setup:

| setting | value |
| ------ | --- |
|OSM extract|`data/raw/ireland-and-northern-ireland-latest.osm.pbf`|
|PostGIS table|`ireland_features`|
|Wikidata country|`Q27`|
|DBpedia country|`Ireland`|
|fastText model|`models/fasttext/cc.en.300.bin`|

The Geofabrik extract includes Ireland and Northern Ireland. The first Wikidata configuration is Ireland-focused (`Q27`), so report the experiment as Ireland-focused against the Ireland/Northern Ireland OSM extract unless a stricter Ireland-only extract is used.

### Regenerating NCA vocabulary CSVs

The original NCA/IGEA setup expects two OSM vocabulary files:

```ini
[nca]
osm_tag_location=./config/osmTagKeyWiki.csv
osm_key_location=./config/osmKeyWiki.csv
```

If the original author-provided CSVs are unavailable, regenerate IGEA-compatible vocabulary files from the same OSM extract used for the experiment. This is not a byte-for-byte reproduction of the original artifact; it keeps the NCA pipeline intact by deriving the vocabulary from linked OSM entities in the experimental extract. All ablation variants should use the same regenerated CSVs, so original-versus-spatial comparisons remain controlled.

From an RDF triples file such as `osm rbf.tsv`:

```bash
python scripts/build_nca_vocab.py "data/experiment X/it_1/osm rbf.tsv" --input-format rdf --link-keys both --tag-output config/osmTagKeyWiki.csv --key-output config/osmKeyWiki.csv
```

From the Ireland/Northern Ireland OSM PBF extract:

```powershell
venv\Scripts\python scripts\build_nca_vocab.py data\raw\ireland-and-northern-ireland-latest.osm.pbf --input-format pbf --link-keys both --tag-output config\osmTagKeyWiki.csv --key-output config\osmKeyWiki.csv
```

Defaults keep only linked OSM entities with `wikidata` or `wikipedia`, write a `Keys` column for `osmKeyWiki.csv`, write a `Tags` column for `osmTagKeyWiki.csv`, require key-value tags to appear at least twice, and drop boolean, numeric, URL-like, and identifier-like values to avoid exploding the schema-alignment feature space. A stats JSON is written beside the tag CSV with linked entity counts, written key/tag counts, and dropped-value reasons. For the current full-feature Ireland/Northern Ireland import, the regenerated vocabulary contains 74,760 linked OSM entities, 1,977 keys, and 22,085 key-value tags.

Run a one-iteration smoke experiment before the full matrix:

```powershell
venv\Scripts\python runExperiment.py config\config_smoke_ireland_wikidata.ini
```

If Wikidata Query Service is temporarily rate-limiting requests, run the DBpedia smoke config to validate the local pipeline:

```powershell
venv\Scripts\python runExperiment.py config\config_smoke_ireland_dbpedia.ini
```

The DBpedia configs use `entity_source=osm_linked`. Instead of relying only on `dbo:country`, OSM `wikipedia` tags are normalized to DBpedia resource titles and queried in batches. DBpedia entities without DBpedia coordinates are excluded from the main run; OSM coordinates are not used as KG-coordinate fallback.

The Wikidata configs also support `entity_source=osm_linked`. In this mode, OSM `wikidata=Q...` tags seed the KG entity list directly, and `scrapeWikiData.py` queries those QIDs for Wikidata coordinates and properties. This avoids the earlier Ireland-only `P17=Q27` country/class bottleneck and makes Wikidata comparable to the OSM-linked DBpedia setup.

Candidate generation has the following safeguards for the OSM-linked DBpedia setup:

| option | use |
| ------ | --- |
|dist_threshold|Main scientific configs use a 2,500 m ordinary nearest-neighbor radius.|
|max_candidates|Main scientific configs keep the nearest 100 ordinary candidates per KG entity.|
|direct gold preservation|Direct OSM `wikipedia`/`wikidata` positives are merged back even when they are outside 2,500 m or outside the top 100 ordinary candidates.|
|max_candidate_area_m2|Exclude very large polygons from ordinary nearest-neighbor candidates unless they are the direct OSM-DBpedia linked feature.|
|max_train_false_per_entity|Keep all positive pairs, but cap false training pairs per KG entity to control class imbalance and runtime.|
|query_timeout_ms|Set a PostGIS statement timeout per candidate query so one slow entity cannot stall the whole run.|

The cached runner writes `coverage_report.csv`, `candidate_generation_audit.csv`, and `candidate_audit.csv` before full variants run. The default gate requires at least 100 true train pairs, at least 2,000 train rows, split-aware test true support of at least 20, zero missing direct gold positives, and non-zero `bbox_overlap` support for topology variants. Reused common artifacts must include a matching `candidate_generation_audit.csv`; stale 10 km / 200-candidate artifacts fail audit and must be regenerated.

Leakage controls are required for scientific runs:

| control | purpose |
| ------ | --- |
|OSM feature filtering|The OSM candidate text excludes `wkid`, `confidence`, and `iteration`; these are labels or prediction metadata, not input features.|
|group split|The attention matcher uses `split_strategy=group` and `split_group_column=wkid`, so candidates for the same KG entity do not appear in both train and test splits.|

Any result produced before these controls were added should be treated as a pipeline smoke test only, not as a valid performance estimate.

The Docker-based OSM import follows the osm2pgsql Docker workflow documented at https://osm2pgsql.org/doc/install/docker.html.

### Spatial ablation matrix

Use the ablation runner to generate configs for both Wikidata and DBpedia:

```powershell
venv\Scripts\python scripts\run_ablation_matrix.py config\config_ireland_wikidata.ini config\config_ireland_dbpedia.ini
```

The command above is a dry run: it writes generated configs and prints the commands. To execute the full matrix:

```powershell
venv\Scripts\python scripts\run_ablation_matrix.py config\config_ireland_wikidata.ini config\config_ireland_dbpedia.ini --execute
```

For DBpedia or Wikidata runs, use the cached runner after one common run has produced NCA/KG/candidate artifacts. This avoids repeating slow KG/SPARQL collection for every spatial variant:

```powershell
venv\Scripts\python scripts\run_cached_ablation_matrix.py config\config_ireland_dbpedia.ini --output-root .\data\ablation_dbpedia_osm_linked_2500m_max100
```

For Wikidata:

```powershell
venv\Scripts\python scripts\run_cached_ablation_matrix.py config\config_ireland_wikidata.ini --output-root .\data\ablation_wikidata_osm_linked_2500m_max100
```

### GPU execution on Windows

Native Windows TensorFlow 2.11+ does not support NVIDIA GPU execution. This repository therefore uses a Linux TensorFlow GPU Docker image for GPU runs while keeping the project files on the Windows workspace.

Verify GPU TensorFlow:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gpu_docker.ps1 -Build
```

Run the cached DBpedia matrix on GPU:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gpu_docker.ps1 -CommandLine "python scripts/run_cached_ablation_matrix.py config/config_ireland_dbpedia_gpu.ini --output-root ./data/ablation_dbpedia_osm_linked_2500m_max100_gpu --seeds 42,43,44,45,46"
```

Run the cached Wikidata matrix on GPU:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gpu_docker.ps1 -CommandLine "python scripts/run_cached_ablation_matrix.py config/config_ireland_wikidata_gpu.ini --output-root ./data/ablation_wikidata_osm_linked_2500m_max100_gpu --seeds 42,43,44,45,46"
```

The GPU configs use `host.docker.internal` for PostGIS because `localhost` inside the GPU container refers to the container itself. Start PostGIS with `docker compose up -d postgis` before running the GPU experiment.

To reuse common artifacts from an earlier matrix directory:

```powershell
venv\Scripts\python scripts\run_cached_ablation_matrix.py config\config_ireland_dbpedia.ini --output-root .\data\ablation_dbpedia_cached_scaled --reuse-common-dir .\data\ablation_dbpedia_cached\20260517-110353\_common\it_1
```

Current OSM-linked DBpedia smoke artifacts can be reused with:

```powershell
venv\Scripts\python scripts\run_cached_ablation_matrix.py config\config_smoke_ireland_dbpedia.ini --variants original,all_spatial --seeds 42 --output-root .\data\ablation_dbpedia_osm_linked_smoke_variants_fast --reuse-common-dir .\data\ablation_dbpedia_osm_linked_smoke\20260517-230315\_common\it_1
```

Default variants:

| variant | purpose |
| ------ | --- |
|original|Original IGEA-style distance-only baseline.|
|bearing|Distance plus bearing_sin and bearing_cos.|
|offset|Distance plus coordinate offsets d_lat and d_lon.|
|topology|Distance plus bbox_overlap.|
|all_spatial|Full Spatial Context Encoder.|
|no_bearing|Leave-one-out test removing bearing from all_spatial.|
|no_offset|Leave-one-out test removing coordinate offsets from all_spatial.|
|no_topology|Leave-one-out test removing topology from all_spatial.|
|all_spatial_dummy_gate|Full spatial model with dummy selective verifier logging enabled.|
|all_spatial_real_llm_gate_margin003|OpenAI verifier follow-up using all_spatial artifacts and a 0.03 low-margin gate.|
|all_spatial_real_llm_gate_margin005|OpenAI verifier follow-up using all_spatial artifacts and a 0.05 low-margin gate.|
|all_spatial_real_llm_gate_margin010|OpenAI verifier follow-up using all_spatial artifacts and a 0.10 low-margin gate.|

After runs complete, summarize the result folders:

```powershell
venv\Scripts\python scripts\summarize_ablation_results.py .\data\ablation_matrix --output ablation_summary.csv
```

The summary table reports precision, recall, F1, delta_f1_vs_original, predicted match counts, and spatial preprocessing metadata. Use delta_f1_vs_original as the primary indicator for which spatial cue contributes most over the original IGEA baseline. The dummy-gate variant should be interpreted through verifier_selected_count and llm verifier log.tsv, not as an LLM quality result.

After the leakage-free full run completes, run the OpenAI verifier follow-up without restarting the full ablation matrix:

```powershell
$env:OPENAI_API_KEY="sk-..."
venv\Scripts\python scripts\run_openai_verifier_followup.py config\config_ireland_dbpedia.ini --source-matrix .\data\ablation_dbpedia_osm_linked_full_leakage_free\20260518-071747 --output-root .\data\llm_verifier_followup_openai --margins 0.03,0.05,0.10 --seeds 42,43,44,45,46
```

For a cost-safe smoke test, add `--max-calls 5`. The follow-up copies completed `all_spatial` artifacts, sets `write_predictions_to_db=False`, and writes verifier-adjusted prediction metrics to metadata and `llm verifier log.tsv`.

### English-region alternatives

In the IGEA paper, "unseen" primarily refers to OSM/KG entities without existing links inside a selected country, not a held-out country split. The reported evaluation countries are France, Germany, India, Italy, Netherlands, Spain, and USA. If a paper-style evaluated English-language country is needed, USA and India are available choices, but USA is large and expensive to run at country scale.

For an English-speaking country outside that reported evaluation list, New Zealand is the cleanest practical alternative: country-scale, English-language DBpedia/OSM tags, and much smaller than USA, Canada, Australia, or Great Britain. A New Zealand run would require a new PBF download, PostGIS import table, regenerated NCA vocabulary, and DBpedia/Wikidata country config.
