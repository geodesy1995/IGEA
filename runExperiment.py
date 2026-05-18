import os
import time
import sys
import configparser
import subprocess

if len(sys.argv) >= 2:
    CONFIG_PATH = sys.argv[1]
else:
    CONFIG_PATH = './config/config.ini'

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

NUM_ITERATIONS = config.getint('meta', 'num_iterations')
DATA_SOURCE = config.get('meta', 'kg_source')
USE_ATTENTION = config.getboolean('entity linking', 'attention')
USE_LEGACY_EMBEDDINGS = config.getboolean('legacy', 'use_legacy_embeddings')
TESTRUN = config.getboolean('misc', 'testrun')
starttime = time.time()
EXPERIMENT_ID = time.strftime('%d_%m_%Y-%H_%M', time.gmtime(starttime))

if config.get('meta', 'data_folder'):
    DATA_FOLDER = config.get('meta', 'data_folder')
else:
    DATA_FOLDER = f"./data/experiment {EXPERIMENT_ID}/"

os.makedirs(DATA_FOLDER, exist_ok=True)

print(f'running experiment: {EXPERIMENT_ID}')
print(f'-with data source {DATA_SOURCE}')
print(f'-running for {NUM_ITERATIONS} iterations')
print(f'-with config {CONFIG_PATH}')

if TESTRUN:
    print("""
    ---------------- STARTING TESTRUN ----------------
    """)

def run_step(script_path, *args):
    command = [sys.executable, script_path, *args]
    subprocess.run(command, check=True)


run_step('./scripts/prepareSchema.py', CONFIG_PATH)
for iteration in range(1, NUM_ITERATIONS + 1):
    print(f"""
    ==================================
    === starting iteration {iteration} / {NUM_ITERATIONS} ===
    ==================================
    """)
    it_folder = DATA_FOLDER + f'it_{iteration}/'
    os.makedirs(it_folder, exist_ok=True)
    run_step('./scripts/osm2rdf.py', it_folder, CONFIG_PATH)

    # read kg data for osm linked entities
    if DATA_SOURCE == 'wikidata':
        run_step('./scripts/readRDFWikidata.py', it_folder, CONFIG_PATH)
    else:
        # DATA_SOURCE == 'dbpedia'
        run_step('./scripts/readRDFDBpedia.py', it_folder, CONFIG_PATH)

    # train class matchings for classes in osm and kg
    run_step('./scripts/schemaMatch.py', it_folder, CONFIG_PATH)

    # read data for all entities of predicted kg class matches
    if DATA_SOURCE == 'wikidata':
        run_step('./scripts/reformClasses.py', it_folder, CONFIG_PATH)
        run_step('./scripts/scrapeWikiData.py', it_folder, CONFIG_PATH)
    else:
        # DATA_SOURCE == 'dbpedia'
        run_step('./scripts/reformClassesDBP.py', it_folder, CONFIG_PATH)
        run_step('./scripts/scrapeDBPedia.py', it_folder, CONFIG_PATH)

    # generate fitting osm candidate pairs
    run_step('./scripts/candidateGeneration.py', it_folder, CONFIG_PATH)

    if not USE_LEGACY_EMBEDDINGS:
        # embed unstructured text information and train classifier
        if USE_ATTENTION:
            run_step('./scripts/entityLinkingAttention.py', it_folder, CONFIG_PATH)
        else:
            run_step('./scripts/computeFTEmbeddings.py', it_folder, CONFIG_PATH)
            run_step('./scripts/entityLinking.py', it_folder, CONFIG_PATH)

        # predict unknown matches for next iteration
        run_step('./scripts/predictUnmatched.py', it_folder, CONFIG_PATH, str(iteration))

    else:
        # run tests using custom trained embeddings
        run_step('./scripts/legacyEmbeddings/transformForKV.py', it_folder, CONFIG_PATH)
        run_step('./scripts/legacyEmbeddings/embeddingKeyValue.py', it_folder, CONFIG_PATH)
        run_step('./scripts/legacyEmbeddings/prepareTrainingFromKV.py', it_folder, CONFIG_PATH)
        run_step('./scripts/entityLinking.py', it_folder, CONFIG_PATH)
        if NUM_ITERATIONS > 1:
            print('Breaking after one iteration')
            print('Legacy embeddings are not used in multiple iteration strategy')
        break

print('experiment finished')
print(f"-runtime: {time.strftime('%H:%M:%S', time.gmtime(time.time() - starttime))}")
