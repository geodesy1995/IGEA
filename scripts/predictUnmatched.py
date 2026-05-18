import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import pickle
import pandas as pd
import sys
import configparser
import asyncio
import asyncpg
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
from sklearn import metrics
from attention import Attention
from crossAttention import CrossAttention
import base_llm
from experiment_config import (
    SPATIAL_SCALER_FILENAME,
    UNMATCHED_PAIRS_FILENAME,
    build_spatial_matrix,
    get_experiment_name,
    get_spatial_features,
    get_verifier_settings,
    transform_spatial_matrix,
    write_experiment_metadata,
)

tf.get_logger().setLevel('ERROR')
DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]
ITERATION = int(sys.argv[3])

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

USE_ATTENTION = config.getboolean('entity linking', 'attention')
if not USE_ATTENTION:
    DATASET_LOCATION = os.path.join(DATA_DIR, 'el prediction set.parquet')  # location of parquet file for unmatched candidate pairs
    MODEL_TYPE = config.get('legacy', 'model')
    CLASSIFIER_LOCATION = os.path.join(DATA_DIR, f'{MODEL_TYPE}.sav')  # location of classifier model saved to pickle file
else:
    DATASET_LOCATION = os.path.join(DATA_DIR, UNMATCHED_PAIRS_FILENAME)
    CLASSIFIER_LOCATION = os.path.join(DATA_DIR, 'keras model')
    OSM_TOKENIZER_LOCATION = os.path.join(DATA_DIR, 'osm tokenizer.sav')
    WIKIDATA_TOKENIZER_LOCATION = os.path.join(DATA_DIR, 'wikidata tokenizer.sav')

OUTPUT_PATH = os.path.join(DATA_DIR, 'predicted entity matches.tsv')
PREDICTION_THRESHOLD = config.getfloat('entity linking', 'prediction_threshold')
PREDICT_BATCH_SIZE = config.getint('entity linking', 'predict_batch_size', fallback=1024)
SPATIAL_FEATURES = get_spatial_features(config)
VERIFIER_SETTINGS = get_verifier_settings(config)

# postGIS config
PW_FILENAME = config.get('postGIS', 'passwordfile')
PG_HOST = config.get('postGIS', 'host')
PG_USER = config.get('postGIS', 'user')
PG_DB_NAME = config.get('postGIS', 'dbname')
PG_PORT = config.getint('postGIS', 'port')
TABLE_NAME = config.get('entity linking', 'prediction_table')
WRITE_PREDICTIONS_TO_DB = config.getboolean('entity linking', 'write_predictions_to_db', fallback=True)

print('predicting entity matches')
print(f'-experiment: {get_experiment_name(config)}')
print('-loading dataset of possible new matchings')
print(f'-from: {DATASET_LOCATION}')


def apply_selective_verifier(data, tags, properties, probabilities, prediction):
    verifier_selected = np.zeros(len(prediction), dtype=bool)
    verifier_results = np.array(['disabled'] * len(prediction), dtype=object)
    verifier_confidence = np.full(len(prediction), np.nan, dtype=float)
    verifier_reason = np.array([''] * len(prediction), dtype=object)
    verifier_stats = {
        'verifier_selected_count': 0,
        'verifier_match_count': 0,
        'verifier_non_match_count': 0,
        'verifier_unsure_count': 0,
        'verifier_error_count': 0,
        'estimated_verifier_cost': 0.0,
    }

    if not VERIFIER_SETTINGS.enabled:
        print('-Selective LLM verifier disabled')
        return probabilities, prediction, verifier_selected, verifier_results, verifier_confidence, verifier_reason, verifier_stats

    verifier = base_llm.get_verifier(
        VERIFIER_SETTINGS.provider,
        model=VERIFIER_SETTINGS.model,
        timeout_seconds=VERIFIER_SETTINGS.timeout_seconds,
        max_output_tokens=VERIFIER_SETTINGS.max_output_tokens,
    )
    log_path = os.path.join(DATA_DIR, VERIFIER_SETTINGS.log_filename)
    print('-Running selective LLM verifier on low margin predictions')
    print(f'-provider: {VERIFIER_SETTINGS.provider}')
    print(f'-model: {VERIFIER_SETTINGS.model}')
    print(f'-margin: {VERIFIER_SETTINGS.margin}')
    if VERIFIER_SETTINGS.max_calls > 0:
        print(f'-max calls: {VERIFIER_SETTINGS.max_calls}')
    print(f'-log: {log_path}')

    fieldnames = [
        'wkid',
        'osm_id',
        'probability_before',
        'probability_after',
        'threshold',
        'margin',
        'prediction_before',
        'prediction_after',
        'verifier_result',
        'verifier_confidence',
        'verifier_reason',
        'estimated_cost',
        'spatial_features',
        'osm_tags',
        'kg_properties',
    ]
    verification_count = 0
    result_counts = {'match': 0, 'non-match': 0, 'unsure': 0}

    with open(log_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        for i in range(len(prediction)):
            probability_before = float(probabilities[i])
            if abs(probability_before - PREDICTION_THRESHOLD) > VERIFIER_SETTINGS.margin:
                continue
            if VERIFIER_SETTINGS.max_calls > 0 and verification_count >= VERIFIER_SETTINGS.max_calls:
                continue

            verification_count += 1
            verifier_selected[i] = True
            prediction_before = bool(prediction[i])
            evidence_pack = {
                'wkid': str(data['wkid'].iloc[i]),
                'osm_id': str(data['osm_id'].iloc[i]),
                'probability': probability_before,
                'threshold': PREDICTION_THRESHOLD,
                'margin': VERIFIER_SETTINGS.margin,
                'spatial_features': {
                    feature: float(pd.to_numeric(pd.Series([data[feature].iloc[i]]), errors='coerce').fillna(0.0).iloc[0])
                    for feature in SPATIAL_FEATURES
                    if feature in data.columns
                },
            }
            try:
                decision = verifier.verify_with_details(
                    tags.iloc[i] if hasattr(tags, 'iloc') else tags[i],
                    properties.iloc[i] if hasattr(properties, 'iloc') else properties[i],
                    evidence_pack,
                )
                result = decision.decision if decision.decision in base_llm.VALID_VERIFIER_RESULTS else 'unsure'
                confidence = float(decision.confidence)
                reason = decision.reason
            except Exception as exc:
                result = 'unsure'
                confidence = 0.0
                reason = f'verifier_error: {exc}'
                verifier_stats['verifier_error_count'] += 1
            verifier_results[i] = result
            verifier_confidence[i] = confidence
            verifier_reason[i] = reason
            result_counts[result] += 1

            if result == 'non-match':
                prediction[i] = False
                probabilities[i] = min(probabilities[i], max(0.0, PREDICTION_THRESHOLD - 1e-6))
            elif result == 'match':
                prediction[i] = True
                probabilities[i] = max(probabilities[i], min(1.0, PREDICTION_THRESHOLD + 1e-6))

            estimated_cost = verification_count * (VERIFIER_SETTINGS.cost_per_1k_calls / 1000.0)
            writer.writerow({
                'wkid': evidence_pack['wkid'],
                'osm_id': evidence_pack['osm_id'],
                'probability_before': probability_before,
                'probability_after': float(probabilities[i]),
                'threshold': PREDICTION_THRESHOLD,
                'margin': VERIFIER_SETTINGS.margin,
                'prediction_before': prediction_before,
                'prediction_after': bool(prediction[i]),
                'verifier_result': result,
                'verifier_confidence': confidence,
                'verifier_reason': reason,
                'estimated_cost': estimated_cost,
                'spatial_features': evidence_pack['spatial_features'],
                'osm_tags': tags.iloc[i] if hasattr(tags, 'iloc') else tags[i],
                'kg_properties': properties.iloc[i] if hasattr(properties, 'iloc') else properties[i],
            })

    estimated_total_cost = verification_count * (VERIFIER_SETTINGS.cost_per_1k_calls / 1000.0)
    verifier_stats.update({
        'verifier_selected_count': verification_count,
        'verifier_match_count': result_counts['match'],
        'verifier_non_match_count': result_counts['non-match'],
        'verifier_unsure_count': result_counts['unsure'],
        'estimated_verifier_cost': estimated_total_cost,
    })
    print(f'-Sent {verification_count} pairs to LLM verifier')
    print(f'-Verifier results: {result_counts}')
    print(f'-Estimated verifier cost: {estimated_total_cost:.6f}')
    return probabilities, prediction, verifier_selected, verifier_results, verifier_confidence, verifier_reason, verifier_stats


def parse_bool_series(series):
    if series.dtype == bool:
        return series.astype(bool)
    return series.astype(str).str.lower().isin(['true', '1', 'yes'])


def compute_prediction_metrics(data, prediction, prefix):
    if 'match' not in data.columns:
        return {}
    y_true = parse_bool_series(data['match']).to_numpy()
    y_pred = np.asarray(prediction).astype(bool)
    return {
        f'{prefix}_precision': float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        f'{prefix}_recall': float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        f'{prefix}_f1': float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        f'{prefix}_support': int(y_true.sum()),
        f'{prefix}_predicted_positive': int(y_pred.sum()),
        f'{prefix}_false_positive': int(((~y_true) & y_pred).sum()),
        f'{prefix}_false_negative': int((y_true & (~y_pred)).sum()),
    }

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if USE_ATTENTION:
    data = pd.read_csv(DATASET_LOCATION, delimiter='\t')

    print('-loading classifier')
    print(f'-from: {CLASSIFIER_LOCATION}')
    model = keras.models.load_model(CLASSIFIER_LOCATION, custom_objects={"CrossAttention": CrossAttention, "f1_m": f1_m, "precision_m": precision_m, "recall_m": recall_m})
    nWords = model.input_shape[0][1]
    print('-loading osm tokenizer')
    print(f'-from: {OSM_TOKENIZER_LOCATION}')
    with open(OSM_TOKENIZER_LOCATION, 'rb') as file:
        osm_tokenizer = pickle.load(file)

    print('-loading wikidata tokenizer')
    print(f'-from: {WIKIDATA_TOKENIZER_LOCATION}')
    with open(WIKIDATA_TOKENIZER_LOCATION, 'rb') as file:
        wiki_tokenizer = pickle.load(file)

    tags = data['tags'].astype(str)
    properties = data['properties'].astype(str)
    x_spatial = build_spatial_matrix(data, SPATIAL_FEATURES)
    spatial_scaler_path = os.path.join(DATA_DIR, SPATIAL_SCALER_FILENAME)
    if os.path.exists(spatial_scaler_path):
        print('-loading spatial scaler')
        print(f'-from: {spatial_scaler_path}')
        with open(spatial_scaler_path, 'rb') as file:
            spatial_scaler = pickle.load(file)
        x_spatial = transform_spatial_matrix(x_spatial, spatial_scaler)
    else:
        print('-spatial scaler not found; using raw spatial features')

    text_sequences_osm = osm_tokenizer.texts_to_sequences(list(tags.values))
    text_sequences_osm = tf.keras.preprocessing.sequence.pad_sequences(text_sequences_osm, maxlen=nWords, padding='post')
    x_osm = np.array(text_sequences_osm)

    text_sequences_wiki = wiki_tokenizer.texts_to_sequences(list(properties.values))
    text_sequences_wiki = tf.keras.preprocessing.sequence.pad_sequences(text_sequences_wiki, maxlen=nWords, padding='post')
    x_wiki = np.array(text_sequences_wiki)

    print(f'-predicting matches with threshold: {PREDICTION_THRESHOLD}')
    probabilities = model.predict([x_osm, x_wiki, x_spatial], batch_size=PREDICT_BATCH_SIZE)
    # Flatten probabilities to 1D to ensure shape compatibility
    probabilities = probabilities.flatten()
    prediction = (probabilities >= PREDICTION_THRESHOLD)
    prediction_before_verifier = prediction.copy()
    prediction_metrics_before = compute_prediction_metrics(data, prediction_before_verifier, 'prediction_before_verifier')
    probabilities, prediction, verifier_selected, verifier_results, verifier_confidence, verifier_reason, verifier_stats = apply_selective_verifier(
        data,
        tags,
        properties,
        probabilities,
        prediction,
    )
    prediction_metrics_after = compute_prediction_metrics(data, prediction, 'prediction_after_verifier')
else:
    data = pd.read_parquet(DATASET_LOCATION, engine='pyarrow')

    print('-loading classifier')
    print(f'-from: {CLASSIFIER_LOCATION}')
    with open(CLASSIFIER_LOCATION, 'rb') as file:
        model = pickle.load(file)

    print(f'-predicting matches with threshold: {PREDICTION_THRESHOLD}')
    predicted_values = model.predict_proba(data.iloc[:, 3:])
    probabilities = predicted_values[:, 1]
    prediction = (probabilities >= PREDICTION_THRESHOLD)
    prediction_before_verifier = prediction.copy()
    prediction_metrics_before = compute_prediction_metrics(data, prediction_before_verifier, 'prediction_before_verifier')
    prediction_metrics_after = compute_prediction_metrics(data, prediction, 'prediction_after_verifier')
    verifier_selected = np.zeros(len(prediction), dtype=bool)
    verifier_results = np.array(['not_applicable'] * len(prediction), dtype=object)
    verifier_confidence = np.full(len(prediction), np.nan, dtype=float)
    verifier_reason = np.array([''] * len(prediction), dtype=object)
    verifier_stats = {
        'verifier_selected_count': 0,
        'verifier_match_count': 0,
        'verifier_non_match_count': 0,
        'verifier_unsure_count': 0,
        'verifier_error_count': 0,
        'estimated_verifier_cost': 0.0,
    }

print(f'-Number of predicted matches: {int(prediction.sum())} / {len(prediction)}')
print(f'-matched percentage: {(int(prediction.sum()) / len(prediction)) * 100: .2f}%')

all_prediction_pairs = pd.DataFrame()
all_prediction_pairs['wkid'] = data['wkid']
if 'osm_uid' in data.columns:
    all_prediction_pairs['osm_uid'] = data['osm_uid']
else:
    all_prediction_pairs['osm_uid'] = data['osm_id'].astype(str)
all_prediction_pairs['osm_id'] = data['osm_id']
all_prediction_pairs['probability'] = probabilities
all_prediction_pairs['prediction'] = prediction
all_prediction_pairs['prediction_before_verifier'] = prediction_before_verifier
all_prediction_pairs['verifier_selected'] = verifier_selected
all_prediction_pairs['verifier_result'] = verifier_results
all_prediction_pairs['verifier_confidence'] = verifier_confidence
all_prediction_pairs['verifier_reason'] = verifier_reason

print('-logging predictions for matches')
print(f'-to: {OUTPUT_PATH}')
with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as file:
    all_prediction_pairs.to_csv(file, sep='\t', index=False)
print('-writing complete')

write_experiment_metadata(
    DATA_DIR,
    config,
    SPATIAL_FEATURES,
    verifier_settings=VERIFIER_SETTINGS,
    extra={
        'iteration': ITERATION,
        'prediction_threshold': PREDICTION_THRESHOLD,
        'predicted_match_count': int(prediction.sum()),
        'candidate_count': len(prediction),
        **prediction_metrics_before,
        **prediction_metrics_after,
        **verifier_stats,
        'spatial_scaler': 'standard_scaler_fit_on_train_split' if os.path.exists(os.path.join(DATA_DIR, SPATIAL_SCALER_FILENAME)) else 'raw',
        'spatial_distance_transform': 'log1p',
    },
)

prediction_pairs = all_prediction_pairs[all_prediction_pairs['prediction']]

if prediction_pairs.empty:
    print(f'-no matches to write to {TABLE_NAME}')
    sys.exit(0)

if not WRITE_PREDICTIONS_TO_DB:
    print('-database prediction write disabled by config')
    sys.exit(0)

# update predictions in database
with open(PW_FILENAME, 'r') as file:
    password = file.read().strip()

def generate_rows(df: pd.DataFrame) -> list:
    entries = []
    for index, row in df.iterrows():
        entries.append((str(row['wkid']), str(row['osm_uid']), int(row['osm_id']), float(row['probability']), ITERATION))
    return entries

async def update_predictions():
    conn = await asyncpg.connect(
        user=PG_USER,
        password=password,
        database=PG_DB_NAME,
        host=PG_HOST,
        port=PG_PORT
    )

    sql = f"INSERT INTO {TABLE_NAME} (wkid, osm_uid, osm_id, confidence, iteration) VALUES ($1, $2, $3, $4, $5)"
    i = 0
    batchsize = 1000

    async with conn.transaction():
        while i < len(prediction_pairs):
            offset = min(batchsize, len(prediction_pairs) - i)
            await conn.executemany(sql, generate_rows(prediction_pairs[i:i + offset]))
            i += offset

    await conn.close()

print(f'-writing matches to {TABLE_NAME}')
asyncio.run(update_predictions())
print(f'-writing complete')
