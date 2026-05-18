import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import keras
from keras.layers import *
from keras.models import Model
from keras import backend as K
import fasttext
import numpy as np
import tensorflow as tf
import configparser
import sys
import pandas as pd
from functools import reduce
from skmultilearn.problem_transform import LabelPowerset
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn import metrics
import pickle
import time
from attention import Attention
from crossAttention import CrossAttention
from experiment_config import (
    SPATIAL_SCALER_FILENAME,
    TEST_PREDICTIONS_FILENAME,
    TRAIN_PAIRS_FILENAME,
    build_spatial_matrix,
    fit_spatial_scaler,
    get_experiment_name,
    get_spatial_encoder_units,
    get_spatial_features,
    transform_spatial_matrix,
    write_experiment_metadata,
)

tf.get_logger().setLevel('ERROR')



DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)


DATASET_PATH = os.path.join(DATA_DIR, TRAIN_PAIRS_FILENAME)
FT_PATH = config.get('fasttext', 'location')
# Model Metadata
NUM_EPOCHS = config.getint('entity linking', 'epochs')
TRAIN_VERBOSE = config.getint('entity linking', 'train_verbose')
PREDICTION_THRESHOLD = config.getfloat('entity linking', 'prediction_threshold')
DIM_ATTENTION = config.getint('entity linking', 'attention_dimension')
DIM_LINEAR = config.getint('entity linking', 'linear_dimension')
MAX_SEQUENCE_LENGTH = config.getint('entity linking', 'max_sequence_length', fallback=128)
MAX_VOCABULARY_SIZE = config.getint('entity linking', 'max_vocabulary_size', fallback=50000)
BATCH_SIZE = config.getint('entity linking', 'batch_size', fallback=512)
EARLY_STOPPING_PATIENCE = config.getint('entity linking', 'early_stopping_patience', fallback=2)
USE_CLASS_WEIGHT = config.getboolean('entity linking', 'use_class_weight', fallback=True)
SPLIT_STRATEGY = config.get('entity linking', 'split_strategy', fallback='group').strip().lower()
SPLIT_GROUP_COLUMN = config.get('entity linking', 'split_group_column', fallback='wkid').strip()
RANDOM_SEED = config.getint('meta', 'random_seed', fallback=42)
SPATIAL_FEATURES = get_spatial_features(config)
SPATIAL_ENCODER_UNITS = get_spatial_encoder_units(config)
USE_SPATIAL_INPUT = len(SPATIAL_FEATURES) > 0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print('entity linking with attention')
print(f'-experiment: {get_experiment_name(config)}')
print(f'-spatial features: {", ".join(SPATIAL_FEATURES) if USE_SPATIAL_INPUT else "none"}')
print('-loading fasttext model')
print(f'-from: {FT_PATH}')

ft_model = fasttext.load_model(FT_PATH)
embedding_dim = 300

print('-loading data')
print(f'-from {DATASET_PATH}')
data = pd.read_csv(DATASET_PATH, delimiter='\t')
tags = data['tags'].astype(str)
properties = data['properties'].astype(str)
if data['match'].dtype == bool:
    y = data['match'].astype(np.float32).values
else:
    y = data['match'].astype(str).str.lower().isin(['true', '1', 'yes']).astype(np.float32).values

starttime = time.time()

# tokenize osm tags
tokenizer_kwargs = {}
if MAX_VOCABULARY_SIZE > 0:
    tokenizer_kwargs["num_words"] = MAX_VOCABULARY_SIZE

tokenizer = tf.keras.preprocessing.text.Tokenizer(**tokenizer_kwargs)
tokenizer.fit_on_texts(list(tags.values))
text_sequences = tokenizer.texts_to_sequences(list(tags.values))

# tokenize wikidata properties
tokenizerWiki = tf.keras.preprocessing.text.Tokenizer(**tokenizer_kwargs)
tokenizerWiki.fit_on_texts(list(properties.values))
text_sequencesWiki = tokenizerWiki.texts_to_sequences(list(properties.values))

avg_osm_len = reduce(lambda count, l: count + len(l), text_sequences, 0) / max(len(text_sequences), 1)
avg_wiki_len = reduce(lambda count, l: count + len(l), text_sequencesWiki, 0) / max(len(text_sequencesWiki), 1)
nWords = max(1, int(max(avg_osm_len, avg_wiki_len)))
if MAX_SEQUENCE_LENGTH > 0:
    nWords = min(nWords, MAX_SEQUENCE_LENGTH)
print(f'-sequence length: {nWords} (avg_osm={avg_osm_len:.2f}, avg_kg={avg_wiki_len:.2f}, cap={MAX_SEQUENCE_LENGTH})')


text_sequences = tf.keras.preprocessing.sequence.pad_sequences(text_sequences, maxlen=nWords, padding='post')
vocab_size = len(tokenizer.word_index) + 1
if MAX_VOCABULARY_SIZE > 0:
    vocab_size = min(vocab_size, MAX_VOCABULARY_SIZE)
X_osm = np.array(text_sequences)

max_length = X_osm.shape[1]

weight_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i >= vocab_size:
        continue
    try:
        embedding_vector = ft_model[word]
        weight_matrix[i] = embedding_vector
    except KeyError:
        weight_matrix[i] = np.random.uniform(0, 0, embedding_dim)


text_sequencesWiki = tf.keras.preprocessing.sequence.pad_sequences(text_sequencesWiki, maxlen=nWords, padding='post')
vocab_sizeWiki = len(tokenizerWiki.word_index) + 1
if MAX_VOCABULARY_SIZE > 0:
    vocab_sizeWiki = min(vocab_sizeWiki, MAX_VOCABULARY_SIZE)
X_wiki = np.array(text_sequencesWiki)

max_lengthWiki = X_wiki.shape[1]

weight_matrixWiki = np.zeros((vocab_sizeWiki, embedding_dim))
for word, i in tokenizerWiki.word_index.items():
    if i >= vocab_sizeWiki:
        continue
    try:
        embedding_vector = ft_model[word]
        weight_matrixWiki[i] = embedding_vector
    except KeyError:
        weight_matrixWiki[i] = np.random.uniform(0, 0, embedding_dim)

def balance(x,y):
    # Import a dataset with X and multi-label y

    lp = LabelPowerset()
    ros = RandomOverSampler(random_state=42)

    # Applies the above stated multi-label (ML) to multi-class (MC) transformation.
    #yt = lp.transform(y)

    X_resampled, y_resampled = ros.fit_resample(x, y)
    # Inverts the ML-MC transformation to recreate the ML set
    #y_resampled = lp.inverse_transform(y_resampled)
    #y_resampled = y_resampled.toarray()
    return X_resampled, y_resampled

X_spatial = build_spatial_matrix(data, SPATIAL_FEATURES) if USE_SPATIAL_INPUT else None

def make_split_indices(frame, labels, test_size, random_state, source_indices=None):
    if source_indices is None:
        source_indices = np.arange(len(frame))
    source_indices = np.asarray(source_indices)
    source_frame = frame.iloc[source_indices].reset_index(drop=True)
    source_labels = labels[source_indices]

    use_group_split = (
        SPLIT_STRATEGY == 'group'
        and SPLIT_GROUP_COLUMN
        and SPLIT_GROUP_COLUMN in source_frame.columns
        and source_frame[SPLIT_GROUP_COLUMN].nunique(dropna=False) >= 3
    )

    if use_group_split:
        groups = source_frame[SPLIT_GROUP_COLUMN].fillna('').astype(str).values
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_local, test_local = next(splitter.split(source_frame, source_labels, groups))
    else:
        label_counts = pd.Series(source_labels).value_counts()
        stratify = source_labels if len(label_counts) > 1 and label_counts.min() >= 2 else None
        train_local, test_local = train_test_split(
            np.arange(len(source_indices)),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    return source_indices[train_local], source_indices[test_local], 'group' if use_group_split else 'row'


train_full_idx, test_idx, split_used = make_split_indices(data, y, 0.20, RANDOM_SEED)
train_idx, val_idx, val_split_used = make_split_indices(data, y, 0.10, RANDOM_SEED, train_full_idx)

if split_used == 'group':
    train_groups = set(data.iloc[train_idx][SPLIT_GROUP_COLUMN].astype(str))
    val_groups = set(data.iloc[val_idx][SPLIT_GROUP_COLUMN].astype(str))
    test_groups = set(data.iloc[test_idx][SPLIT_GROUP_COLUMN].astype(str))
    print(f'-split strategy: group by {SPLIT_GROUP_COLUMN}')
    print(f'-group overlap train/test: {len(train_groups & test_groups)}')
    print(f'-group overlap train/val: {len(train_groups & val_groups)}')
else:
    print('-split strategy: row')

X_osm_train, X_osm_val, X_osm_test = X_osm[train_idx], X_osm[val_idx], X_osm[test_idx]
X_wiki_train, X_wiki_val, X_wiki_test = X_wiki[train_idx], X_wiki[val_idx], X_wiki[test_idx]
y_osm_train, y_osm_val, y_osm_test = y[train_idx], y[val_idx], y[test_idx]

if USE_SPATIAL_INPUT:
    X_spatial_train, X_spatial_val, X_spatial_test = X_spatial[train_idx], X_spatial[val_idx], X_spatial[test_idx]
    spatial_scaler = fit_spatial_scaler(X_spatial_train)
    X_spatial_train = transform_spatial_matrix(X_spatial_train, spatial_scaler)
    X_spatial_val = transform_spatial_matrix(X_spatial_val, spatial_scaler)
    X_spatial_test = transform_spatial_matrix(X_spatial_test, spatial_scaler)
else:
    spatial_scaler = None


# Note y_bal will be the same due to seeding
#x_osm_bal, y_bal = balance(X_osm_train, y_osm_train)
#x_wiki_bal, y_bal = balance(X_wiki_train, y_wiki_train)


# ------- osm tags path ------------
sentence_input = tf.keras.layers.Input(shape=(max_length,))
x = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[weight_matrix],
                              input_length=max_length)(sentence_input)
lstm = Bidirectional(LSTM(64, return_sequences = True), name="bi_lstm_0")(x)


#-----------Wikidata properties path  --------------------
sentence_inputWiki = tf.keras.layers.Input(shape=(max_lengthWiki,))
xwiki = tf.keras.layers.Embedding(vocab_sizeWiki, embedding_dim, weights=[weight_matrixWiki],
                                  input_length=max_lengthWiki)(sentence_inputWiki)


lstmwiki = Bidirectional(LSTM(64, return_sequences=True), name="bi_lstm_0wiki")(xwiki)


cross_att = CrossAttention(output_dim=64)([lstm, lstmwiki])

self_att1 = MultiHeadAttention(num_heads=2, key_dim=32)(cross_att, cross_att)
self_att1 = LayerNormalization()(self_att1)
self_att1 = Concatenate()([cross_att, self_att1])

lstm3 = Bidirectional(LSTM(32))(self_att1)

cross_att2 = CrossAttention(output_dim=64)([lstmwiki, lstm])

self_att2 = MultiHeadAttention(num_heads=2, key_dim=32)(cross_att2, cross_att2)
self_att2 = LayerNormalization()(self_att2)
self_att2 = Concatenate()([cross_att2, self_att2])

lstm4 = Bidirectional(LSTM(32))(self_att2)


# ------- combine ----------
model_inputs = [sentence_input, sentence_inputWiki]
concat_inputs = [lstm3, lstm4]
if USE_SPATIAL_INPUT:
    sentence_input_spatial = tf.keras.layers.Input(shape=(len(SPATIAL_FEATURES),), name="spatial_features")
    model_inputs.append(sentence_input_spatial)
    if SPATIAL_FEATURES == ["dist"]:
        spatial_encoded = Dense(1, activation="relu", name="distance_scalar")(sentence_input_spatial)
    else:
        spatial_encoded = sentence_input_spatial
        for index, units in enumerate(SPATIAL_ENCODER_UNITS, start=1):
            spatial_encoded = Dense(units, activation="relu", name=f"spatial_encoder_{index}")(spatial_encoded)
    concat_inputs.append(spatial_encoded)

concat = tf.keras.layers.concatenate(concat_inputs)

concat = Dense(50, activation="relu")(concat)
#dropout = Dropout(0.05)(dense1)
output = Dense(1, activation="sigmoid")(concat)

model = tf.keras.Model(inputs=model_inputs, outputs=output)


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


METRICS = [keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall')]

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])
model.summary()
callbacks = []
if EARLY_STOPPING_PATIENCE > 0:
    callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True))

class_weight = None
if USE_CLASS_WEIGHT:
    train_counts = pd.Series(y_osm_train).value_counts()
    if 0.0 in train_counts and 1.0 in train_counts:
        total = float(train_counts.sum())
        class_weight = {
            0: total / (2.0 * float(train_counts[0.0])),
            1: total / (2.0 * float(train_counts[1.0])),
        }
        print(f'-class weights: {class_weight}')

train_inputs = [X_osm_train, X_wiki_train]
val_inputs = [X_osm_val, X_wiki_val]
test_inputs = [X_osm_test, X_wiki_test]
if USE_SPATIAL_INPUT:
    train_inputs.append(X_spatial_train)
    val_inputs.append(X_spatial_val)
    test_inputs.append(X_spatial_test)

model.fit(
    train_inputs,
    y_osm_train,
    validation_data=(val_inputs, y_osm_val),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    shuffle=True,
    verbose=TRAIN_VERBOSE,
    callbacks=callbacks,
    class_weight=class_weight,
)

#add confusion matrix

probabilities = model.predict(test_inputs, batch_size=BATCH_SIZE).reshape(-1)
prediction = (probabilities >= PREDICTION_THRESHOLD)
runtime = time.time() - starttime

test_frame = data.iloc[test_idx].reset_index(drop=True).copy()
if 'osm_uid' not in test_frame.columns and 'osm_id' in test_frame.columns:
    test_frame['osm_uid'] = test_frame['osm_id'].astype(str)
test_predictions = pd.DataFrame({
    'wkid': test_frame['wkid'] if 'wkid' in test_frame.columns else '',
    'osm_uid': test_frame['osm_uid'] if 'osm_uid' in test_frame.columns else '',
    'osm_id': test_frame['osm_id'] if 'osm_id' in test_frame.columns else '',
    'match': y_osm_test.astype(bool),
    'probability': probabilities,
    'prediction': prediction.astype(bool),
})
for column in ['dist', 'bearing_sin', 'bearing_cos', 'd_lat', 'd_lon']:
    test_predictions[column] = (
        pd.to_numeric(test_frame[column], errors='coerce').fillna(0.0)
        if column in test_frame.columns
        else 0.0
    )
test_predictions.to_csv(
    os.path.join(DATA_DIR, TEST_PREDICTIONS_FILENAME),
    sep='\t',
    index=False,
)

with open(os.path.join(DATA_DIR, 'class_report.txt'), 'w', encoding='utf-8') as file:
    report = metrics.classification_report(y_osm_test, prediction, zero_division=0)
    file.write(f'Performance Attention Model:\n')
    file.write(f'Experiment: {get_experiment_name(config)}\n')
    file.write(f'On Dataset: {DATASET_PATH}\n')
    file.write(f"Spatial features: {', '.join(SPATIAL_FEATURES) if USE_SPATIAL_INPUT else 'none'}\n")
    file.write(f"Split strategy: {split_used}")
    if split_used == 'group':
        file.write(f" by {SPLIT_GROUP_COLUMN}")
    file.write("\n")
    file.write(f"runtime: {time.strftime('%H:%M:%S', time.gmtime(runtime))}\n")
    file.write(f"for {NUM_EPOCHS} epochs\n\n")
    file.write(report)
    print(report)

# save model for possible later reuse
def save_object(model, name: str):
    FILENAME = os.path.join(DATA_DIR, f'{name}.sav')
    with open(FILENAME, 'wb') as file:
        pickle.dump(model, file)

write_experiment_metadata(
    DATA_DIR,
    config,
    SPATIAL_FEATURES,
    extra={
        "prediction_threshold": PREDICTION_THRESHOLD,
        "epochs": NUM_EPOCHS,
        "random_seed": RANDOM_SEED,
        "attention_dimension": DIM_ATTENTION,
        "linear_dimension": DIM_LINEAR,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "max_vocabulary_size": MAX_VOCABULARY_SIZE,
        "effective_sequence_length": nWords,
        "batch_size": BATCH_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "use_class_weight": USE_CLASS_WEIGHT,
        "split_strategy": split_used,
        "split_group_column": SPLIT_GROUP_COLUMN if split_used == 'group' else "",
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "test_positive_support": int(np.sum(y_osm_test == 1.0)),
        "test_predictions_filename": TEST_PREDICTIONS_FILENAME,
        "spatial_scaler": "standard_scaler_fit_on_train_split" if USE_SPATIAL_INPUT else "none",
        "spatial_distance_transform": "log1p" if "dist" in SPATIAL_FEATURES else "none",
    },
)

model.save(os.path.join(DATA_DIR, 'keras model'))
save_object(tokenizer, 'osm tokenizer')
save_object(tokenizerWiki, 'wikidata tokenizer')
if USE_SPATIAL_INPUT:
    save_object(spatial_scaler, os.path.splitext(SPATIAL_SCALER_FILENAME)[0])
