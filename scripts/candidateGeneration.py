import os
import csv
import sys
import time
import configparser
import pandas as pd
import asyncio
import asyncpg
from tqdm import tqdm
from json import dumps, loads
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from dbpedia_utils import normalize_wikipedia_tag
from experiment_config import TRAIN_PAIRS_FILENAME, UNMATCHED_PAIRS_FILENAME

DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

PW_FILENAME = config.get('postGIS', 'passwordfile')
PG_HOST = config.get('postGIS', 'host')
PG_USER = config.get('postGIS', 'user')
PG_DB_NAME = config.get('postGIS', 'dbname')
PG_PORT = config.getint('postGIS', 'port')
VIEW_NAME = config.get('entity linking', 'view_name')
DATA_SOURCE = config.get('meta', 'kg_source')
DBPEDIA_SOURCE = config.get('dbpedia scrape', 'dbpedia_source', fallback='en')
MAX_CANDIDATES = config.getint('candidate generation', 'max_candidates')
DIST_THRESHOLD = config.getint('candidate generation', 'dist_threshold')
MAX_CANDIDATE_AREA_M2 = config.getfloat('candidate generation', 'max_candidate_area_m2', fallback=50_000_000.0)
MAX_TRAIN_FALSE_PER_ENTITY = config.getint('candidate generation', 'max_train_false_per_entity', fallback=20)
QUERY_TIMEOUT_MS = config.getint('candidate generation', 'query_timeout_ms', fallback=120_000)
GENERATION_METHOD = config.get('candidate generation', 'method')
USE_LEGACY_EMBEDDINGS = config.getboolean('legacy', 'use_legacy_embeddings')
LOG_FILENAME = os.path.join(DATA_DIR, 'generate_candidates_log.txt')
AUDIT_FILENAME = os.path.join(DATA_DIR, 'candidate_generation_audit.csv')
MATCH_FILENAME = os.path.join(DATA_DIR, TRAIN_PAIRS_FILENAME)
NO_MATCH_FILENAME = os.path.join(DATA_DIR, UNMATCHED_PAIRS_FILENAME)
DATA_PATH = os.path.join(DATA_DIR, 'wikidata dump.parquet')
TESTRUN = config.getboolean('misc', 'testrun')
LIMIT = config.getint('misc', 'limit')

with open(PW_FILENAME, 'r') as file:
    password = file.read().strip()

wiki_data = pd.read_parquet(DATA_PATH, engine='pyarrow')

if TESTRUN:
    print(f'Restricting candidate generation to {LIMIT} entities')
    wiki_data = wiki_data[:LIMIT]


audit_metrics = {
    'dist_threshold': DIST_THRESHOLD,
    'max_candidates': MAX_CANDIDATES,
    'distance_metric': 'geography_meters',
    'gold_total': 0,
    'gold_within_2500m': 0,
    'gold_outside_2500m': 0,
    'gold_within_threshold': 0,
    'gold_outside_threshold': 0,
    'gold_missing_after_candidate_generation': 0,
    'dropped_by_limit_count': 0,
}
gold_distances = []


def normalize_json_object(data):
    if isinstance(data, str):
        return loads(data)
    return data


def filter_tags_concat(data: dict) -> str:
    data = normalize_json_object(data)
    tag = []
    excluded_top_level = {
        'way',
        'osm_id',
        'osm_uid',
        'osm_type',
        'tags',
        'wkid',
        'confidence',
        'iteration',
    }
    for k, v in data.items():
        if k not in excluded_top_level:
            tag.extend([k, v])
        elif k == 'tags':
            if v:
                for k_tag, v_tag in v.items():
                    if k_tag not in ['wikidata', 'wikipedia'] and not k_tag.startswith('osm_'):
                        tag.extend([k_tag, v_tag])
    return ' '.join(tag).replace('\n', ' ')


def filter_tags_json(data: dict) -> str:
    data = normalize_json_object(data)
    tags_dict = {}
    excluded_top_level = {
        'way',
        'osm_id',
        'osm_uid',
        'osm_type',
        'tags',
        'wkid',
        'confidence',
        'iteration',
    }
    for k, v in data.items():
        if k not in excluded_top_level:
            tags_dict.update({k: v})
        elif k == 'tags':
            if v:
                for k_tag, v_tag in v.items():
                    if k_tag not in ['wikidata', 'wikipedia'] and not k_tag.startswith('osm_'):
                        tags_dict.update({k_tag: v_tag})
    return dumps(tags_dict)


def normalize_linked_id(value):
    if value is None:
        return None
    if DATA_SOURCE == 'dbpedia':
        normalized, _reason = normalize_wikipedia_tag(value, DBPEDIA_SOURCE)
        return normalized
    return str(value).strip().strip('"')


def dbpedia_direct_link_sql(parameter_index: int = 6) -> str:
    source = DBPEDIA_SOURCE.replace("'", "''")
    return f"""
                     REPLACE(
                         regexp_replace(
                             regexp_replace(COALESCE(g.wkid, ''), '^https?://{source}\\.wikipedia\\.org/wiki/', '', 'i'),
                             '^{source}:',
                             '',
                             'i'
                         ),
                         ' ',
                         '_'
                     ) = ${parameter_index}"""


def direct_link_sql(parameter_index: int = 6) -> str:
    if DATA_SOURCE == 'dbpedia':
        return dbpedia_direct_link_sql(parameter_index)
    return f"COALESCE(g.wkid, '') = ${parameter_index}"


def direct_link_exact_sql(parameter_index: int) -> str:
    return f"COALESCE(g.wkid, '') = ANY(${parameter_index}::text[])"


def linked_value_variants(wiki_id: str) -> list:
    if DATA_SOURCE != 'dbpedia':
        return [wiki_id]
    title = str(wiki_id)
    title_space = title.replace('_', ' ')
    encoded_title = quote(title, safe='()_,-.')
    encoded_space = quote(title_space, safe='()_,-.')
    source = DBPEDIA_SOURCE
    return sorted({
        title,
        title_space,
        f'{source}:{title}',
        f'{source}:{title_space}',
        f'http://{source}.wikipedia.org/wiki/{title}',
        f'https://{source}.wikipedia.org/wiki/{title}',
        f'http://{source}.wikipedia.org/wiki/{encoded_title}',
        f'https://{source}.wikipedia.org/wiki/{encoded_title}',
        f'http://{source}.wikipedia.org/wiki/{encoded_space}',
        f'https://{source}.wikipedia.org/wiki/{encoded_space}',
    })


def record_key(record) -> str:
    osm_uid = record['osm_uid']
    if osm_uid is not None:
        return str(osm_uid)
    return f"{record['osm_type']}:{record['osm_id']}"


def point_row(wiki_id, record, match, data, tag_filter):
    return [
        wiki_id,
        record['osm_uid'],
        record['osm_id'],
        match,
        record['dist'],
        record['bearing_sin'],
        record['bearing_cos'],
        record['d_lat'],
        record['d_lon'],
        record['bbox_overlap'],
        tag_filter(record['jsonb_strip_nulls']),
    ] + data


def merge_direct_gold_rows(wiki_id, distance_records, direct_records, data, tag_filter):
    rows_by_key = {}
    distance_keys = set()
    is_linked = False

    for record in distance_records:
        key = record_key(record)
        distance_keys.add(key)
        match = normalize_linked_id(record['wkid']) == wiki_id
        if match:
            is_linked = True
        rows_by_key[key] = point_row(wiki_id, record, match, data, tag_filter)

    for record in direct_records:
        key = record_key(record)
        match = normalize_linked_id(record['wkid']) == wiki_id
        if match:
            is_linked = True
        rows_by_key[key] = point_row(wiki_id, record, match, data, tag_filter)

    direct_keys = {record_key(record) for record in direct_records}
    audit_metrics['gold_total'] += len(direct_records)
    for record in direct_records:
        key = record_key(record)
        dist = float(record['dist']) if record['dist'] is not None else 0.0
        gold_distances.append(dist)
        if bool(record['direct_within_2500']):
            audit_metrics['gold_within_2500m'] += 1
        else:
            audit_metrics['gold_outside_2500m'] += 1
        if bool(record['direct_within_threshold']):
            audit_metrics['gold_within_threshold'] += 1
            if key not in distance_keys:
                audit_metrics['dropped_by_limit_count'] += 1
        else:
            audit_metrics['gold_outside_threshold'] += 1
        if key not in rows_by_key:
            audit_metrics['gold_missing_after_candidate_generation'] += 1

    # A direct row should always be present after the merge. Keep this explicit
    # so audit output catches future refactors that break gold preservation.
    missing_after_merge = direct_keys - set(rows_by_key.keys())
    audit_metrics['gold_missing_after_candidate_generation'] += len(missing_after_merge)
    return is_linked, list(rows_by_key.values())


def write_candidate_generation_audit():
    distances = sorted(gold_distances)
    metrics = dict(audit_metrics)
    if distances:
        series = pd.Series(distances)
        metrics.update({
            'gold_distance_min': float(series.min()),
            'gold_distance_median': float(series.median()),
            'gold_distance_p95': float(series.quantile(0.95)),
            'gold_distance_max': float(series.max()),
        })
    else:
        metrics.update({
            'gold_distance_min': 0.0,
            'gold_distance_median': 0.0,
            'gold_distance_p95': 0.0,
            'gold_distance_max': 0.0,
        })

    with open(AUDIT_FILENAME, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def log_candidate_skip(wiki_id, reason):
    with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
        file.write(f'Skipped candidate query for {wiki_id}: {reason}\n')


def enqueue_results(is_linked, res, pair_queue, single_queue):
    if is_linked:
        if MAX_TRAIN_FALSE_PER_ENTITY >= 0:
            true_rows = [entry for entry in res if bool(entry[3])]
            false_rows = [entry for entry in res if not bool(entry[3])][:MAX_TRAIN_FALSE_PER_ENTITY]
            res = true_rows + false_rows
        for entry in res:
            pair_queue.put(entry)
    else:
        for entry in res:
            single_queue.put(entry)


async def table_has_column(conn, table_name: str, column_name: str) -> bool:
    return await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = $1
              AND column_name = $2
        )
        """,
        table_name,
        column_name,
    )


def consume(stop, queue, filename) -> None:
    with open(filename, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        while True:
            if not queue.empty():
                i = queue.get()
                writer.writerow(i)
            elif stop():
                print('- Stopping file writing thread')
                return


async def fetch_candidates_for_point(pool, wiki_id, location, data, pair_queue, single_queue=None, threshold=2500, limit=100, osm_uid_expr="osm_id::text", osm_type_expr="'N'"):
    direct_link_expr = direct_link_sql(6)
    sql = f"""SELECT osm_id,
                     {osm_uid_expr} AS osm_uid,
                     {osm_type_expr} AS osm_type,
                     ST_Distance(ST_Transform(way, 4326)::geography, ST_GeomFromEWKT($1)::geography) dist,
                     COALESCE(sin(ST_Azimuth(ST_GeomFromEWKT($1)::geography, ST_Transform(ST_Centroid(way), 4326)::geography)), 0) bearing_sin,
                     COALESCE(cos(ST_Azimuth(ST_GeomFromEWKT($1)::geography, ST_Transform(ST_Centroid(way), 4326)::geography)), 0) bearing_cos,
                     (ST_Y(ST_Transform(ST_Centroid(way), 4326)) - ST_Y(ST_GeomFromEWKT($1))) d_lat,
                     (ST_X(ST_Transform(ST_Centroid(way), 4326)) - ST_X(ST_GeomFromEWKT($1))) d_lon,
                     (ST_Intersects(way, ST_Transform(ST_GeomFromEWKT($1), 3857)))::int bbox_overlap,
                     jsonb_strip_nulls(to_jsonb(g)), wkid
              FROM {VIEW_NAME} g
              WHERE way && ST_Expand(ST_Transform(ST_GeomFromEWKT($2), 3857), $3::double precision * 2.5)
                AND ST_DWithin(ST_Transform(way, 4326)::geography, ST_GeomFromEWKT($2)::geography, $3)
                AND NOT ST_IsEmpty(way)
                AND (
                    $5 <= 0
                    OR COALESCE(ST_Area(way), 0) <= $5
                    OR {direct_link_expr}
                )
              ORDER BY dist ASC LIMIT $4"""
    direct_sql = f"""SELECT osm_id,
                     {osm_uid_expr} AS osm_uid,
                     {osm_type_expr} AS osm_type,
                     ST_Distance(ST_Transform(way, 4326)::geography, ST_GeomFromEWKT($1)::geography) dist,
                     COALESCE(sin(ST_Azimuth(ST_GeomFromEWKT($1)::geography, ST_Transform(ST_Centroid(way), 4326)::geography)), 0) bearing_sin,
                     COALESCE(cos(ST_Azimuth(ST_GeomFromEWKT($1)::geography, ST_Transform(ST_Centroid(way), 4326)::geography)), 0) bearing_cos,
                     (ST_Y(ST_Transform(ST_Centroid(way), 4326)) - ST_Y(ST_GeomFromEWKT($1))) d_lat,
                     (ST_X(ST_Transform(ST_Centroid(way), 4326)) - ST_X(ST_GeomFromEWKT($1))) d_lon,
                     (ST_Intersects(way, ST_Transform(ST_GeomFromEWKT($1), 3857)))::int bbox_overlap,
                     ST_DWithin(ST_Transform(way, 4326)::geography, ST_GeomFromEWKT($2)::geography, $3) direct_within_threshold,
                     ST_DWithin(ST_Transform(way, 4326)::geography, ST_GeomFromEWKT($2)::geography, 2500) direct_within_2500,
                     jsonb_strip_nulls(to_jsonb(g)), wkid
              FROM {VIEW_NAME} g
              WHERE NOT ST_IsEmpty(way)
                AND ({direct_link_sql(4)} OR {direct_link_exact_sql(5)})
              ORDER BY dist ASC"""

    distance_records = []
    direct_records = []
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('statement_timeout', $1, true)", f'{QUERY_TIMEOUT_MS}ms')
                async for record in conn.cursor(sql, f'SRID=4326; {location}', f'SRID=4326; {location}', threshold, limit, MAX_CANDIDATE_AREA_M2, wiki_id):
                    distance_records.append(record)
                direct_records = await conn.fetch(direct_sql, f'SRID=4326; {location}', f'SRID=4326; {location}', threshold, wiki_id, linked_value_variants(wiki_id))
    except (asyncio.TimeoutError, asyncpg.exceptions.QueryCanceledError) as exc:
        log_candidate_skip(wiki_id, f'timeout after {QUERY_TIMEOUT_MS}ms ({type(exc).__name__})')
        return

    is_linked, res = merge_direct_gold_rows(wiki_id, distance_records, direct_records, data, filter_tags_concat)
    enqueue_results(is_linked, res, pair_queue, single_queue)


async def fetch_candidates_for_name(pool, wiki_id, name, data, pair_queue, single_queue=None, limit=100, osm_uid_expr="osm_id::text", osm_type_expr="'N'"):
    sql = f"""SELECT osm_id, {osm_uid_expr} AS osm_uid, {osm_type_expr} AS osm_type, (similarity(lower(g."name"), lower($1))) as sim, jsonb_strip_nulls(to_jsonb(g)), wkid
              FROM {VIEW_NAME} g
              WHERE g.name is not null
              ORDER BY sim DESC LIMIT $2"""

    is_linked = False
    res = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            async for record in conn.cursor(sql, name, limit):
                match = False
                id = normalize_linked_id(record['wkid'])
                if id == wiki_id:
                    is_linked = True
                    match = True
                res.append([wiki_id, record['osm_uid'], record['osm_id'], match, record['sim'], filter_tags_concat(record['jsonb_strip_nulls'])] + data)

    enqueue_results(is_linked, res, pair_queue, single_queue)


async def fetch_candidates_legacy(pool, wiki_id, location, data, pair_queue, single_queue=None, threshold=2500, limit=100, osm_uid_expr="osm_id::text", osm_type_expr="'N'"):
    direct_link_expr = direct_link_sql(6)
    sql = f"""SELECT osm_id,
                     {osm_uid_expr} AS osm_uid,
                     {osm_type_expr} AS osm_type,
                     ST_Distance(ST_Transform(way, 4326)::geography, ST_GeomFromEWKT($1)::geography) dist,
                     COALESCE(sin(ST_Azimuth(ST_GeomFromEWKT($1)::geography, ST_Transform(ST_Centroid(way), 4326)::geography)), 0) bearing_sin,
                     COALESCE(cos(ST_Azimuth(ST_GeomFromEWKT($1)::geography, ST_Transform(ST_Centroid(way), 4326)::geography)), 0) bearing_cos,
                     (ST_Y(ST_Transform(ST_Centroid(way), 4326)) - ST_Y(ST_GeomFromEWKT($1))) d_lat,
                     (ST_X(ST_Transform(ST_Centroid(way), 4326)) - ST_X(ST_GeomFromEWKT($1))) d_lon,
                     (ST_Intersects(way, ST_Transform(ST_GeomFromEWKT($1), 3857)))::int bbox_overlap,
                     jsonb_strip_nulls(to_jsonb(g)), wkid
              FROM {VIEW_NAME} g
              WHERE way && ST_Expand(ST_Transform(ST_GeomFromEWKT($2), 3857), $3::double precision * 2.5)
                AND ST_DWithin(ST_Transform(way, 4326)::geography, ST_GeomFromEWKT($2)::geography, $3)
                AND NOT ST_IsEmpty(way)
                AND (
                    $5 <= 0
                    OR COALESCE(ST_Area(way), 0) <= $5
                    OR {direct_link_expr}
                )
              ORDER BY dist ASC LIMIT $4"""
    direct_sql = f"""SELECT osm_id,
                     {osm_uid_expr} AS osm_uid,
                     {osm_type_expr} AS osm_type,
                     ST_Distance(ST_Transform(way, 4326)::geography, ST_GeomFromEWKT($1)::geography) dist,
                     COALESCE(sin(ST_Azimuth(ST_GeomFromEWKT($1)::geography, ST_Transform(ST_Centroid(way), 4326)::geography)), 0) bearing_sin,
                     COALESCE(cos(ST_Azimuth(ST_GeomFromEWKT($1)::geography, ST_Transform(ST_Centroid(way), 4326)::geography)), 0) bearing_cos,
                     (ST_Y(ST_Transform(ST_Centroid(way), 4326)) - ST_Y(ST_GeomFromEWKT($1))) d_lat,
                     (ST_X(ST_Transform(ST_Centroid(way), 4326)) - ST_X(ST_GeomFromEWKT($1))) d_lon,
                     (ST_Intersects(way, ST_Transform(ST_GeomFromEWKT($1), 3857)))::int bbox_overlap,
                     ST_DWithin(ST_Transform(way, 4326)::geography, ST_GeomFromEWKT($2)::geography, $3) direct_within_threshold,
                     ST_DWithin(ST_Transform(way, 4326)::geography, ST_GeomFromEWKT($2)::geography, 2500) direct_within_2500,
                     jsonb_strip_nulls(to_jsonb(g)), wkid
              FROM {VIEW_NAME} g
              WHERE NOT ST_IsEmpty(way)
                AND ({direct_link_sql(4)} OR {direct_link_exact_sql(5)})
              ORDER BY dist ASC"""

    distance_records = []
    direct_records = []
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT set_config('statement_timeout', $1, true)", f'{QUERY_TIMEOUT_MS}ms')
                async for record in conn.cursor(sql, f'SRID=4326; {location}', f'SRID=4326; {location}', threshold, limit, MAX_CANDIDATE_AREA_M2, wiki_id):
                    distance_records.append(record)
                direct_records = await conn.fetch(direct_sql, f'SRID=4326; {location}', f'SRID=4326; {location}', threshold, wiki_id, linked_value_variants(wiki_id))
    except (asyncio.TimeoutError, asyncpg.exceptions.QueryCanceledError) as exc:
        log_candidate_skip(wiki_id, f'timeout after {QUERY_TIMEOUT_MS}ms ({type(exc).__name__})')
        return

    is_linked, res = merge_direct_gold_rows(wiki_id, distance_records, direct_records, data, filter_tags_json)
    enqueue_results(is_linked, res, pair_queue, single_queue)


async def main():
    pool = await asyncpg.create_pool(
        user=PG_USER, 
        password=password, 
        database=PG_DB_NAME, 
        host=PG_HOST, 
        port=PG_PORT,
        min_size=1,
        max_size=10,
        command_timeout=max(30, QUERY_TIMEOUT_MS / 1000 + 10),
    )

    async with pool.acquire() as conn:
        has_osm_uid = await table_has_column(conn, VIEW_NAME, 'osm_uid')
        has_osm_type = await table_has_column(conn, VIEW_NAME, 'osm_type')
    osm_uid_expr = "g.osm_uid" if has_osm_uid else "g.osm_id::text"
    osm_type_expr = "g.osm_type" if has_osm_type else "'N'"

    with open(LOG_FILENAME, 'w', encoding='utf-8') as file:
        file.write('Starting candidate search\n')
        file.write(f'Matched entities written to: {MATCH_FILENAME}\n')
        file.write(f'Unmatched entities written to: {NO_MATCH_FILENAME}\n')
        file.write(f'max_candidate_area_m2={MAX_CANDIDATE_AREA_M2}\n')
        file.write(f'max_train_false_per_entity={MAX_TRAIN_FALSE_PER_ENTITY}\n')
        file.write(f'query_timeout_ms={QUERY_TIMEOUT_MS}\n')

    match_queue = Queue()
    single_queue = Queue()
    stop_threads = False

    start_time = time.time()

    print('- Starting file writing threads')

    with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
        file.write('Starting consumer threads\n')

    match_consumer = Thread(target=consume, daemon=True, args=(lambda: stop_threads, match_queue, MATCH_FILENAME))
    match_consumer.start()
    single_consumer = Thread(target=consume, daemon=True, args=(lambda: stop_threads, single_queue, NO_MATCH_FILENAME))
    single_consumer.start()

    with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
        file.write('Starting candidate generation threads\n')

    print('- Starting candidate generation')
    print(f'- Method used: {GENERATION_METHOD}')
    if USE_LEGACY_EMBEDDINGS:
        print('- Loading with JSON data for legacy embeddings')

    tasks = []

    if not USE_LEGACY_EMBEDDINGS:
        if GENERATION_METHOD == 'distance':
            col_mask = [c not in ['wkid', 'location'] for c in wiki_data.columns]
            header_names = ['wkid', 'osm_uid', 'osm_id', 'match', 'dist', 'bearing_sin', 'bearing_cos', 'd_lat', 'd_lon', 'bbox_overlap', 'tags'] + list(wiki_data.columns[col_mask])
            match_queue.put(header_names)
            single_queue.put(header_names)
            for index, row in wiki_data.iterrows():
                tasks.append(fetch_candidates_for_point(pool, row['wkid'], row['location'], list(row[col_mask]), match_queue, single_queue, DIST_THRESHOLD, MAX_CANDIDATES, osm_uid_expr, osm_type_expr))
        else:  # GENERATION_METHOD == 'name':
            col_mask = [c not in ['wkid', 'name'] for c in wiki_data.columns]
            header_names = ['wkid', 'osm_uid', 'osm_id', 'match', 'sim', 'tags'] + list(wiki_data.columns[col_mask])
            match_queue.put(header_names)
            single_queue.put(header_names)
            for index, row in wiki_data.iterrows():
                tasks.append(fetch_candidates_for_name(pool, row['wkid'], row['name'], list(row[col_mask]), match_queue, single_queue, MAX_CANDIDATES, osm_uid_expr, osm_type_expr))
    else:
        col_mask = [c not in ['wkid', 'location', 'name'] for c in wiki_data.columns]
        header_names = ['wkid', 'osm_uid', 'osm_id', 'match', 'dist', 'bearing_sin', 'bearing_cos', 'd_lat', 'd_lon', 'bbox_overlap', 'tags'] + list(wiki_data.columns[col_mask])
        match_queue.put(header_names)
        single_queue.put(header_names)
        for index, row in wiki_data.iterrows():
            tasks.append(fetch_candidates_legacy(pool, row['wkid'], row['location'], list(row[col_mask]), match_queue, single_queue, DIST_THRESHOLD, MAX_CANDIDATES, osm_uid_expr, osm_type_expr))

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='- Finding candidates'):
        await f

    with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
        file.write('Finished candidate generation threads\n')

    stop_threads = True
    match_consumer.join()
    single_consumer.join()
    write_candidate_generation_audit()

    with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
        file.write('Stopped consumer threads\n')

    await pool.close()

    with open(LOG_FILENAME, 'a', encoding='utf-8') as file:
        file.write('Closed connection\n')
        file.write(f"Execution ended successfully at {time.strftime('%d.%M.%Y %H:%M:%S', time.gmtime(time.time()))}\n")
        file.write(f"Execution time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}\n")


if __name__ == '__main__':
    asyncio.run(main())
