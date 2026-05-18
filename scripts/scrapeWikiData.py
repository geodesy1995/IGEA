from SPARQLWrapper import SPARQLWrapper
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import configparser
import asyncio
import asyncpg
import re
import csv
import time
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)


OUTPUT_PATH = DATA_DIR + 'wikidata dump.parquet'
CLASSFILE_PATH = DATA_DIR + 'wikidata classes.txt'
COVERAGE_PATH = DATA_DIR + 'coverage_report.csv'
TESTRUN = config.getboolean('misc', 'testrun')
LIMIT = config.getint('misc', 'limit')
COUNTRY_ID = config.get('wikidata scrape', 'country_id')
SCRAPE_MODES = [s.strip() for s in config.get('wikidata scrape', 'scrape_values').split(',')]
NAME_LANGUAGE = config.get('wikidata scrape', 'name_language')
ENTITY_SOURCE = config.get('wikidata scrape', 'entity_source', fallback='country')
PW_FILENAME = config.get('postGIS', 'passwordfile', fallback='./config/pw.txt')
PG_HOST = config.get('postGIS', 'host', fallback='localhost')
PG_USER = config.get('postGIS', 'user', fallback='user')
PG_DB_NAME = config.get('postGIS', 'dbname', fallback='db')
PG_PORT = config.getint('postGIS', 'port', fallback=5432)
BASE_TABLE = config.get('entity linking', 'base_table', fallback='')
COORDINATE_SOURCE = config.get('wikidata scrape', 'coordinate_source', fallback='api' if ENTITY_SOURCE == 'osm_linked' else 'sparql').lower()
API_BATCH_SIZE = config.getint('wikidata scrape', 'api_batch_size', fallback=50)
SPARQL_BATCH_SIZE = config.getint('wikidata scrape', 'batch_size', fallback=500)
PROPERTY_BATCH_SIZE = config.getint('wikidata scrape', 'property_batch_size', fallback=250)
REQUEST_SLEEP_SECONDS = config.getfloat('wikidata scrape', 'request_sleep_seconds', fallback=0.75)
MAX_RETRIES = config.getint('wikidata scrape', 'max_retries', fallback=4)
RETRY_BACKOFF_SECONDS = config.getfloat('wikidata scrape', 'retry_backoff_seconds', fallback=10.0)
USER_AGENT = config.get('wikidata scrape', 'user_agent', fallback='IGEA-UrbanAI/0.1 (https://github.com/geodesy1995/IGEA; research experiment)')

sparql = SPARQLWrapper("https://query.wikidata.org/sparql",
                       returnFormat='json',
                       agent=USER_AGENT)
sparql.setTimeout(config.getint('wikidata scrape', 'timeout_seconds', fallback=120))

endpoint_error_count = 0
last_request_time = 0.0
coverage_metrics = {
    'entity_source': ENTITY_SOURCE,
    'coordinate_source': COORDINATE_SOURCE,
    'scrape_values': ','.join(SCRAPE_MODES),
    'osm_linked_qids': 0,
    'wikidata_coordinate_entities': 0,
    'endpoint_error_count': 0,
    'sparql_batch_size': SPARQL_BATCH_SIZE,
    'api_batch_size': API_BATCH_SIZE,
    'request_sleep_seconds': REQUEST_SLEEP_SECONDS,
}

linked_classes = []
if ENTITY_SOURCE != 'osm_linked':
    print('reading classes')
    print(f'from: {CLASSFILE_PATH}')
    with open(CLASSFILE_PATH, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            text = line.strip().replace('\n', '')
            if text:
                linked_classes.append(text)

def entity_batches(entity_dict: dict, step_size: int):
    keys = list(entity_dict.keys())
    i = 0
    while i < len(keys):
        batch = keys[i:min(i + step_size, len(keys))]
        yield i, batch
        i += len(batch)


def record_endpoint_error(context: str, exc: Exception) -> None:
    global endpoint_error_count
    endpoint_error_count += 1
    print(f'An error occurred {context}: {repr(exc)}')


def throttle_request() -> None:
    global last_request_time
    elapsed = time.monotonic() - last_request_time
    if elapsed < REQUEST_SLEEP_SECONDS:
        time.sleep(REQUEST_SLEEP_SECONDS - elapsed)
    last_request_time = time.monotonic()


def sparql_query(query: str, context: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            throttle_request()
            sparql.setQuery(query)
            return sparql.query().convert()
        except Exception as exc:
            if attempt < MAX_RETRIES:
                sleep_for = RETRY_BACKOFF_SECONDS * attempt
                print(f'Retrying {context} after {repr(exc)} in {sleep_for:.1f}s')
                time.sleep(sleep_for)
            else:
                record_endpoint_error(context, exc)
    return None


def api_query(params: dict, context: str):
    endpoint = 'https://www.wikidata.org/w/api.php'
    params = {
        **params,
        'format': 'json',
        'formatversion': '2',
    }
    url = endpoint + '?' + urlencode(params)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            throttle_request()
            request = Request(url, headers={'User-Agent': USER_AGENT})
            with urlopen(request, timeout=config.getint('wikidata scrape', 'timeout_seconds', fallback=120)) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as exc:
            if attempt < MAX_RETRIES:
                sleep_for = RETRY_BACKOFF_SECONDS * attempt
                print(f'Retrying {context} after {repr(exc)} in {sleep_for:.1f}s')
                time.sleep(sleep_for)
            else:
                record_endpoint_error(context, exc)
    return None


async def collect_osm_wikidata_qids() -> list:
    with open(PW_FILENAME, 'r', encoding='utf-8') as file:
        password = file.read().strip()

    conn = await asyncpg.connect(
        user=PG_USER,
        password=password,
        database=PG_DB_NAME,
        host=PG_HOST,
        port=PG_PORT,
    )
    rows = await conn.fetch(
        f"""
        SELECT DISTINCT tags -> 'wikidata' AS qid
        FROM {BASE_TABLE}
        WHERE tags -> 'wikidata' IS NOT NULL
        """
    )
    await conn.close()

    qid_pattern = re.compile(r'^Q[0-9]+$')
    qids = sorted({str(row['qid']).strip().replace('"', '') for row in rows if row['qid']})
    return [qid for qid in qids if qid_pattern.match(qid)]


def collect_osm_linked_entities() -> dict:
    qids = asyncio.run(collect_osm_wikidata_qids())
    if TESTRUN:
        qids = qids[:LIMIT]
    coverage_metrics['osm_linked_qids'] = len(qids)
    print(f'-valid OSM-linked qids: {len(qids)}')

    if COORDINATE_SOURCE == 'api':
        return collect_osm_linked_entities_from_api(qids)

    query = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?item ?location (SAMPLE(?type) AS ?type) WHERE {
    VALUES ?item {%s}
    ?item wdt:P625 ?location.
    OPTIONAL { ?item wdt:P31 ?type. }
    FILTER (strstarts(str(?location), 'Point'))
} GROUP BY ?item ?location
"""

    entities = {}
    step_size = SPARQL_BATCH_SIZE
    with tqdm(total=len(qids), desc='-Gathering OSM-linked Wikidata entities', miniters=1) as pbar:
        i = 0
        while i < len(qids):
            batch = qids[i:min(i + step_size, len(qids))]
            try:
                id_string = ''.join(f"wd:{qid} " for qid in batch)
                results = sparql_query(query % id_string, f'gathering linked qids {i} - {i + len(batch)}')
                if results is None:
                    pbar.update(len(batch))
                    i += len(batch)
                    continue
                for res in results['results']['bindings']:
                    wkid = res['item']['value'].split('/')[-1]
                    clazz = res.get('type', {}).get('value', '').split('/')[-1] or ''
                    entities[wkid] = {'wkid': wkid, 'location': res['location']['value'], 'type': clazz}
            except Exception as exc:
                record_endpoint_error(f'processing linked qids {i} - {i + len(batch)}', exc)
            pbar.update(len(batch))
            i += len(batch)
    return entities


def claim_value(entity: dict, pid: str):
    claims = entity.get('claims', {}).get(pid, [])
    if not claims:
        return None
    mainsnak = claims[0].get('mainsnak', {})
    datavalue = mainsnak.get('datavalue', {})
    return datavalue.get('value')


def collect_osm_linked_entities_from_api(qids: list) -> dict:
    entities = {}
    with tqdm(total=len(qids), desc='-Gathering OSM-linked Wikidata entities via API', miniters=1) as pbar:
        i = 0
        while i < len(qids):
            batch = qids[i:min(i + API_BATCH_SIZE, len(qids))]
            result = api_query(
                {
                    'action': 'wbgetentities',
                    'ids': '|'.join(batch),
                    'props': 'claims|labels',
                    'languages': NAME_LANGUAGE,
                },
                f'gathering linked qids via API {i} - {i + len(batch)}',
            )
            if result is None:
                pbar.update(len(batch))
                i += len(batch)
                continue
            for qid, entity in result.get('entities', {}).items():
                if entity.get('missing'):
                    continue
                coordinate = claim_value(entity, 'P625')
                if not coordinate:
                    continue
                lat = coordinate.get('latitude')
                lon = coordinate.get('longitude')
                if lat is None or lon is None:
                    continue
                type_value = claim_value(entity, 'P31') or {}
                type_qid = type_value.get('id', '') if isinstance(type_value, dict) else ''
                label = entity.get('labels', {}).get(NAME_LANGUAGE, {}).get('value', '')
                claim_props = sorted(entity.get('claims', {}).keys())
                entities[qid] = {
                    'wkid': qid,
                    'location': f'Point({lon} {lat})',
                    'type': type_qid,
                    'pop': len(claim_props),
                    'labels': type_qid,
                    'name': label,
                    'properties': f"label {label} type {type_qid} claims {' '.join(claim_props)}".strip(),
                }
            pbar.update(len(batch))
            i += len(batch)
    return entities


def collect_country_entities() -> dict:
    # collect geo entities for classes and initial data
    query = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?item ?type ?location WHERE {
    ?item wdt:P17 wd:%s.
    ?item wdt:P31* wd:%s. # instance or subinstance of (to be specified)
    ?item wdt:P625 ?location.
    FILTER (strstarts(str(?location), 'Point'))
}
"""

    entities = {}
    with tqdm(total=len(linked_classes), desc=f'-Gathering entities in ({COUNTRY_ID})', miniters=1) as pbar:
        for clazz in linked_classes:
            try:
                pbar.set_postfix_str(f'class: {clazz}')
                results = sparql_query(query % (COUNTRY_ID, clazz), f'gathering {clazz}')
                if results is None:
                    pbar.update(1)
                    continue
                for res in results['results']['bindings']:
                    wkid = res['item']['value'].split('/')[-1]
                    entities.update({wkid: {'wkid': wkid, 'location': res['location']['value'], 'type': clazz}})
            except Exception as exc:
                record_endpoint_error(f'processing {clazz}', exc)
            pbar.update(1)
            if TESTRUN:
                if len(entities) > LIMIT:
                    break
    if TESTRUN:
        entities = dict(list(entities.items())[:LIMIT])
    return entities

if ENTITY_SOURCE == 'osm_linked':
    print('-entity source: OSM-linked wikidata tags with Wikidata coordinates')
    entities = collect_osm_linked_entities()
else:
    print('-entity source: Wikidata country/class query')
    entities = collect_country_entities()
print(f'-wikidata entities with coordinates: {len(entities)}')

# update popularity
RUN_SPARQL_ENRICHMENT = not (ENTITY_SOURCE == 'osm_linked' and COORDINATE_SOURCE == 'api')

# update popularity
if RUN_SPARQL_ENRICHMENT and 'popularity' in SCRAPE_MODES:
    pop_query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    
    SELECT ?kgentity  (count(distinct ?p) as ?prop) {
      VALUES ?kgentity {%s}
      ?kgentity ?p ?statement .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    } group by ?kgentity
    """

    # set defaults
    for k, v in entities.items():
        v.update({'pop': 0})

    i = 0
    step_size = SPARQL_BATCH_SIZE
    with tqdm(total=len(entities), desc='-updating popularity') as pbar:
        for i, batch in entity_batches(entities, step_size):
            try:
                id_string = ''.join(f"wd:{e} " for e in batch)
                results = sparql_query(pop_query % id_string, f'updating popularity {i} - {i + len(batch)}')
                if results is not None:
                    for res in results['results']['bindings']:
                        entities[res['kgentity']['value'].split('/')[-1]].update({'pop': int(res['prop']['value'])})
            except Exception as exc:
                record_endpoint_error(f'processing popularity {i} - {i + len(batch)}', exc)
            pbar.update(len(batch))



# add type labels for entities
if RUN_SPARQL_ENRICHMENT and 'type labels' in SCRAPE_MODES:
    label_query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    
    SELECT ?kgentity (GROUP_CONCAT(distinct ?typeLabel; SEPARATOR = "; ") as ?labels) {
      VALUES ?kgentity {%s}
      ?kgentity wdt:P31 ?label.
      ?label rdfs:label ?typeLabel.
      FILTER(lang(?typeLabel) = 'en').
    } group by ?kgentity
    """

    # set defaults
    for k, v in entities.items():
        v.update({'labels': ''})

    i = 0
    step_size = SPARQL_BATCH_SIZE

    with tqdm(total=len(entities), desc='-updating labels', miniters=1) as pbar:
        for i, batch in entity_batches(entities, step_size):
            try:
                id_string = ''.join(f"wd:{e} " for e in batch)
                results = sparql_query(label_query % id_string, f'updating labels {i} - {i + len(batch)}')
                if results is not None:
                    for res in results['results']['bindings']:
                        entities[res['kgentity']['value'].split('/')[-1]].update({'labels': res['labels']['value']})
            except Exception as exc:
                record_endpoint_error(f'processing labels {i} - {i + len(batch)}', exc)
            pbar.update(len(batch))

# add names to entities
if RUN_SPARQL_ENRICHMENT and 'name' in SCRAPE_MODES:
    name_query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    
    SELECT ?kgentity ?kgentityLabel {
        VALUES ?kgentity {%s}
        SERVICE wikibase:label { bd:serviceParam wikibase:language '%s', 'en'}
    }
    """

    # set defaults
    for k, v in entities.items():
        v.update({'name': ''})

    i = 0
    step_size = SPARQL_BATCH_SIZE

    with tqdm(total=len(entities), desc='-updating names', miniters=1) as pbar:
        for i, batch in entity_batches(entities, step_size):
            try:
                id_string = ''.join(f"wd:{e} " for e in batch)
                results = sparql_query(name_query % (id_string, NAME_LANGUAGE), f'updating names {i} - {i + len(batch)}')
                if results is not None:
                    for res in results['results']['bindings']:
                        entities[res['kgentity']['value'].split('/')[-1]].update({'name': res['kgentityLabel']['value']})
            except Exception as exc:
                record_endpoint_error(f'processing names {i} - {i + len(batch)}', exc)
            pbar.update(len(batch))


# add full properties per entity
if RUN_SPARQL_ENRICHMENT and 'full properties' in SCRAPE_MODES:
    property_query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    
    SELECT ?kgentity ?kgentityLabel ?wdLabel ?ps_Label {
        VALUES ?kgentity {%s}
        ?kgentity ?p ?statement .
        ?statement ?ps ?ps_ .
  
        ?wd wikibase:claim ?p.
        ?wd wikibase:statementProperty ?ps.
  
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    } ORDER BY ?kgentity
    """

    # set defaults
    for k, v in entities.items():
        v.update({'properties': ''})

    i = 0
    step_size = PROPERTY_BATCH_SIZE

    with tqdm(total=len(entities), desc='-updating properties', miniters=1) as pbar:
        for i, batch in entity_batches(entities, step_size):
            try:
                id_string = ''.join(f"wd:{e} " for e in batch)
                results = sparql_query(property_query % id_string, f'gathering properties {i} - {i + len(batch)}')
                if results is None:
                    pbar.update(len(batch))
                    continue
                cur_id = ''
                property_pairs = set()
                for res in results['results']['bindings']:
                    wkid = res['kgentity']['value'].split('/')[-1]
                    if cur_id: # skip first test
                        if cur_id != wkid:
                            entities[cur_id].update({'properties': ' '.join(property_pairs)})
                            property_pairs = set()
                            property_pairs.add(f"{'label'} {res['kgentityLabel']['value']}")
                    else:
                        property_pairs.add(f"{'label'} {res['kgentityLabel']['value']}")
                    property_pairs.add(f"{res['wdLabel']['value']} {res['ps_Label']['value']}")
                    cur_id = wkid
                if cur_id:
                    entities[cur_id].update({'properties': ' '.join(property_pairs)})
            except Exception as exc:
                record_endpoint_error(f'processing properties {i} - {i + len(batch)}', exc)
            pbar.update(len(batch))


def ensure_entity_defaults(entity_dict: dict) -> None:
    for wkid, entity in entity_dict.items():
        entity.setdefault('wkid', wkid)
        entity.setdefault('location', '')
        entity.setdefault('type', '')
        entity.setdefault('pop', 0)
        entity.setdefault('labels', '')
        entity.setdefault('name', '')
        if not entity.get('properties'):
            parts = [
                f"label {entity.get('name', '')}".strip(),
                f"type {entity.get('type', '')}".strip(),
                f"labels {entity.get('labels', '')}".strip(),
            ]
            entity['properties'] = ' '.join(part for part in parts if len(part.split()) > 1)


def write_classes(entity_dict: dict) -> None:
    classes = sorted({entity.get('type', '') for entity in entity_dict.values() if entity.get('type')})
    with open(CLASSFILE_PATH, 'w', encoding='utf-8', newline='') as file:
        file.write('\n'.join(classes))


def write_coverage_report() -> None:
    coverage_metrics['wikidata_coordinate_entities'] = len(entities)
    coverage_metrics['endpoint_error_count'] = endpoint_error_count
    with open(COVERAGE_PATH, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['metric', 'value'])
        writer.writeheader()
        for metric, value in coverage_metrics.items():
            writer.writerow({'metric': metric, 'value': value})

def write_to_file(entity_dict: dict, filename: str) -> None:
    """
    function to transform entity dictionary into dataframe for saving in parquet format
    :param entity_dict: dictionary containing wikidata information
    :param filename: path to write file to
    :return:
    """
    ensure_entity_defaults(entity_dict)
    data_list = list(entity_dict.values())
    table = pa.Table.from_pylist(data_list)
    with open(filename, 'wb') as file:
        pq.write_table(table, file)

if not entities:
    write_classes(entities)
    write_coverage_report()
    raise RuntimeError('No Wikidata entities with coordinates were collected; check WDQS rate limits or connectivity.')

write_classes(entities)
write_coverage_report()
print('-writing scraped data')
print(f'-to: {OUTPUT_PATH}')
write_to_file(entities, OUTPUT_PATH)
print('-writing complete')
