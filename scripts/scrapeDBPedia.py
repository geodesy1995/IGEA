import re
import csv
import asyncio
import asyncpg
from SPARQLWrapper import SPARQLWrapper
from SPARQLWrapper import SPARQLExceptions
from urllib.error import HTTPError, URLError
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import configparser
import time
import socket
from urllib.parse import unquote
from dbpedia_utils import dbpedia_endpoint, dbpedia_resource_uri, normalize_wikipedia_tag

DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)


OUTPUT_PATH = DATA_DIR + 'wikidata dump.parquet'
CLASSFILE_PATH = DATA_DIR + 'wikidata classes.txt'
TESTRUN = config.getboolean('misc', 'testrun')
LIMIT = config.getint('misc', 'limit')
DBPEDIA_SOURCE = config.get('dbpedia scrape', 'dbpedia_source')
DBPEDIA_COUNTRY = config.get('dbpedia scrape', 'country')
ENTITY_SOURCE = config.get('dbpedia scrape', 'entity_source', fallback='country')
PW_FILENAME = config.get('postGIS', 'passwordfile', fallback='./config/pw.txt')
PG_HOST = config.get('postGIS', 'host', fallback='localhost')
PG_USER = config.get('postGIS', 'user', fallback='user')
PG_DB_NAME = config.get('postGIS', 'dbname', fallback='db')
PG_PORT = config.getint('postGIS', 'port', fallback=5432)
BASE_TABLE = config.get('entity linking', 'base_table', fallback='')
COVERAGE_REPORT_PATH = DATA_DIR + 'coverage_report.csv'

sparql = SPARQLWrapper(dbpedia_endpoint(DBPEDIA_SOURCE),
                   returnFormat='json',
                   agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)')

sparql.setTimeout(30)

print('-reading classes')
print(f'-from: {CLASSFILE_PATH}')
linked_classes = []
with open(CLASSFILE_PATH, 'r', encoding='utf-8') as file:
    for line in file.readlines():
        text = line.strip().replace('\n', '')
        if text:
            linked_classes.append(text)

def write_coverage_report(rows: list) -> None:
    if not rows:
        return
    with open(COVERAGE_REPORT_PATH, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['metric', 'value'])
        writer.writeheader()
        writer.writerows(rows)


async def collect_osm_wikipedia_titles() -> tuple:
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
        SELECT tags -> 'wikipedia' AS wikipedia, COUNT(*) AS count
        FROM {BASE_TABLE}
        WHERE tags -> 'wikipedia' IS NOT NULL
        GROUP BY tags -> 'wikipedia'
        """
    )
    await conn.close()

    title_counts = {}
    skip_counts = {}
    raw_count = 0
    for row in rows:
        raw_count += int(row['count'])
        title, reason = normalize_wikipedia_tag(row['wikipedia'], DBPEDIA_SOURCE)
        if title:
            title_counts[title] = title_counts.get(title, 0) + int(row['count'])
        else:
            skip_counts[reason or 'unknown'] = skip_counts.get(reason or 'unknown', 0) + int(row['count'])
    return title_counts, skip_counts, raw_count


def fetch_osm_linked_entities() -> tuple:
    title_counts, skip_counts, raw_count = asyncio.run(collect_osm_wikipedia_titles())
    titles = sorted(title_counts)
    if TESTRUN:
        titles = titles[:LIMIT]

    entity_query = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?item (AVG(?lat) AS ?lat) (AVG(?lon) AS ?lon) (SAMPLE(?type) AS ?type) WHERE {
    VALUES ?item {%s}
    ?item geo:lat ?lat.
    ?item geo:long ?lon.
    OPTIONAL {
        ?item rdf:type ?type.
        FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
    }
} GROUP BY ?item
"""

    entities = {}
    step_size = 70
    with tqdm(total=len(titles), desc='-Gathering OSM-linked DBpedia entities') as pbar:
        i = 0
        while i < len(titles):
            batch = titles[i:min(i + step_size, len(titles))]
            id_string = ' '.join(dbpedia_resource_uri(title, DBPEDIA_SOURCE) for title in batch)
            try:
                sparql.setQuery(entity_query % id_string)
                results = sparql.query().convert()
                for res in results['results']['bindings']:
                    id = unquote(res['item']['value'].split('/')[-1])
                    clazz = res.get('type', {}).get('value', '').split('/')[-1] or 'Thing'
                    entities[id] = {
                        'wkid': id,
                        'location': f"Point({res['lon']['value']} {res['lat']['value']})",
                        'type': clazz,
                        'source_wikipedia_count': title_counts.get(id, 0),
                    }
            except SPARQLExceptions.QueryBadFormed as e:
                pbar.write(repr(e))
            except HTTPError as e:
                time.sleep(5)
                pbar.write(f'{i}: {repr(e)}')
            except KeyboardInterrupt as e:
                print(repr(e))
                break
            except (socket.timeout, TimeoutError, URLError, OSError) as e:
                pbar.write(f'timeout: {i}: {repr(e)}')
                time.sleep(5)
            i += len(batch)
            pbar.update(len(batch))

    coverage_rows = [
        {'metric': 'osm_wikipedia_tag_rows', 'value': raw_count},
        {'metric': 'normalized_unique_titles', 'value': len(title_counts)},
        {'metric': 'queried_titles', 'value': len(titles)},
        {'metric': 'dbpedia_entities_with_coordinates', 'value': len(entities)},
        {'metric': 'dbpedia_entities_without_coordinates', 'value': max(0, len(titles) - len(entities))},
    ]
    for reason, count in sorted(skip_counts.items()):
        coverage_rows.append({'metric': f'skipped_{reason}', 'value': count})
    write_coverage_report(coverage_rows)
    return entities, coverage_rows


def fetch_country_entities() -> dict:
    query = """
PREFIX de: <http://de.dbpedia.org/resource/>
PREFIX fr: <http://fr.dbpedia.org/resource/>
PREFIX country: <http://dbpedia.org/resource/>
PREFIX db: <http://dbpedia.org/resource/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?item AVG(?lat) as ?lat AVG(?lon) as ?lon WHERE {
    ?item rdf:type dbo:%s.
    ?item dbo:country country:%s.
    ?item geo:lat ?lat.
    ?item geo:long ?lon
} group by ?item
"""

    entities = {}
    with tqdm(total=len(linked_classes), desc=f'-Gathering entities in ({DBPEDIA_COUNTRY})', miniters=1) as pbar:
        for clazz in linked_classes:
            try:
                pbar.set_postfix_str(f'class: {clazz}')
                sparql.setQuery(query % (clazz, DBPEDIA_COUNTRY))
                results = sparql.query().convert()
                for res in results['results']['bindings']:
                    id = res['item']['value'].split('/')[-1]
                    entities.update({id: {'wkid': id, 'location': f"Point({res['lon']['value']} {res['lat']['value']})", 'type': clazz}})
            except SPARQLExceptions.QueryBadFormed as e:
                print(repr(e))
            except HTTPError as e:
                time.sleep(5)
                pbar.write(f'{clazz}: {repr(e)}')
            except KeyboardInterrupt as e:
                print(repr(e))
                break
            except (socket.timeout, TimeoutError, URLError, OSError) as e:
                pbar.write(f'timeout: {clazz}: {repr(e)}')
                time.sleep(5)
            pbar.update(1)
            if TESTRUN:
                if len(entities) > LIMIT:
                    pbar.write(f'Test limit {LIMIT} reached')
                    break
    if TESTRUN:
        entities = dict(list(entities.items())[:LIMIT])
    return entities


if ENTITY_SOURCE == 'osm_linked':
    print('-entity source: OSM-linked wikipedia tags with DBpedia coordinates')
    entities, coverage_rows = fetch_osm_linked_entities()
else:
    print('-entity source: DBpedia country/class query')
    entities = fetch_country_entities()

def filter_key_value_pairs(pairs: list) -> str:
    """
    filter list of key value tuples and transform into single string
    :param pairs: list of key value tuples
    :return: concatenation of filtered key value pairs
    """
    filtered_items = []
    key_filter = ['owl#sameAs', 'subject', 'wikiPageUsesTemplate', 'wikiPageWikiLink', 'rdf-schema#comment', 'abstract', 'rdf-schema#label']
    value_filter = [DBPEDIA_COUNTRY]
    for k, v in pairs:
        if k in key_filter:
            # remove meta information
            pass
        elif str(k).startswith('wikiPage'):
            pass
        elif v in value_filter:
            pass
        elif re.match(r"Q[0-9]+", str(v)):
            # corresponding wikidata id
            pass
        else:
            # handle underscores from uri format
            key = str(k).replace('_', ' ')
            value = str(v).replace('_', ' ')
            filtered_items.extend([key, value])
    return ' '.join(filtered_items)

property_query = """
PREFIX de: <http://de.dbpedia.org/resource/>
PREFIX fr: <http://fr.dbpedia.org/resource/>
PREFIX db: <http://dbpedia.org/resource/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX dbo: <http://dbpedia.org/ontology/>

select distinct ?item ?property ?value
    where {
        VALUES ?item {%s}
        ?item ?property ?value.
    }
"""

for k, v in entities.items():
    v.update({'properties': ''})

i = 0
step_size = 70
with tqdm(total=len(entities), desc='-updating properties') as pbar:
    while i < len(entities):
        batch = list(entities.keys())[i:min(i + step_size, len(entities))]
        id_string = ' '.join(f"<http://dbpedia.org/resource/{e}>" for e in batch)
        try:
            sparql.setQuery(property_query % (id_string))
            results = sparql.query().convert()
            properties = {}
            for res in results['results']['bindings']:
                id = res['item']['value'].split('/')[-1]
                if id not in properties:
                    properties.update({id: []})

                key = res['property']['value'].split('/')[-1]
                value = res['value']['value'].split('/')[-1]
                properties[id].append((key, value))
            for k, v in properties.items():
                entities[k].update({'properties': filter_key_value_pairs(v)})
        except SPARQLExceptions.QueryBadFormed as e:
            print(repr(e))
        except HTTPError as e:
            time.sleep(5)
            pbar.write(f'{i}: {repr(e)}')
        except KeyboardInterrupt as e:
            print(repr(e))
            break
        except (socket.timeout, TimeoutError, URLError, OSError) as e:
            pbar.write(f'timeout: {i}: {repr(e)}')
            time.sleep(5)
        i += len(batch)
        pbar.update(len(batch))

def write_to_file(entity_dict: dict, filename: str) -> None:
    """
    function to transform entity dictionary into dataframe for saving in parquet format
    :param entity_dict: dictionary containing wikidata information
    :param filename: path to write file to
    :return:
    """
    data_list = list(entity_dict.values())
    table = pa.Table.from_pylist(data_list)
    with open(filename, 'wb') as file:
        pq.write_table(table, file)

print('-writing scraped data')
print(f'-to: {OUTPUT_PATH}')
write_to_file(entities, OUTPUT_PATH)
print('-writing complete')
