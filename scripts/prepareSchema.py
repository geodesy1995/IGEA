import sys
import configparser
import asyncio
import asyncpg
import random

CONFIG_PATH = sys.argv[1]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

PW_FILENAME = config.get('postGIS', 'passwordfile')
PG_HOST = config.get('postGIS', 'host')
PG_USER = config.get('postGIS', 'user')
PG_DB_NAME = config.get('postGIS', 'dbname')
PG_PORT = config.getint('postGIS', 'port')
TABLE_NAME = config.get('entity linking', 'prediction_table')
BASE_TABLE = config.get('entity linking', 'base_table')
VIEW_NAME = config.get('entity linking', 'view_name')
INDEX_NAME = config.get('entity linking', 'index_name')
DATA_SOURCE = config.get('meta', 'kg_source')
DBPEDIA_SOURCE = config.get('dbpedia scrape', 'dbpedia_source', fallback='en')
GOLD_SPLIT_ENABLED = config.getboolean('gold split', 'enabled', fallback=False)
GOLD_SPLIT_SEED_FRACTION = config.getfloat('gold split', 'seed_fraction', fallback=1.0)
GOLD_SPLIT_RANDOM_SEED = config.getint('gold split', 'random_seed', fallback=42)
HELDOUT_TABLE = config.get('gold split', 'heldout_table', fallback='heldout_entities')

print('preparing schema')

with open(PW_FILENAME, 'r') as file:
    password = file.read().strip()

delete_index_sql = f"DROP INDEX IF EXISTS {INDEX_NAME}"
delete_view = f"DROP MATERIALIZED VIEW IF EXISTS {VIEW_NAME}"
delete_table = f"DROP TABLE IF EXISTS {TABLE_NAME}"
delete_heldout_table = f"DROP TABLE IF EXISTS {HELDOUT_TABLE}"

table_sql = """
CREATE TABLE {table_name}(
	id BIGSERIAL,
	wkid text not null,
	osm_uid text not null,
	osm_type text,
	osm_id BIGINT not null,
	confidence float not null,
	iteration int not null
)
"""

sql = table_sql.format(table_name=TABLE_NAME)
heldout_sql = table_sql.format(table_name=HELDOUT_TABLE)


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

async def execute_sql():
    conn = await asyncpg.connect(
        user=PG_USER,
        password=password,
        database=PG_DB_NAME,
        host=PG_HOST,
        port=PG_PORT
    )

    has_osm_uid = await table_has_column(conn, BASE_TABLE, 'osm_uid')
    has_osm_type = await table_has_column(conn, BASE_TABLE, 'osm_type')

    osm_uid_expr = "osm_uid" if has_osm_uid else "osm_id::text"
    osm_type_expr = "osm_type" if has_osm_type else "'N'"

    if DATA_SOURCE == 'wikidata':
        gold_sql = f"""
            SELECT tags -> 'wikidata', {osm_uid_expr}, {osm_type_expr}, osm_id, 1.0, 0
            FROM {BASE_TABLE}
            WHERE tags -> 'wikidata' ~ '^Q[0-9]+$'
        """
    else:
        gold_sql = f"""
            SELECT tags -> 'wikipedia', {osm_uid_expr}, {osm_type_expr}, osm_id, 1.0, 0
            FROM {BASE_TABLE}
            WHERE tags -> 'wikipedia' is not null
              AND (
                  lower(tags -> 'wikipedia') LIKE '{DBPEDIA_SOURCE}:%'
                  OR lower(tags -> 'wikipedia') LIKE 'http://{DBPEDIA_SOURCE}.wikipedia.org/wiki/%'
                  OR lower(tags -> 'wikipedia') LIKE 'https://{DBPEDIA_SOURCE}.wikipedia.org/wiki/%'
              )
        """

    async with conn.transaction():
        print('-deleting old view')
        await conn.execute(delete_index_sql)
        await conn.execute(delete_view)
        print('-deleting old prediction table')
        await conn.execute(delete_table)
        await conn.execute(delete_heldout_table)
        print(f'-creating table {TABLE_NAME}')
        await conn.execute(sql)
        print(f'-creating table {HELDOUT_TABLE}')
        await conn.execute(heldout_sql)
        print(f'-filling {TABLE_NAME} with ground truth')
        gold_rows = await conn.fetch(gold_sql)
        seed_rows = []
        heldout_rows = []
        if GOLD_SPLIT_ENABLED and GOLD_SPLIT_SEED_FRACTION < 1.0:
            rng = random.Random(GOLD_SPLIT_RANDOM_SEED)
            groups = sorted({str(row[0]) for row in gold_rows})
            seed_groups = {group for group in groups if rng.random() < GOLD_SPLIT_SEED_FRACTION}
            if groups and not seed_groups:
                seed_groups.add(groups[0])
            if len(seed_groups) == len(groups) and len(groups) > 1:
                seed_groups.remove(groups[-1])

            for row in gold_rows:
                entry = (str(row[0]), str(row[1]), row[2], int(row[3]), float(row[4]), int(row[5]))
                if str(row[0]) in seed_groups:
                    seed_rows.append(entry)
                else:
                    heldout_rows.append(entry)
        else:
            seed_rows = [
                (str(row[0]), str(row[1]), row[2], int(row[3]), float(row[4]), int(row[5]))
                for row in gold_rows
            ]

        insert_seed_sql = f"""
        INSERT INTO {TABLE_NAME} (wkid, osm_uid, osm_type, osm_id, confidence, iteration)
        VALUES ($1, $2, $3, $4, $5, $6)
        """
        insert_heldout_sql = f"""
        INSERT INTO {HELDOUT_TABLE} (wkid, osm_uid, osm_type, osm_id, confidence, iteration)
        VALUES ($1, $2, $3, $4, $5, $6)
        """
        if seed_rows:
            await conn.executemany(insert_seed_sql, seed_rows)
        if heldout_rows:
            await conn.executemany(insert_heldout_sql, heldout_rows)

        print(f'-direct gold rows: {len(gold_rows)}')
        print(f'-seed gold rows: {len(seed_rows)}')
        print(f'-heldout gold rows: {len(heldout_rows)}')
        if GOLD_SPLIT_ENABLED:
            print(f'-gold split: seed_fraction={GOLD_SPLIT_SEED_FRACTION}, random_seed={GOLD_SPLIT_RANDOM_SEED}')

    await conn.close()

asyncio.run(execute_sql())
