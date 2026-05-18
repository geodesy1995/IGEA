import pandas as pd
import sys
import configparser
import asyncio
import asyncpg

DATA_DIR = sys.argv[1]
CONFIG_PATH = sys.argv[2]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

CLASS_FILE = DATA_DIR + 'predicted classes.tsv'
CLASS_OUTPUT_PATH = DATA_DIR + 'wikidata classes.txt'
COLUMN_FILE = config.get('nca', 'columns_location').strip()
PW_FILENAME = config.get('postGIS', 'passwordfile')
PG_HOST = config.get('postGIS', 'host')
PG_USER = config.get('postGIS', 'user')
PG_DB_NAME = config.get('postGIS', 'dbname')
PG_PORT = config.getint('postGIS', 'port')
VIEW_NAME = config.get('entity linking', 'view_name')
INDEX_NAME = config.get('entity linking', 'index_name')
BASE_TABLE = config.get('entity linking', 'base_table')
PREDICTION_TABLE = config.get('entity linking', 'prediction_table')
VIEW_SQL_PATH = DATA_DIR + 'create view.sql'

classes = pd.read_csv(CLASS_FILE, delimiter='\t')

print('preparing class filtered entities')


def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"

# Collect all unique QIDs
print('-gathering dbpedia classes for entity linking')
async def gather_classes():
    unique_classes = []
    for val in set(classes.iloc[:, 1]):
        if 'person' not in str(val).lower():
            unique_classes.append(val)
    print(f'-found {len(unique_classes)} unique wiki classes matched')
    with open(CLASS_OUTPUT_PATH, 'w', encoding='utf-8', newline='') as file:
        file.write('\n'.join(unique_classes))

asyncio.run(gather_classes())

print('-selecting osm classes for entity linking')

if COLUMN_FILE:
    with open(COLUMN_FILE, 'r', encoding='utf-8') as file:
        column_names = file.read().split(', ')
else:
    column_names = []

tags = sorted(list(classes['tag']))
prev = ''
terms = []
same_group = set()
for s in tags:
    typ, subtype = s.split('=', 1)
    if typ != prev and same_group:
        is_column = False
        if prev in column_names:
            is_column = True
        elif f"\"{prev}\"" in column_names:
            is_column = True
            prev = f"\"{prev}\""

        if 'type' in prev:
            pass
        elif is_column:
            terms.append(f"{prev} in ({', '.join(same_group)})")
        else:
            terms.append(f"tags -> {sql_literal(prev)} in ({', '.join(same_group)})")
        same_group = set()
    same_group.add(sql_literal(subtype))
    prev = typ

is_column = False
if prev in column_names:
    is_column = True
elif f"\"{prev}\"" in column_names:
    is_column = True
    prev = f"\"{prev}\""

if 'type' in prev:
    pass
elif is_column:
    terms.append(f"{prev} in ({', '.join(same_group)})")
else:
    terms.append(f"tags -> {sql_literal(prev)} in ({', '.join(same_group)})")

delete_index_sql = f"DROP INDEX IF EXISTS {INDEX_NAME}"

delete_sql = f"DROP MATERIALIZED VIEW IF EXISTS {VIEW_NAME}"

index_sql = f"""
CREATE INDEX {INDEX_NAME} 
    ON {VIEW_NAME} 
    USING GIST (way)"""

verification_sql = f"SELECT COUNT(*) FROM {VIEW_NAME}"


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
    with open(PW_FILENAME, 'r', encoding='utf-8') as file:
        password = file.read().strip()

    conn = await asyncpg.connect(
        user=PG_USER,
        password=password,
        database=PG_DB_NAME,
        host=PG_HOST,
        port=PG_PORT
    )

    has_osm_uid = await table_has_column(conn, BASE_TABLE, 'osm_uid')
    join_expr = "gp.osm_uid = pe.osm_uid" if has_osm_uid else "gp.osm_id = pe.osm_id"
    term_clause = f"({' or '.join(terms)}) OR pe.wkid IS NOT NULL" if terms else "pe.wkid IS NOT NULL"
    sql = f"""
    CREATE MATERIALIZED VIEW {VIEW_NAME} as
    SELECT gp.*, pe.wkid
    FROM {BASE_TABLE} gp LEFT JOIN {PREDICTION_TABLE} pe ON {join_expr}
    WHERE {term_clause} WITH DATA"""

    print(f'-creating view {VIEW_NAME}')
    with open(VIEW_SQL_PATH, 'w', encoding='utf-8', newline='') as file:
        file.write(sql)

    async with conn.transaction():
        await conn.execute(delete_index_sql)
        await conn.execute(delete_sql)
        await conn.execute(sql)
        await conn.execute(index_sql)
        count = await conn.fetchval(verification_sql)
        
    print(f'-view contains {count} entries')

    await conn.close()

asyncio.run(execute_sql())

print('-view created')
