import asyncio
import configparser
import sys

import asyncpg


CONFIG_PATH = sys.argv[1]

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

PW_FILENAME = config.get('postGIS', 'passwordfile')
PG_HOST = config.get('postGIS', 'host')
PG_USER = config.get('postGIS', 'user')
PG_DB_NAME = config.get('postGIS', 'dbname')
PG_PORT = config.getint('postGIS', 'port')
BASE_TABLE = config.get('entity linking', 'base_table')
PREDICTION_TABLE = config.get('entity linking', 'prediction_table')
VIEW_NAME = config.get('entity linking', 'view_name')
INDEX_NAME = config.get('entity linking', 'index_name')


with open(PW_FILENAME, 'r', encoding='utf-8') as file:
    password = file.read().strip()


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
        port=PG_PORT,
    )

    has_osm_uid = await table_has_column(conn, BASE_TABLE, 'osm_uid')
    join_expr = "gp.osm_uid = pe.osm_uid" if has_osm_uid else "gp.osm_id = pe.osm_id"

    delete_index_sql = f"DROP INDEX IF EXISTS {INDEX_NAME}"
    delete_view_sql = f"DROP MATERIALIZED VIEW IF EXISTS {VIEW_NAME}"
    create_view_sql = f"""
    CREATE MATERIALIZED VIEW {VIEW_NAME} AS
    SELECT gp.*, pe.wkid
    FROM {BASE_TABLE} gp
    LEFT JOIN {PREDICTION_TABLE} pe ON {join_expr}
    WHERE gp.way IS NOT NULL
      AND NOT ST_IsEmpty(gp.way)
    WITH DATA
    """
    create_index_sql = f"""
    CREATE INDEX {INDEX_NAME}
        ON {VIEW_NAME}
        USING GIST (way)
    """
    verification_sql = f"SELECT COUNT(*) FROM {VIEW_NAME}"
    linked_sql = f"SELECT COUNT(*) FROM {VIEW_NAME} WHERE wkid IS NOT NULL"

    print('creating direct candidate view')
    async with conn.transaction():
        await conn.execute(delete_index_sql)
        await conn.execute(delete_view_sql)
        await conn.execute(create_view_sql)
        await conn.execute(create_index_sql)
        count = await conn.fetchval(verification_sql)
        linked_count = await conn.fetchval(linked_sql)

    print(f'-view contains {count} entries')
    print(f'-direct linked entries: {linked_count}')
    await conn.close()


asyncio.run(execute_sql())
