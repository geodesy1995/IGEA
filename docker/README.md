# IGEA Docker Runtime

This folder contains the local Docker runtime for IGEA experiments.

Start PostGIS:

```powershell
docker compose up -d postgis
```

Check the database:

```powershell
docker compose exec postgis psql -U user -d db -c "\dt"
```

The experiment config expects:

- host: `localhost`
- port: `5432`
- user: `user`
- database: `db`
- password file: `config/pw.txt`

For local experiments, `config/pw.txt` should contain one line:

```text
igea
```

The actual PostgreSQL data is stored in the Docker named volume
`igea_postgis_data`, not inside the Git workspace.

To avoid installing `osm2pgsql` on Windows, import the Ireland/Northern Ireland
PBF with the Docker-based helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\import_ireland_osm.ps1
```

The helper uses `docker/osm2pgsql/ireland_nodes.lua` and writes the table
`ireland_nodes`.
