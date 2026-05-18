local tables = {}

tables.node = osm2pgsql.define_table{
    name = "ireland_nodes",
    ids = { type = "node", id_column = "osm_id" },
    columns = {
        { column = "tags", type = "hstore" },
        { column = "way", type = "point", not_null = true },
    }
}

function clean_tags(tags)
    tags.odbl = nil
    tags.created_by = nil
    tags.source = nil
    tags["source:ref"] = nil

    return next(tags) == nil
end

function osm2pgsql.process_node(object)
    if clean_tags(object.tags) then
        return
    end

    tables.node:insert({
        tags = object.tags,
        way = object:as_point()
    })
end
