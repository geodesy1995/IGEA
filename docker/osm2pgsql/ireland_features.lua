local tables = {}

tables.features = osm2pgsql.define_table{
    name = "ireland_features",
    ids = { type = "any", id_column = "osm_id", type_column = "osm_type" },
    columns = {
        { column = "osm_uid", type = "text", not_null = true },
        { column = "tags", type = "hstore" },
        { column = "way", type = "geometry", projection = 3857, not_null = true },
    }
}

function clean_tags(tags)
    tags.odbl = nil
    tags.created_by = nil
    tags.source = nil
    tags["source:ref"] = nil

    return next(tags) == nil
end

function insert_feature(object, osm_type, geom)
    if geom == nil then
        return
    end

    tables.features:insert({
        osm_uid = osm_type .. tostring(object.id),
        tags = object.tags,
        way = geom
    })
end

function osm2pgsql.process_node(object)
    if clean_tags(object.tags) then
        return
    end

    insert_feature(object, "N", object:as_point())
end

function osm2pgsql.process_way(object)
    if clean_tags(object.tags) then
        return
    end

    local geom = nil
    if object.is_closed then
        geom = object:as_polygon()
    else
        geom = object:as_linestring()
    end

    insert_feature(object, "W", geom)
end

function osm2pgsql.process_relation(object)
    if clean_tags(object.tags) then
        return
    end

    if object.tags.type == "multipolygon" or object.tags.type == "boundary" then
        insert_feature(object, "R", object:as_multipolygon())
    end
end
