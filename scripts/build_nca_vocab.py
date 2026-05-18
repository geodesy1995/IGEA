import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


OSM_KEY_PREFIX = "<https://wiki.openstreetmap.org/wiki/Key:"
OSM_OBJECT_PREFIX = "<https://www.openstreetmap.org/"
LINK_KEYS = {"wikidata", "wikipedia"}
DEFAULT_EXCLUDED_KEYS = {
    "wikidata",
    "wikipedia",
    "created_by",
    "source",
    "source:ref",
    "odbl",
}
GEOMETRY_OR_RDF_KEYS = {
    "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
    "<http://www.w3.org/2003/01/geo/wgs84_pos#lat>",
    "<http://www.w3.org/2003/01/geo/wgs84_pos#long>",
    "<http://www.w3.org/2003/01/geo/wgs84_pos#Point>",
}
BOOLEAN_VALUES = {"yes", "no", "true", "false"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate IGEA/NCA OSM vocabulary CSV files from linked OSM entities. "
            "RDF triples are supported without extra dependencies; PBF input requires python-osmium."
        )
    )
    parser.add_argument("input", help="Input OSM RDF triples file or .osm.pbf/.pbf extract.")
    parser.add_argument(
        "--input-format",
        choices=["auto", "rdf", "pbf"],
        default="auto",
        help="Input format. Default infers PBF from .pbf/.osm.pbf extensions and RDF otherwise.",
    )
    parser.add_argument(
        "--link-keys",
        choices=["wikidata", "wikipedia", "both"],
        default="both",
        help="Which linked OSM entities to use when building the vocabulary.",
    )
    parser.add_argument(
        "--tag-output",
        default="./config/osmTagKeyWiki.csv",
        help="Output CSV path for key=value tags. Must contain a Tags column.",
    )
    parser.add_argument(
        "--key-output",
        default="./config/osmKeyWiki.csv",
        help="Output CSV path for keys. Must contain a Keys column.",
    )
    parser.add_argument(
        "--stats-output",
        default=None,
        help="Optional JSON stats path. Defaults to <tag-output basename>.stats.json.",
    )
    parser.add_argument(
        "--min-tag-frequency",
        type=int,
        default=2,
        help="Minimum frequency for key=value tags to be written.",
    )
    parser.add_argument(
        "--max-value-length",
        type=int,
        default=64,
        help="Drop key=value tags whose value is longer than this many characters.",
    )
    parser.add_argument(
        "--keep-yes-no",
        action="store_true",
        help="Keep boolean yes/no/true/false tag values. Default drops them.",
    )
    parser.add_argument(
        "--keep-numeric",
        action="store_true",
        help="Keep purely numeric tag values. Default drops them.",
    )
    parser.add_argument(
        "--keep-url",
        action="store_true",
        help="Keep URL-like tag values. Default drops them.",
    )
    parser.add_argument(
        "--keep-id-values",
        action="store_true",
        help="Keep identifier-like tag values. Default drops them.",
    )
    return parser.parse_args()


def infer_format(path: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lower = path.lower()
    if lower.endswith(".pbf") or lower.endswith(".osm.pbf"):
        return "pbf"
    return "rdf"


def selected_link_keys(option: str) -> Set[str]:
    if option == "both":
        return set(LINK_KEYS)
    return {option}


def normalize_key(raw_key: str) -> Optional[str]:
    key = raw_key.strip()
    if key in GEOMETRY_OR_RDF_KEYS:
        return None
    if key.startswith(OSM_KEY_PREFIX):
        key = key[len(OSM_KEY_PREFIX):]
        if key.endswith(">"):
            key = key[:-1]
    elif key.startswith("<http://") or key.startswith("<https://"):
        return None

    key = key.strip().strip("<>").replace(" ", "")
    if not key:
        return None
    return key


def normalize_node(raw_node: str) -> str:
    node = raw_node.strip()
    if node.startswith(OSM_OBJECT_PREFIX):
        node = node[len(OSM_OBJECT_PREFIX):]
        if node.endswith(">"):
            node = node[:-1]
    return node.strip("<>")


def normalize_value(raw_value: str) -> str:
    value = raw_value.strip()
    if value.endswith("."):
        value = value[:-1].strip()
    if "^^" in value and value.startswith('"'):
        value = value.split("^^", 1)[0]
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    value = value.replace('\\"', '"').replace("\\\\", "\\")
    return " ".join(value.split())


def is_numeric_value(value: str) -> bool:
    return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value.strip()))


def is_url_value(value: str) -> bool:
    lower = value.lower()
    return lower.startswith(("http://", "https://")) or lower.startswith("www.")


def is_identifier_like(value: str) -> bool:
    stripped = value.strip()
    if re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}", stripped):
        return True
    if re.fullmatch(r"[A-Za-z]{0,5}\d{4,}", stripped):
        return True
    return False


def should_keep_value(value: str, args: argparse.Namespace, drop_reasons: Counter) -> bool:
    if not value:
        drop_reasons["empty"] += 1
        return False
    if len(value) > args.max_value_length:
        drop_reasons["too_long"] += 1
        return False
    lower = value.lower()
    if not args.keep_yes_no and lower in BOOLEAN_VALUES:
        drop_reasons["boolean"] += 1
        return False
    if not args.keep_numeric and is_numeric_value(value):
        drop_reasons["numeric"] += 1
        return False
    if not args.keep_url and is_url_value(value):
        drop_reasons["url"] += 1
        return False
    if not args.keep_id_values and is_identifier_like(value):
        drop_reasons["identifier"] += 1
        return False
    return True


def add_linked_entity_tags(
    linked_entity_tags: Iterable[Dict[str, str]],
    args: argparse.Namespace,
) -> Tuple[Counter, Counter, Counter]:
    key_counts = Counter()
    tag_counts = Counter()
    drop_reasons = Counter()

    for tags in linked_entity_tags:
        for key, value in tags.items():
            if key in DEFAULT_EXCLUDED_KEYS:
                drop_reasons["excluded_key"] += 1
                continue
            key_counts[key] += 1
            if should_keep_value(value, args, drop_reasons):
                tag_counts[f"{key}={value}"] += 1

    return key_counts, tag_counts, drop_reasons


def read_linked_tags_from_rdf(path: str, link_keys: Set[str]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    node_tags: Dict[str, Dict[str, str]] = defaultdict(dict)
    linked_nodes = set()
    malformed_lines = 0

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                malformed_lines += 1
                continue
            node = normalize_node(parts[0])
            key = normalize_key(parts[1])
            if key is None:
                continue
            value = normalize_value(parts[2])
            node_tags[node][key] = value
            if key in link_keys:
                linked_nodes.add(node)

    linked_entity_tags = [node_tags[node] for node in sorted(linked_nodes) if node in node_tags]
    stats = {
        "total_entities_with_tags": len(node_tags),
        "linked_entity_count": len(linked_entity_tags),
        "malformed_lines": malformed_lines,
    }
    return linked_entity_tags, stats


def read_linked_tags_from_pbf(path: str, link_keys: Set[str]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    try:
        import osmium
    except ImportError as exc:
        raise RuntimeError(
            "PBF input requires the python-osmium package. Install it with `pip install osmium`, "
            "or pass an RDF triples file generated by scripts/osm2rdf.py."
        ) from exc

    class LinkedFeatureHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.total_entities_with_tags = 0
            self.total_nodes_with_tags = 0
            self.total_ways_with_tags = 0
            self.total_relations_with_tags = 0
            self.linked_entity_tags: List[Dict[str, str]] = []

        def add_tags(self, osm_object, object_kind: str):
            if not osm_object.tags:
                return
            tags = {str(key): str(value) for key, value in osm_object.tags}
            self.total_entities_with_tags += 1
            if object_kind == "node":
                self.total_nodes_with_tags += 1
            elif object_kind == "way":
                self.total_ways_with_tags += 1
            elif object_kind == "relation":
                self.total_relations_with_tags += 1
            if any(key in tags for key in link_keys):
                self.linked_entity_tags.append(tags)

        def node(self, node):
            self.add_tags(node, "node")

        def way(self, way):
            self.add_tags(way, "way")

        def relation(self, relation):
            self.add_tags(relation, "relation")

    handler = LinkedFeatureHandler()
    handler.apply_file(path, locations=False)
    stats = {
        "total_entities_with_tags": handler.total_entities_with_tags,
        "total_nodes_with_tags": handler.total_nodes_with_tags,
        "total_ways_with_tags": handler.total_ways_with_tags,
        "total_relations_with_tags": handler.total_relations_with_tags,
        "linked_entity_count": len(handler.linked_entity_tags),
        "malformed_lines": 0,
    }
    return handler.linked_entity_tags, stats


def write_counter_csv(path: str, header: str, counter: Counter, minimum: int = 1) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = [(value, count) for value, count in counter.items() if count >= minimum]
    rows.sort(key=lambda row: (-row[1], row[0]))

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([header, "Count"])
        writer.writerows(rows)

    return len(rows)


def default_stats_path(tag_output: str) -> str:
    root, _ = os.path.splitext(tag_output)
    return f"{root}.stats.json"


def main() -> None:
    args = parse_args()
    if args.min_tag_frequency < 1:
        raise ValueError("--min-tag-frequency must be at least 1.")
    if args.max_value_length < 1:
        raise ValueError("--max-value-length must be at least 1.")

    input_format = infer_format(args.input, args.input_format)
    link_keys = selected_link_keys(args.link_keys)

    if input_format == "rdf":
        linked_entity_tags, source_stats = read_linked_tags_from_rdf(args.input, link_keys)
    else:
        linked_entity_tags, source_stats = read_linked_tags_from_pbf(args.input, link_keys)

    key_counts, tag_counts, drop_reasons = add_linked_entity_tags(linked_entity_tags, args)

    written_key_count = write_counter_csv(args.key_output, "Keys", key_counts)
    written_tag_count = write_counter_csv(args.tag_output, "Tags", tag_counts, args.min_tag_frequency)

    stats = {
        "input": args.input,
        "input_format": input_format,
        "link_keys": sorted(link_keys),
        "min_tag_frequency": args.min_tag_frequency,
        "max_value_length": args.max_value_length,
        **source_stats,
        "raw_key_count": len(key_counts),
        "raw_tag_count": len(tag_counts),
        "written_key_count": written_key_count,
        "written_tag_count": written_tag_count,
        "dropped_values": dict(sorted(drop_reasons.items())),
        "key_output": args.key_output,
        "tag_output": args.tag_output,
    }

    stats_path = args.stats_output or default_stats_path(args.tag_output)
    os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, sort_keys=True)

    print(f"linked entities: {stats['linked_entity_count']}")
    print(f"written keys: {written_key_count} -> {args.key_output}")
    print(f"written tags: {written_tag_count} -> {args.tag_output}")
    print(f"stats: {stats_path}")


if __name__ == "__main__":
    main()
