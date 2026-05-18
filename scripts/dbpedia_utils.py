from typing import Optional, Tuple
from urllib.parse import unquote, urlparse


def normalize_wikipedia_tag(value: object, dbpedia_source: str = "en") -> Tuple[Optional[str], Optional[str]]:
    """Return (DBpedia resource title, skip reason) for an OSM wikipedia tag."""
    if value is None:
        return None, "missing"

    text = str(value).strip().strip('"').strip("'")
    if not text:
        return None, "empty"

    language = None
    title = text

    if "wikipedia.org/wiki/" in text:
        parsed = urlparse(text)
        host = parsed.netloc.lower()
        if host.endswith("wikipedia.org"):
            language = host.split(".", 1)[0]
        title = parsed.path.split("/wiki/", 1)[-1]
    elif ":" in text:
        prefix, remainder = text.split(":", 1)
        if len(prefix) in {2, 3} and prefix.isalpha():
            language = prefix.lower()
            title = remainder

    if language and language != dbpedia_source:
        return None, f"language:{language}"

    title = unquote(title).replace(" ", "_")
    title = title.replace("\\", "").strip("_")
    if not title:
        return None, "empty_title"
    if title.startswith("http://") or title.startswith("https://"):
        return None, "unsupported_url"
    return title, None


def dbpedia_resource_uri(title: str, dbpedia_source: str = "en") -> str:
    base = "http://dbpedia.org/resource" if dbpedia_source == "en" else f"http://{dbpedia_source}.dbpedia.org/resource"
    return f"<{base}/{title}>"


def dbpedia_endpoint(dbpedia_source: str = "en") -> str:
    return "http://dbpedia.org/sparql" if dbpedia_source == "en" else f"http://{dbpedia_source}.dbpedia.org/sparql"
