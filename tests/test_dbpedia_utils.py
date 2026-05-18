import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from dbpedia_utils import normalize_wikipedia_tag


def test_normalize_wikipedia_language_prefix():
    assert normalize_wikipedia_tag("en:Some Title", "en") == ("Some_Title", None)


def test_normalize_wikipedia_url():
    assert normalize_wikipedia_tag("https://en.wikipedia.org/wiki/Some_Title", "en") == ("Some_Title", None)


def test_normalize_wikipedia_space_and_underscore_equivalent():
    assert normalize_wikipedia_tag("en:Some Title", "en")[0] == normalize_wikipedia_tag("en:Some_Title", "en")[0]


def test_normalize_wikipedia_rejects_other_language():
    assert normalize_wikipedia_tag("fr:Some_Title", "en") == (None, "language:fr")
