import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


VALID_VERIFIER_RESULTS = {"match", "non-match", "unsure"}


@dataclass(frozen=True)
class VerificationDecision:
    decision: str
    confidence: float = 0.0
    reason: str = ""
    raw_response: str = ""


VERIFIER_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["match", "non-match", "unsure"],
        },
        "confidence": {
            "type": "number",
        },
        "reason": {
            "type": "string",
        },
    },
    "required": ["decision", "confidence", "reason"],
}


def _normalize_decision(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"non_match", "nonmatch", "not-match", "not match", "different"}:
        return "non-match"
    if text in VALID_VERIFIER_RESULTS:
        return text
    return "unsure"


def _parse_json_decision(text: str) -> VerificationDecision:
    payload = json.loads(text)
    decision = _normalize_decision(payload.get("decision"))
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(payload.get("reason", "")).strip()
    return VerificationDecision(decision=decision, confidence=confidence, reason=reason, raw_response=text)


class LLMVerifier(ABC):
    """
    Abstract base class for LLM Verifiers.
    Implement this class to connect a local or cloud-based LLM (e.g., HuggingFace, Ollama, LangChain, OpenAI).
    """

    @abstractmethod
    def verify(self, osm_tags: str, wikidata_properties: str, evidence_pack: Optional[Dict[str, object]] = None) -> str:
        """
        Consumes an evidence pack (osm_tags and wikidata_properties) and returns a verification result.
        
        Args:
            osm_tags (str): The tags of the OSM entity.
            wikidata_properties (str): The properties of the Wikidata entity.
            evidence_pack (dict): Compact structured evidence such as scores and spatial cues.
            
        Returns:
            str: "match" if the LLM is confident they are the same entity,
                 "non-match" if the LLM is confident they are not the same,
                 "unsure" if the LLM cannot confidently decide.
        """
        pass

    def verify_with_details(
        self,
        osm_tags: str,
        wikidata_properties: str,
        evidence_pack: Optional[Dict[str, object]] = None,
    ) -> VerificationDecision:
        decision = _normalize_decision(self.verify(osm_tags, wikidata_properties, evidence_pack))
        return VerificationDecision(decision=decision)


class DummyLLMVerifier(LLMVerifier):
    """
    A dummy implementation of the LLMVerifier used as a placeholder.
    In practice, you should instantiate your custom verifier class and use it.
    """
    def verify(self, osm_tags: str, wikidata_properties: str, evidence_pack: Optional[Dict[str, object]] = None) -> str:
        # Default behavior: do not override the neural network's decision
        return "unsure"


class OpenAILLMVerifier(LLMVerifier):
    """
    OpenAI-backed verifier for low-margin GeoEA candidate pairs.
    The API key is read from OPENAI_API_KEY and is never stored in experiment outputs.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 180,
    ):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Set it in the shell before running the OpenAI verifier.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is not installed. Install project requirements before running provider=openai."
            ) from exc

        self.client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        self.model = model
        self.max_output_tokens = max_output_tokens

    def verify(self, osm_tags: str, wikidata_properties: str, evidence_pack: Optional[Dict[str, object]] = None) -> str:
        return self.verify_with_details(osm_tags, wikidata_properties, evidence_pack).decision

    def verify_with_details(
        self,
        osm_tags: str,
        wikidata_properties: str,
        evidence_pack: Optional[Dict[str, object]] = None,
    ) -> VerificationDecision:
        prompt = self._build_prompt(osm_tags, wikidata_properties, evidence_pack or {})
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You verify whether an OSM feature and a DBpedia geographic entity refer to the same "
                        "real-world entity. Use only the provided evidence. Return strict JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "geo_entity_verification",
                    "strict": True,
                    "schema": VERIFIER_RESPONSE_SCHEMA,
                }
            },
            max_output_tokens=self.max_output_tokens,
        )
        output_text = getattr(response, "output_text", "")
        if not output_text:
            output_text = self._extract_output_text(response)
        return _parse_json_decision(output_text)

    @staticmethod
    def _truncate(text: object, limit: int = 2500) -> str:
        value = str(text or "")
        if len(value) <= limit:
            return value
        return value[:limit] + " ... [truncated]"

    def _build_prompt(self, osm_tags: str, wikidata_properties: str, evidence_pack: Dict[str, object]) -> str:
        compact_evidence = {
            "wkid": evidence_pack.get("wkid"),
            "osm_id": evidence_pack.get("osm_id"),
            "probability": evidence_pack.get("probability"),
            "threshold": evidence_pack.get("threshold"),
            "margin": evidence_pack.get("margin"),
            "spatial_features": evidence_pack.get("spatial_features", {}),
        }
        return (
            "Decide whether the OSM feature and DBpedia entity are the same entity.\n"
            "Return decision='match' only when names/types/location evidence strongly agree.\n"
            "Return decision='non-match' when evidence clearly conflicts.\n"
            "Return decision='unsure' when evidence is ambiguous or insufficient.\n\n"
            f"Evidence pack JSON:\n{json.dumps(compact_evidence, ensure_ascii=False, sort_keys=True)}\n\n"
            f"OSM tags:\n{self._truncate(osm_tags)}\n\n"
            f"DBpedia properties:\n{self._truncate(wikidata_properties)}"
        )

    @staticmethod
    def _extract_output_text(response: object) -> str:
        output = getattr(response, "output", None) or []
        parts = []
        for item in output:
            if isinstance(item, dict):
                content = item.get("content") or []
            else:
                content = getattr(item, "content", None) or []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                else:
                    text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)


# Define the active verifier to be used in the prediction pipeline
def get_verifier(
    provider: str = "dummy",
    model: str = "gpt-4o-mini",
    timeout_seconds: float = 30.0,
    max_output_tokens: int = 180,
) -> LLMVerifier:
    normalized_provider = (provider or "dummy").strip().lower()
    if normalized_provider == "dummy":
        return DummyLLMVerifier()
    if normalized_provider == "openai":
        return OpenAILLMVerifier(
            model=model,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )
    else:
        raise ValueError(
            f"Unsupported LLM verifier provider '{provider}'. "
            "Supported providers: dummy, openai."
        )
