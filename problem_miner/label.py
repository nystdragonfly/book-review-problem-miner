"""
LLM-generated cluster title + one-sentence summary, via a local Ollama
server. Chosen over the Anthropic API to keep the whole pipeline local/
free, consistent with every other model used (see CLAUDE.md "Working
conventions"). A TF-IDF keyword-tag alternative was built and compared
side by side against real output, then dropped -- inconsistent value,
often redundant with the LLM title (see devlog 2026-08-20).
"""
import requests

from .config import DEFAULT_CONFIG, PipelineConfig


def _strip_markdown(s: str) -> str:
    # Models (including the default one here, mistral-nemo:12b) like to
    # wrap labels in markdown bold ("**TITLE:**") despite being told not
    # to -- strip before matching or parsing silently fails. Real bug
    # hit and fixed during development (devlog 2026-08-20).
    return s.strip().strip("*").strip()


def label_cluster(
    example_sentences: list[str],
    category: str,
    book_context: str | None = None,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[str, str]:
    """book_context: a short (1-3 sentence) description of the book these
    reviews are about, e.g. genre/premise. Optional -- omitted from the
    prompt if not supplied, since there's no generalized way yet to
    auto-derive this for an arbitrary book/source (see pipeline.py)."""
    context_block = f"{book_context}\n\n" if book_context else ""
    prompt = f"""{context_block}Below are representative sentences from ONE cluster of reader review sentences, all grouped together because they discuss a similar theme. This cluster's overall category is: {category}.

Sentences:
{chr(10).join(f"- {s}" for s in example_sentences)}

Write a short title (5-8 words, naming the specific theme) and a one-sentence summary of what reviewers are saying about it.

Respond in EXACTLY this format, nothing else:
TITLE: <title>
SUMMARY: <one sentence>"""

    resp = requests.post(config.ollama_url, json={
        "model": config.ollama_model, "prompt": prompt, "stream": False,
    }, timeout=config.ollama_timeout_seconds)
    resp.raise_for_status()
    text = resp.json()["response"].strip()

    title, summary = "(parse failed)", text
    for line in text.splitlines():
        cleaned = _strip_markdown(line)
        if cleaned.upper().startswith("TITLE:"):
            title = _strip_markdown(cleaned.split(":", 1)[1])
        elif cleaned.upper().startswith("SUMMARY:"):
            summary = _strip_markdown(cleaned.split(":", 1)[1])
    return title, summary
