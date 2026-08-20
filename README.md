# Book Review Problem-Miner

A tool that reads through a book's reviews and surfaces the *specific
problems readers report* — grouped into themes, not buried in a pile of
star ratings. Point it at a book; it tells you "here's what people
actually complain about," "here's what they praise," and "here's what
you should know before picking it up" — instead of a single blended
4.2★ average that tells you nothing about *why*.

**Status: working pipeline, not a hardcoded demo.** `problem_miner/` is a
real, installable package with a CLI (`python -m problem_miner --source
{jsonl,goodreads} --reviews-file PATH --book-id ID`) — book-agnostic,
not one script per book. Validated end-to-end against two different real
data sources (see "Results" below). See [`devlog.md`](devlog.md) for the
complete session-by-session history and [`CLAUDE.md`](CLAUDE.md) for
current implementation notes.

## Built in collaboration with AI (Claude Code)

I'm upfront about this: the implementation happened in partnership with
an AI coding assistant. What I actually did — direct the approach,
make the scope and product calls, catch problems in the output, and
verify results rather than trust them on sight — is the point of this
README, not something to gloss over. The case study below is a concrete
example of what that looks like in practice, not a hypothetical.

## Case study: why clustering isn't `import kmeans` and call it done

The pipeline embeds review sentences (turns them into vectors that
capture meaning, not just literal words) and clusters similar ones
together to find recurring themes. The first working version used
k-means, the standard/default clustering algorithm, and it produced 8
clean-looking clusters with readable labels. Nothing about it looked
broken.

But I'd already run a smaller sanity check earlier — asking the
embedding model to find sentences similar to *"the artwork was hard to
follow"* (a phrase that appears nowhere in the dataset verbatim) — and
it correctly surfaced real complaints like *"I found the art
distractingly poor"* and *"Didn't like the artwork."* So I knew that
signal existed in the data.

I looked at the k-means output and **that theme wasn't there.** Not
missing outright — folded into a much larger "general negative
sentiment" cluster, indistinguishable from complaints about the plot,
the pacing, anything else negative.

We traced it: k-means always assigns *every* point to one of a fixed
number of clusters. It has no concept of "this is a real but small
group" — a handful of scattered artwork complaints, sitting near a much
bigger cloud of generic negativity, just gets pulled into that bigger
cloud. The fix wasn't "try a different number of clusters" (we tried —
it didn't help); it was switching to a fundamentally different kind of
algorithm.

The first attempt at that fix — HDBSCAN, which allows "this doesn't
belong to any cluster" as a valid answer instead of forcing everything
in — also failed, in a different way: run directly on the raw
768-dimensional embeddings, it produced either one giant undifferentiated
blob or almost 100% noise, depending on how it was tuned. The reason:
density-based clustering becomes unreliable in very high-dimensional
spaces (points stop being meaningfully "close" or "far" from each
other). The actual fix was reducing the embeddings to 5 dimensions
first (via UMAP), *then* clustering — at which point the artwork
complaints separated out cleanly into their own 29-sentence cluster,
distinct from a separate artwork-*praise* cluster.

That's three attempts, two real failures, and two different diagnosable
reasons *why* — found because a known-good signal was checked against
the actual output rather than trusted because the output looked
plausible.

## Case study #2: when the obvious fix doesn't hold up

A second example of the same practice — catch it, investigate with real
data, and specifically be willing to abandon a fix that *looks* right
once the data says otherwise.

While reviewing labeled output, a 4★ review's sentence — *"The
trap-engineering stuff scratched an itch I didn't know I had"* — showed
up tagged **negative**. Clearly positive to any human reader. Worth
checking whether that was a one-off or a real pattern: it turned out to
affect **9–23% of 4★ sentences** (depending on dataset), always the same
shape — praise phrased as a subverted expectation ("didn't expect to
care," "no stat screens, no XP bars, just..."). The sentiment model
keys on the negation word, not the sentence's actual meaning — a known,
general weakness in sentiment analysis, not specific to this pipeline.

The obvious fix — swap in a general-purpose LLM (already running
locally for cluster labeling) — turned out to be **not a clean win**.
Tested directly rather than assumed: the LLM did better on negated
praise (4/8 vs. 0/8) but *worse* on genuinely negative terse dismissals
like *"Not for me."* (0/4 vs. 4/4) — it needs context a one-line
sentence doesn't give it. A cheaper idea — route only "probably
mismatched" sentences to the LLM based on sentence length, which looked
clean on a handful of hand-picked examples — **failed when checked
against the full 445-sentence candidate set** (the length pattern was a
small-sample coincidence, not real). Abandoned rather than shipped.

Landed on a narrower, verified rule: re-check negative-classified,
negation-containing sentences against the local LLM, but only
auto-apply the correction where disagreement was spot-checked as
reliable (negative→positive, ~100% real corrections) — not the direction
that turned out to be a genuine mixed bag (negative→neutral, real fixes
mixed with real regressions). Implemented in
[`problem_miner/categorize.py`](problem_miner/categorize.py).

## A licensing decision, not just a technical one

Worth being upfront about the actual sequence here, not just the tidy
version: this project **started** on that same real Goodreads dataset —
it's what the case study above ran on, and at the time it was the
obvious choice for prototyping (large, well-documented, free, exactly
the shape of data needed). It was only partway through, once the
technical approach was already validated, that the dataset's license
got read closely: "academic use only... should not be used for
commercial purposes." Fine for the exploratory work already done —
genuinely non-commercial, nothing redistributed — but a real conflict
with this project's actual purpose: demonstrating skills to attract
paid work. That's a mistake caught and corrected mid-project, not
something avoided from the start, and I'd rather show that than pretend
the dataset decision was clean from day one.

Rather than quietly keep using it, I went looking for a genuinely
permissively-licensed alternative and verified rather than assumed —
which turned out to matter. One dataset labeled "CC0" on Hugging Face
traced back to the exact same restricted academic source, with the
license apparently relabeled somewhere in a Kaggle→HuggingFace
re-upload chain by someone who didn't have the rights to grant it.
Found zero trustworthy free/reusable options. The legitimate paid
alternative exists but isn't a real option for a project like this one —
compliant dataset access starts around $800/month for a single dataset,
a barrier for an independent operator that a company with a data-
licensing budget wouldn't face, and cheaper providers showed clear signs
of sourcing their data via the same kind of questionable scraping this
whole exercise was trying to avoid. Considered scraping review data
myself and rejected it too — that would've been a *worse* legal position
than the original problem, personally committing a platform ToS
violation specifically to build the commercial-facing demo.

Landed on synthetic data instead: AI-generated reviews of a real,
fully-owned manuscript, with zero licensing ambiguity since nothing is
scraped or redistributed. Disclosed limitation, not a solved problem:
synthetic review text is measurably cleaner and less messy than real
internet text — confirmed directly by comparing cluster output between
the two datasets, not just assumed. The debugging story above doesn't
depend on which dataset it ran on; the licensing decision is a separate
piece of judgment from the clustering one.

## How it works

| Step | What | Why this choice |
|---|---|---|
| Filter | Drop 5★ and unrated reviews | Complaints concentrate in 1–4★; 5★ reviews are low-signal/promotional |
| Clean | Normalize whitespace, drop non-English (`langdetect`) | Raw scraped text has formatting artifacts; mixed-language embeddings cluster poorly together |
| Split | Sentence-level (`nltk`), not whole-review | A single review often mixes praise and complaint — sentence granularity keeps them separable |
| Embed | `sentence-transformers` (`all-mpnet-base-v2`), GPU-accelerated | Captures meaning, not keywords — see case study above for why that matters concretely |
| Cluster | UMAP (dimensionality reduction) → HDBSCAN | See case study — the straightforward approach (k-means) provably lost real themes |
| Categorize | Per-sentence sentiment classification (not review star rating) → negative / positive / neutral, plus a junk filter, plus a targeted LLM re-check for negation misreads (see case study #2) | A sentence's own text is classified directly, since one review's rating doesn't apply evenly to all its sentences |
| Label | LLM-generated title + one-sentence summary per cluster, via a local model (Ollama, `mistral-nemo:12b`) — skipped for junk clusters, which get a fixed label instead | Raw example sentences aren't a finished answer to "what is this theme" — a local model keeps this step free and consistent with everything else being local; asking it to summarize content-free junk clusters caused it to hallucinate a theme (an actual bug hit and fixed — see `problem_miner/pipeline.py`); a TF-IDF keyword-tag alternative was built and compared, then dropped as net noise |

Implementation: [`problem_miner/`](problem_miner/) — `config.py` (settings), `sources/` (data-source abstraction), `clean.py`/`split.py`/`embed.py`/`cluster.py`/`categorize.py`/`label.py` (pipeline stages), `pipeline.py` (orchestration), `results.py` (structured output, no ranking applied), `cli.py` (entry point).

## Running it yourself

```bash
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt_tab')"   # one-time
# requires a local Ollama server running with a model pulled, e.g.:
#   ollama pull mistral-nemo:12b

python -m problem_miner \
  --source jsonl \
  --reviews-file synthetic_data/aria7_reviews.jsonl \
  --book-id synthetic-aria7-b1 \
  --output output/results.json
```

Runs entirely locally — no API keys, nothing sent externally. The
Watchmen/real-data path (`--source goodreads`) needs the UCSD dataset
downloaded separately (not committed to the repo — see `CLAUDE.md` for
the download URLs and license terms); the synthetic Aria-7 path above
needs nothing beyond what's already in the repo.

## Results on real data (Watchmen, 1,757 Goodreads reviews — technical validation)

This is where the debugging story above happened. Real, unfiltered Goodreads
reviews, used for internal validation only (see the licensing section above
for why this isn't what runs as the shown demo).

- 771 reviews passed the filter → 5,523 individual sentences
- 73 clusters found (1,832 sentences didn't fit any cluster densely
  enough — treated as noise, not forced into a false grouping)
- Categorized: **25 complaint clusters** (1,192 sentences), **26 praise
  clusters** (1,577 sentences), **18 informative clusters** (754
  sentences), **4 junk clusters** (168 sentences — bare interjections
  like "No." or "4." with no real content)

Two examples of what a cluster actually looks like:

> **Complaint cluster (artwork), 29 sentences, 83% negative:**
> "I found the art to distractingly poor." · "Didn't like the artwork."
> · "The artwork was a bit 'flat' for my liking."

> **Informative cluster (content note), 15 sentences:**
> "At first I thought I could read this one with my 11-year-old son, but
> it soon became apparent that this is really an adult story." ·
> "Definitely not for little kids, but a fascinating read for adults"

That second one is worth noting on its own: it's not a complaint and
not praise, it's a genuinely useful thing a prospective reader would
want to know. The tool was originally scoped as "complaints only";
finding this cluster in the real output was the reason to deliberately
broaden that scope rather than discard it.

## Results on the demo dataset (synthetic reviews, cleanly licensed)

275 AI-generated reviews of a real, fully-owned manuscript — small
enough that results are noticeably coarser than the Watchmen run above
(smaller dataset, fewer clusters), which is an accepted trade-off for
having something legally clean to actually show. This is also where
cluster labeling ran for the first time:

- 206 reviews passed the filter → 424 sentences → 13 clusters (145
  sentences landed as noise)
- Two findings from the Watchmen run **independently reappeared** here,
  on a completely different and synthetic dataset — good evidence
  they're real properties of the method, not one-off quirks of one
  dataset: a cluster of reviewers doing rating-justification math
  ("solid 3 stars, good but not amazing") that reads as a topic more
  than a sentiment, and a cluster mixing genuine praise and genuine
  complaint about the same specific topic

Three labeled clusters, title + summary generated by a local model, not
hand-written:

> **TITLE: Repetitive Internal Monologue**
> Reviewers found the frequent use of "she filed/catalogued/processed"
> phrases tiresome and distracting.

> **TITLE: Trap Engineering Pacing**
> Readers find trap engineering sections engaging but overlengthy.

> **TITLE: Cat Mystery's Slow Burn**
> Reviewers praise ARIA-7 Book One's cat subplot as a well-paced and
> satisfying mystery.

One honest caveat, confirmed directly rather than just anticipated:
synthetic review text reads measurably cleaner than the real Watchmen
data — less noise, fewer of the genuinely weird edge cases real internet
text produces. Disclosed as a known limitation of this dataset, not
hidden — see the licensing section above for the reasoning behind
accepting that trade-off anyway.

## Tech stack

- **Package:** `problem_miner/` — installable, CLI-driven, source-agnostic (see "Running it yourself" above)
- **Datasets:** [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) (Goodreads reviews scraped by Julian McAuley's lab) for internal technical validation only — academic use only, not redistributed, not what's shown as the demo (see "A licensing decision" above); a synthetic, fully-owned dataset for the actual demo-facing results.
- **Embeddings:** `sentence-transformers` / `all-mpnet-base-v2`
- **Dimensionality reduction:** `umap-learn`
- **Clustering:** `HDBSCAN` (via `scikit-learn`)
- **Sentiment classification:** `transformers` / `cardiffnlp/twitter-roberta-base-sentiment-latest`, plus a targeted local-LLM re-check (see case study #2)
- **Cluster labeling:** local Ollama (`mistral-nemo:12b`) for title + summary generation
- **Text processing:** `nltk`, `langdetect`
- Runs entirely locally (NVIDIA RTX 5080) — no data sent to any external API, for either the ML pipeline or the labeling step.

## What's next

Real remaining gaps, not just hypothetical polish:
- A `negative→neutral` sentiment-recheck bucket (167 sentences in
  testing) is a genuine mixed bag — some real corrections, some real
  regressions — not yet resolved, and probably related to the
  still-deferred compound-sentence limitation (see `CLAUDE.md`).
- No automated tests yet — validation so far has been manual
  spot-checking against known-good numbers, which was rigorous but
  isn't the same as a regression-proof test suite.
- Only two data sources exist (real Goodreads, synthetic JSONL) —
  the abstraction supports more, nothing else has been built or tested.

Longer-term/exploratory:
- A separate, more fundamentals-focused project is planned to
  demonstrate ML/AI understanding at a lower level than "use the
  standard library well" — not started yet.
- Compound sentences, quoted-dialogue-as-opinion, and other known
  limitations in `CLAUDE.md` remain deliberately deferred — real,
  documented, not blocking.

See [`CLAUDE.md`](CLAUDE.md) for full working notes and
[`devlog.md`](devlog.md) for the complete decision history.
