# Book Review Problem-Miner

A tool that reads through a book's reviews and surfaces the *specific
problems readers report* — grouped into themes, not buried in a pile of
star ratings. Point it at a book; it tells you "here's what people
actually complain about," "here's what they praise," and "here's what
you should know before picking it up" — instead of a single blended
4.2★ average that tells you nothing about *why*.

**Status: exploratory prototype.** The full pipeline is validated
end-to-end on one real book (see below), not yet generalized to work on
arbitrary books or built out as production code. See
[`devlog.md`](devlog.md) for the complete session-by-session history.

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

## How it works

| Step | What | Why this choice |
|---|---|---|
| Filter | Drop 5★ and unrated reviews | Complaints concentrate in 1–4★; 5★ reviews are low-signal/promotional |
| Clean | Normalize whitespace, drop non-English (`langdetect`) | Raw scraped text has formatting artifacts; mixed-language embeddings cluster poorly together |
| Split | Sentence-level (`nltk`), not whole-review | A single review often mixes praise and complaint — sentence granularity keeps them separable |
| Embed | `sentence-transformers` (`all-mpnet-base-v2`), GPU-accelerated | Captures meaning, not keywords — see case study above for why that matters concretely |
| Cluster | UMAP (dimensionality reduction) → HDBSCAN | See case study — the straightforward approach (k-means) provably lost real themes |
| Categorize | Per-sentence sentiment classification (not review star rating) → negative / positive / neutral, plus a junk filter | A sentence's own text is classified directly, since one review's rating doesn't apply evenly to all its sentences |

## Results on a real book (Watchmen, 1,757 Goodreads reviews)

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

## Tech stack

- **Dataset:** [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) (Goodreads reviews scraped by Julian McAuley's lab) — academic use only, not redistributed, downloaded fresh via the scripts noted in `CLAUDE.md`.
- **Embeddings:** `sentence-transformers` / `all-mpnet-base-v2`
- **Dimensionality reduction:** `umap-learn`
- **Clustering:** `HDBSCAN` (via `scikit-learn`)
- **Sentiment classification:** `transformers` / `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Text processing:** `nltk`, `langdetect`
- Runs GPU-accelerated locally (NVIDIA RTX 5080) — no data sent to any external API.

## What's next

- Give each cluster an actual title/summary instead of raw example
  sentences (currently the weakest part of the output).
- Generalize beyond a single hardcoded book.
- Turn the validated `scratch/` exploration scripts into real,
  reusable pipeline code.
- A separate, more fundamentals-focused project is planned to
  demonstrate ML/AI understanding at a lower level than "use the
  standard library well" — not started yet.

See [`CLAUDE.md`](CLAUDE.md) for full working notes and
[`devlog.md`](devlog.md) for the complete decision history.
