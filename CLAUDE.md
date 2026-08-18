# Book Review Problem-Miner

## What this is

A tool that takes book reviews and surfaces the *problems readers report*, grouped
into themes — not a sentiment classifier, not a summarizer of praise.

Given reviews for a book, the pipeline:

1. Filters OUT 5-star reviews (treated as low-signal / promotional — the
   assumption is that complaints live in the 1–4 star range).
2. Cleans/preprocesses the remaining review text.
3. Extracts complaint-relevant snippets from each review.
4. Embeds snippets as sentence vectors.
5. Clusters the embeddings (UMAP + HDBSCAN — see "Where things stand") to
   find recurring themes.
6. Labels each cluster (top keywords + representative examples).
7. Outputs the top N complaint themes for a given book.

Explicitly **not** goals: general sentiment analysis, positive-review
summarization, star-rating prediction, or per-review classification as an
end product.

> **Note (2026-08-17):** in practice we ended up classifying each cluster's
> sentiment (negative/positive/neutral) as part of deciding which clusters
> are actual complaints — see "Where things stand." That's sentiment
> classification *in service of* complaint-mining (labeling themes, not
> scoring reviews), but it's close enough to the thing ruled out above
> that it's worth flagging rather than quietly drifting past it. Revisit
> this framing if it keeps feeling contradictory.

## Where things stand

As of 2026-08-17: pipeline approach validated end-to-end on one book
(Watchmen) via disposable `scratch/` scripts — see `devlog.md` for the
full session narrative and reasoning. Nothing is real pipeline code yet
(see "Working conventions"). `README.md` is the portfolio-facing writeup
(different audience than this file — written for a client/hiring
manager, not for me) covering the same ground with the debugging story
as the centerpiece.

**Idea, not started:** a separate, more fundamentals-focused project to
demonstrate ML/AI understanding at a lower level than "use
sentence-transformers/UMAP/HDBSCAN well" — this project is real evidence
of judgment and tool-use skill (see README), but not of from-scratch
algorithmic understanding the way the past neural-net-from-scratch
project was. Explicitly deferred to a future session, not scoped yet.

**Decided, not yet done: switch away from the UCSD Book Graph dataset.**
Its license says "academic use only... should not be used for commercial
purposes" — a real conflict with this project's purpose (demonstrating
skills to attract paid/professional work), not just optics. Fine for the
technical validation work done so far (genuinely academic/exploratory,
nothing redistributed), but shouldn't be what a portfolio demo runs on
going forward. Options considered: (1) find a dataset with actually
permissive licensing (CC0/CC-BY/explicit commercial terms) — needs real
verification before trusting any specific one, don't repeat the mistake
of assuming a license is fine without checking; (2) synthetic/generated
reviews for the demo-facing artifact specifically — zero licensing risk,
and the debugging story (k-means → HDBSCAN → UMAP) doesn't actually
depend on the data being real; (3) hybrid — keep current dataset for
internal validation, swap only what gets shown to people. Not decided
which option yet.

Dataset chosen: **UCSD Book Graph** (Goodreads reviews, scraped by Julian
McAuley / Mengting Wan's group, UCSD). Using the Comics & Graphic Novels
genre slice for now:
- `data/goodreads_reviews_comics_graphic.json.gz` — 542,338 reviews, 89,311
  unique books. Fields: `user_id`, `book_id`, `review_id`, `rating` (0–5,
  where 0 = no rating given), `review_text`, `date_added`, `date_updated`,
  `read_at`, `started_at`, `n_votes`, `n_comments`.
- `data/goodreads_books_comics_graphic.json.gz` — matching book metadata
  (title, author, average_rating, etc.), keyed by `book_id`.
- Both downloaded directly from `mcauleylab.ucsd.edu` (no login/API key
  needed). **Academic use only per the dataset's license — do not
  redistribute or use commercially.** Not committed to git (see `data/` in
  `.gitignore`); re-download via the URLs above if the directory is missing.
- Verified `Watchmen` (book_id `472331`) as a good demo case: 1,757 reviews,
  771 usable non-5-star reviews with real text (>50 chars) — plenty of
  signal for clustering.
- Note: rating `0` means "shelved but not rated" — filter these out
  alongside 5-star reviews in preprocessing, they're not 1-star complaints.
- The full (non-genre-split) dataset is ~15GB / 15.7M reviews if the
  project later needs more than one genre; other genre slices (romance,
  mystery/thriller/crime, YA, etc.) are available at the same host under
  `byGenre/`.

Pipeline approach validated on Watchmen via `scratch/` scripts (exploratory,
not real pipeline code yet — see "Working conventions"):
- Clean (whitespace normalize + drop non-English via `langdetect`) → split
  into sentences (`nltk.sent_tokenize`) → embed with `sentence-transformers`
  (`all-mpnet-base-v2`, GPU) → cluster.
- Clustering: **UMAP (768→5 dims) + HDBSCAN**, not plain k-means. K-means was
  tried first but forces every point into one of k clusters — a real but
  rare theme (artwork complaints, ~29 sentences out of 5,523) always got
  absorbed into a larger "general negative sentiment" cluster instead of
  forming its own. Plain HDBSCAN on the raw 768-dim embeddings also failed
  (curse of dimensionality — density estimates are unreliable at that many
  dimensions), producing either one giant blob or almost all noise. UMAP
  dimensionality reduction first (the standard BERTopic recipe) fixed this:
  at `min_cluster_size=10`, HDBSCAN correctly separated a 29-sentence
  artwork-complaints cluster from a separate artwork-praise cluster — real
  sentiment-level separation within a topic, not just topic-level grouping.
- Clusters aren't exclusively complaints — many are praise, character
  discussion, or reader-background commentary ("this is my first graphic
  novel"). **Scope decision: broadened beyond "complaints only."** A
  genuinely useful cluster turned up flagging Watchmen as adult content
  ("not for kids") — not a complaint, not praise, but clearly useful to
  a prospective reader. Decided to categorize every cluster as
  negative (complaint) / positive (praise) / neutral (informative)
  rather than discarding non-complaints, since that's a more complete,
  more useful tool. This is a deliberate departure from the original
  "complaints only" framing at the top of this file — worth revisiting
  that framing if it causes confusion later.
- Sentiment categorization: classify each **sentence's own text** with a
  dedicated 3-class model (`cardiffnlp/twitter-roberta-base-sentiment-latest`,
  positive/negative/neutral), not the source review's star rating — a 3★
  review contains both praise and complaint sentences, so the review's
  rating is a noisy proxy for any single sentence's sentiment.
- Added a 4th **junk** category on top of the three sentiment ones, for
  clusters with no real content ("No.", "4.", "Really?"). Detected via
  average word count per cluster — junk clusters average 1.3–2.4
  words/sentence; the next-lowest legitimate cluster averages 5.0 —
  clean gap, threshold set at <4 words.
- On Watchmen: 73 clusters / 3,691 clustered sentences (1,832 sentences
  landed as HDBSCAN noise) → 25 negative, 26 positive, 18 neutral, 4 junk
  clusters (1192 / 1577 / 754 / 168 sentences respectively).
- Next real step: give each cluster a title/summary (not just top-5
  example sentences) — this is pipeline step 6, not yet built.

## Who's working on this

Self-taught, comfortable with Python. Has built a neural net from scratch,
but this is the first project involving embeddings and clustering — those
are new territory. Explanations of *why* a library/method is chosen (not
just usage) are wanted throughout, especially at each new technique.

## Hardware

Desktop, not laptop — no need to be stingy with compute:
- AMD Ryzen 9 9950X (16C/32T), 64GB RAM, NVIDIA RTX 5080 (16GB VRAM), 4TB disk.
- Ubuntu 24.04 LTS, kernel 6.17, X11.
- GPU-accelerated embedding models are on the table (e.g. sentence-transformers
  with CUDA); no strong need to reach for the smallest/quantized model by default.

## Known limitations (deliberately deferred, not forgotten)

- **Compound sentences blend multiple complaints into one embedding.** e.g.
  "The juxtaposition of several storylines left me confused, and I didn't
  get as much out of the artwork..." carries two separate complaints
  (narrative confusion + artwork) but gets encoded as one vector, so
  k-means can only assign it to one cluster — the other complaint it
  carries doesn't fully register. ~22% of Watchmen sentences look
  compound (contain " but " or ", and "). A rule-based split on
  conjunctions was considered and rejected — too fragile (e.g. "great
  story and art" isn't two complaints). A real fix would need dependency
  parsing (spaCy) to find clause boundaries properly. Deferred until we
  see whether it visibly muddies real cluster output.
- **Short/context-free sentences carry no signal or are ambiguous.** e.g.
  "It was difficult to read." could be about prose or artwork with zero
  textual signal to disambiguate, since sentence-level splitting throws
  away surrounding context — still unresolved. The "outright junk" half
  of this is now partially handled: junk sentences tend to cluster
  together (short interjections/numbers are semantically similar to each
  other), so a per-*cluster* avg-word-count check catches them without
  needing a per-sentence length cutoff or URL stripping. Doesn't help a
  junk sentence that happens to land in a real cluster, but resolves the
  common case.
- **Quoted book dialogue gets treated as reviewer opinion.** Discovered
  via sentiment labeling: one cluster (Watchmen cluster 13) turned out to
  be reviewers quoting the book's own dialogue back ("She's screaming
  'No!'"), not expressing their own view. Not caught by the junk-length
  heuristic (this cluster averages 7.5 words — well above the junk
  threshold). No fix attempted; would need something like quote-mark
  detection or checking if a sentence's language style matches Watchmen's
  in-story dialogue vs. reviewer prose.
- **Sentiment model may not be well-calibrated for book-review prose.**
  `cardiffnlp/twitter-roberta-base-sentiment-latest` is trained on tweets.
  The overall 3-way sentiment split across all 5,523 Watchmen sentences
  came out suspiciously even (1848 neutral / 1838 positive / 1837
  negative) — could be a real property of mixed reviews, or could be the
  model not discriminating well on longer/more nuanced prose than it was
  trained on. Individual cluster labels looked sensible on inspection
  (e.g. 95% negative for an "unlikable characters" cluster), so not
  blocking on this, but worth a closer look later — maybe compare against
  a model trained on longer-form review text specifically.

## Working conventions

- Python, using the `venv/` in this repo (Python 3.12.3, currently empty —
  install packages as they're actually needed rather than front-loading a
  big requirements list).
- Prefer explaining trade-offs before locking in a library/method choice
  (e.g. which embedding model, which clustering algorithm) rather than
  silently picking one.
- Keep exploratory/dataset work (notebooks, scratch scripts, sample data)
  clearly separated from pipeline code as the project grows — exact
  structure TBD, don't over-engineer folders before there's real code.
- Large datasets should NOT be committed to git — download scripts /
  instructions instead, with the data directory gitignored.
- `.claude/settings.json` is configured (deny reads on `.env`/secrets/
  credentials; ask before `git push`/`git commit`/`rm -rf`/`curl`/`wget`/
  `pip install`). A reusable copy for future professional projects lives
  at `~/prof_projects/settings-template.json` — copy it into a new
  project's `.claude/settings.json` manually, nothing applies
  automatically.
- PyCharm is set up with the interpreter pointed at this repo's `venv`
  (not a separate PyCharm-managed one). Scripts under `scratch/` resolve
  their own paths via `Path(__file__)` rather than assuming a working
  directory, so they run correctly whether launched from a terminal or
  from PyCharm's per-script default working directory.
- For anything substantial, prefer running scripts directly (terminal or
  PyCharm) over having output pasted into chat — full output, not a
  curated excerpt.
