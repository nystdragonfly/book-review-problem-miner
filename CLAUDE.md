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
6. Labels each cluster (LLM-generated title + one-sentence summary — see
   "Where things stand"; TF-IDF keyword tags were tried and dropped).
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

**Done (2026-08-20): switched away from the UCSD Book Graph dataset for
anything demo-facing, via the hybrid option.** Real-data work on
Watchmen stays as-is (genuinely academic/exploratory, nothing
redistributed, and it's the source of the k-means → HDBSCAN → UMAP
debugging story in the README). Went looking for a purchasable/reusable
CC0 dataset first — found none trustworthy (see devlog: the one "CC0"
Amazon-reviews dataset found traces back to the same restricted McAuley
academic source with a mislabeled license from a third-party re-upload;
Amazon-Reviews-2023 has no stated license at all, itself a yellow flag).
Considered self-scraping and rejected it — would likely be a *worse*
legal position than the original problem, personally committing a ToS
violation specifically to build the commercial-facing artifact.

User's own independent research, worth recording: found paid dataset
providers that likely have usable review data, but two real problems —
(1) confirming they didn't themselves scrape against the source
platform's terms would need real diligence, not just trusting the
seller's claim; (2) legitimate-seeming ones start around $800/month for
access to a single dataset, not viable for an independent operator/
freelancer just starting out (as opposed to a company with budget for
compliant data licensing). Cheaper alternatives showed clear signs of
being sourced via questionable scraping themselves. Reinforces the
synthetic-data decision with a sharper reason than "no free clean option
exists": the legitimate paid option isn't viable for who this project
is actually for.

Landed on synthetic data, using a real manuscript the user wrote (`ARIA-7, Book
One`, in `book-source/`) as the subject — see the new "Synthetic
dataset" entry below. Real-data testing isn't abandoned going forward
either — the licensing issue was specifically about what's shown/
distributed, not about internal validation, which stays fine.

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

**Synthetic dataset (`synthetic_data/aria7_reviews.jsonl`)** — 275
AI-written reviews of `ARIA-7, Book One` (the user's own manuscript,
`book-source/ARIA-7-Book-One.pdf`), fully owned, zero licensing risk.
Rating spread `{0:10, 1:38, 2:43, 3:59, 4:87, 5:38}`. Deliberately
written with realistic noise, not polished: typos, informal grammar, a
few non-English reviews, short junk-style reactions, occasional
rating/text mismatches (e.g. rating 3 with review_text "5") — the kind
of thing a human reviewer actually does, initially "cleaned up" by
mistake and restored after feedback that it read as more authentic
left alone. Genuinely written from having read the manuscript (opening,
middle, and ending), not generic filler — themes include the
Thomas/ARIA relationship, the repetitive "she filed it/catalogued it/
processed it" narration style, the cat-pride subplot, genre-mismatch
complaints (readers expecting stat-heavy LitRPG getting something more
literary instead), and reactions to the cliffhanger ending.
**Caveat, stated plainly, not hidden:** this is a weaker test than real
data by construction — the user's own read after seeing the clustered
output was "I can see how the synthetic data reads cleaner than real
data would." Kept as a known, disclosed limitation rather than a solved
problem — see README.
- Pipeline run on this dataset (`scratch/09_aria7_full_pipeline.py`):
  206 cleaned reviews → 424 sentences → 13 clusters (145 noise points).
  Notably, two Watchmen-specific findings **independently reproduced**
  on this completely different (and synthetic) dataset — strong evidence
  they're real properties of the method, not one-off quirks: (1) a
  cluster of rating-justification meta-commentary ("solid 3 stars...",
  "3 stars for concept, 1 for pacing") showing the same "topic, not
  sentiment" mislabeling as Watchmen's cluster 3; (2) a cluster (trap-
  engineering) mixing genuine praise and genuine complaint about the
  same specific topic, same as Watchmen's artwork cluster before finer
  `min_cluster_size` tuning.
- **Cluster labeling (pipeline step 6) — done**, via
  `scratch/10_cluster_labeling.py`: LLM-generated title + one-sentence
  summary per cluster, using a **local Ollama server already running on
  this machine** (`localhost:11434`, model `mistral-nemo:12b` — the same
  model already loaded for the user's personal AI setup; `llama3.1:8b`
  and `mistral:7b` also available). Chosen over the Anthropic API
  specifically to keep the pipeline local/free, consistent with every
  other model used so far. Real bug hit and fixed: the model prefixes
  labels with markdown bold (`**TITLE:**`) despite being told not to —
  parsing must strip markdown before matching, or it silently fails.
  Also tried TF-IDF keyword tags alongside the LLM title/summary,
  specifically to compare and decide — **dropped after comparing real
  output**: inconsistent value, sometimes added a nuance the title
  missed, often just redundant/weak. Verdict: LLM title + summary alone
  is sufficient; keyword tags added noise, not value.

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
  - **Concrete mechanism found (2026-08-20):** a chunk of the junk-
    category bare-number sentences ("2.", "3.") trace back to numbered
    lists inside otherwise substantive reviews (e.g. "A few more notes
    regarding the movie: 1. I actually liked Adrian more in the film. 2.
    Jackie Earle Haley..."), not people typing a bare rating as their
    whole review — `nltk.sent_tokenize` doesn't understand list markers
    and either orphans them as standalone fragments or glues them onto
    the end of an unrelated sentence. Verified by tracing review_ids
    back to the original full review text, not just assumed. The actual
    list *content* survives fine as clean separate sentences — only the
    bare markers are noise. Corrects an earlier wrong guess (that these
    were literally people typing just a star number). **Not treated as
    a new problem to fix** — it's a specific instance of the
    context-free-sentence limitation immediately above, already
    accepted as a known trade-off of sentence-level splitting.
- **Quoted book dialogue gets treated as reviewer opinion.** Discovered
  via sentiment labeling: one cluster (Watchmen cluster 13) turned out to
  be reviewers quoting the book's own dialogue back ("She's screaming
  'No!'"), not expressing their own view. Not caught by the junk-length
  heuristic (this cluster averages 7.5 words — well above the junk
  threshold). No fix attempted; would need something like quote-mark
  detection or checking if a sentence's language style matches Watchmen's
  in-story dialogue vs. reviewer prose.
- **Sentiment classification misreads rating-justification meta-commentary.**
  A cluster of sentences where reviewers explain/justify their *numeric*
  star rating (e.g. "I gave it only four stars because I felt the
  romance... was not very well done", "I wanted to give it 1 star, but
  gave it 2 stars instead") got tagged NEGATIVE overall — but the
  sentiment breakdown was a near-perfect three-way tie (32 negative / 32
  neutral / 23 positive), and negative only won by the plurality
  tiebreak. The model has no concept of the number itself as a verdict —
  it reads the surrounding hedging/critique language ("only", "because...
  wasn't very well done") and skews negative even when the stated rating
  is actually high (4-4.5 stars). This is a topic cluster (meta-
  commentary about rating justification), not a true sentiment cluster —
  worth checking whether other clusters with near-tied sentiment
  breakdowns have the same "topic, not sentiment" mislabeling problem.
- **RESOLVED (partially), 2026-08-20: sentiment model systematically
  misreads negated praise.** Started from a user-spotted example: "The
  trap-engineering stuff scratched an itch I didn't know I had." (a
  4★ review) classified negative. Investigated with real data instead
  of guessing:
  - Quantified: 9.3% of 4★ Aria-7 sentences and 23.1% of 4★ Watchmen
    sentences get classified negative. Reading a sample, every one was
    genuinely positive, sharing one pattern — praise phrased as a
    subverted/negated expectation ("didn't expect to care", "no stat
    screens, no XP bars, just..."). The model keys on the negation word
    rather than the sentence's actual meaning. This is a known general
    weakness in sentiment analysis, not unique to this pipeline or
    model — confirms/sharpens the earlier vague "suspiciously even
    3-way split" suspicion with a real mechanism.
  - Tested the hypothesis that a general-purpose local LLM (already
    running for cluster labeling) would do better: **mixed result, not
    a clean win.** On the negated-praise set the LLM got 4/8 right
    (small classifier got 0/8) but hedged the rest to "neutral" rather
    than getting the direction right. On a control set of genuinely
    negative terse dismissals ("Not for me.") the LLM did *worse*
    (0/4, vs. the small classifier's 4/4) — it needs more context to
    read short text correctly, and even added context only partially
    fixed this (2/4).
  - A word-count heuristic to route only "probably-negated-praise"
    candidates to the LLM looked clean on a small hand-picked sample
    (11-25 words vs. 3-5 words) but **did not hold up at scale** —
    checked against the full 445-candidate set, median length was 16
    words either way. Correctly abandoned rather than shipped on a
    small-sample coincidence.
  - Landed on: run the LLM re-check on the full candidate set (negative
    + contains a negation marker — see `categorize.py`), but **only
    auto-apply the negative→positive disagreement direction** (16/445
    candidates, spot-checked as essentially all genuine corrections).
    negative→neutral disagreements (167/445) were deliberately left
    alone — spot-checking that bucket found a real mix of genuine
    corrections (hedged/compound sentences like "Good, not great.")
    and real regressions (terse dismissals wrongly softened), not safe
    to auto-apply without per-sentence judgment.
  - Implemented in `problem_miner/categorize.py`
    (`_recheck_negation_candidates`, gated by
    `config.enable_negation_recheck`). Verified the original reported
    sentence now classifies positive.
  - **Still open:** the negative→neutral bucket is real, uninvestigated
    further, and the compound-sentence overlap noticed there ("Good,
    not great.", "Prose is nice but the plot doesn't move much") ties
    back to the still-deferred compound-sentence limitation above —
    the two problems may be more related than previously treated.

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
- A local Ollama server is already running on this machine
  (`localhost:11434`) for the user's personal AI setup — `mistral-nemo:12b`,
  `llama3.1:8b`, and `mistral:7b` available. Use this instead of a paid
  API for any generative-LLM step (e.g. cluster labeling) when the
  content being sent isn't the restricted real dataset — keeps things
  local/free and consistent with the rest of the pipeline.
