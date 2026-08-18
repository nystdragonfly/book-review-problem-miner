# Dev Log

Session-by-session narrative of what happened and why. Newest entry on
top. For "what's true right now," see `CLAUDE.md` — this file is the
history/reasoning behind how it got that way.

---

## 2026-08-17 — Project kickoff + pipeline validated end-to-end on Watchmen

**Goal for the session:** write the scope file, find a dataset, and (this
went further than planned) validate the whole complaint-mining pipeline
on one real book before generalizing to "any book."

### What got done

1. Wrote `CLAUDE.md`.
2. Picked a dataset: **UCSD Book Graph** (Goodreads reviews). Downloaded
   the Comics & Graphic Novels genre slice (542K reviews, no login/API
   key needed, academic-use-only license). Picked **Watchmen**
   (`book_id 472331`, 771 usable non-5★ reviews) as the validation case.
3. Set up `.claude/settings.json` (deny reads on secrets/credentials, ask
   before push/commit/rm -rf/curl/wget/pip install) and saved a reusable
   template to `~/prof_projects/settings-template.json` for future
   projects.
4. Set up PyCharm against the existing `venv` and fixed a path bug in the
   scratch scripts so they run correctly regardless of working directory.
5. Built and validated the pipeline step by step in `scratch/`
   (disposable exploration code, not real pipeline code — see scripts
   list at the bottom):
   - **Clean:** normalize whitespace, drop non-English reviews
     (`langdetect`).
   - **Split:** into sentences (`nltk.sent_tokenize`).
   - **Embed:** `sentence-transformers` (`all-mpnet-base-v2`), on the
     RTX 5080 — 5,523 sentences encode in under 2 seconds.
   - **Cluster:** UMAP (768→5 dims) + HDBSCAN. See decision log below —
     this took two failed attempts (k-means, then plain HDBSCAN) before
     landing here.
   - **Categorize:** each cluster labeled negative (complaint) / positive
     (praise) / neutral (informative) / junk, via per-sentence sentiment
     classification + a word-count-based junk filter.

### Key decisions and why

- **k-means → UMAP+HDBSCAN.** K-means forces every sentence into one of
  k clusters. A real but rare theme — artwork complaints, ~29 of 5,523
  sentences — kept getting absorbed into a much larger "general negative
  sentiment" cluster instead of forming its own, because k-means has no
  concept of "too sparse to be its own cluster." Confirmed empirically by
  tracing 6 known artwork-complaint sentences and seeing exactly where
  they landed. Tried HDBSCAN (density-based, allows "noise" instead of
  forcing every point into a cluster) as a fix — but running it directly
  on the raw 768-dimensional embeddings failed too: either one giant blob
  or ~100% noise depending on `min_cluster_size`, because density
  estimates get unreliable in high-dimensional spaces (curse of
  dimensionality). The fix was reducing to 5 dimensions with UMAP first —
  the standard recipe used by the BERTopic library — which then let
  HDBSCAN correctly split artwork complaints from artwork praise as two
  separate clusters at `min_cluster_size=10`.
- **Sentiment labeling uses each sentence's own text, not its source
  review's star rating.** A 3★ review contains both praise and complaint
  sentences; using the review rating as a sentiment proxy would mislabel
  individual sentences. Used a dedicated 3-class sentiment model instead
  (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
- **Added a "junk" category.** Some clusters had no real content
  ("No." ×5, "4." ×3, "Really?" ×5). Checked empirically: junk clusters
  average 1.3–2.4 words/sentence, and the next-lowest *legitimate*
  cluster averages 5.0 words — clean gap, so used avg-words-per-cluster
  < 4 as the junk threshold rather than picking a number blind.
- **Scope broadened from "complaints only" to complaint/praise/
  informative.** A cluster turned up that was genuinely useful but not a
  complaint: reviewers flagging Watchmen as adult content ("not for my
  11-year-old"). Decided this is worth keeping rather than discarding,
  since "what should I know before reading this" is a more complete
  product than "complaints only" — but this is a real departure from the
  scope written in `CLAUDE.md` on day one, flagged there rather than
  drifted past silently.
- **Product framing note:** this is more naturally a buyer-side/internal
  tool than something a seller hands to their own customers — it makes a
  mediocre product's specific flaws much more legible than a star
  average does, which most sellers wouldn't want to hand their own
  customers. Framed instead as an internal quality-improvement tool.

### Known limitations carried forward (full detail in `CLAUDE.md`)

- Compound sentences ("...left me confused, and I didn't get as much out
  of the artwork...") blend two complaints into one embedding — ~22% of
  sentences. No fix attempted; rule-based conjunction-splitting was
  considered and rejected as too fragile.
- Quoted book dialogue sometimes gets clustered as if it were the
  reviewer's own opinion (discovered via sentiment labeling — a
  Watchmen cluster turned out to be quoted lines like "She's screaming
  'No!'"). Not caught by the junk-length filter since it's 7.5 words on
  average, well above that threshold.
- The sentiment model is trained on tweets, not book-review prose — the
  overall 3-way split across all sentences came out suspiciously even
  (1848/1838/1837). Individual cluster labels looked sensible on
  inspection, so not blocking on it, but worth a second look later.

### Next steps (pick up here)

1. Cluster **labeling**: give each cluster an actual title/summary
   instead of just showing 5 raw example sentences (pipeline step 6).
2. **Generalize beyond Watchmen** — parameterize the scratch logic by
   `book_id` instead of hardcoding it.
3. **Turn validated scratch/ logic into real pipeline code.** Everything
   so far is deliberately disposable exploration script — nothing has
   been "promoted" into an actual reusable pipeline yet.

### Scratch scripts as of this session

- `common.py` — shared loading/cleaning/embedding-cache helpers used by
  everything below.
- `01_inspect_raw_text.py` — eyeball raw review text for noise patterns.
- `02_clean_and_split.py` — whitespace normalize, language filter,
  sentence splitting.
- `03_embed_sentences.py` — first embeddings sanity check (nearest-
  neighbor search; confirmed semantic generalization beyond literal
  keyword overlap).
- `04_cluster_sentences.py` — k-means attempt. **Superseded** — kept for
  reference/comparison, not the validated approach.
- `05_trace_artwork_complaints.py` — diagnostic script tracing where
  known artwork complaints landed under k-means; motivated the pivot to
  HDBSCAN.
- `06_hdbscan.py` — plain HDBSCAN on raw embeddings. **Failed** — kept
  for reference, shows why UMAP is needed first.
- `07_umap_hdbscan.py` — UMAP + HDBSCAN. **The validated clustering
  approach.**
- `08_sentiment_labels.py` — adds sentiment + junk categorization on top
  of `07`. **Current end state of the exploration.**

### Later in the session: portfolio framing + AI-collaboration honesty

Committed the above as the initial commit (`34dd0ee`), then kept going
on planning rather than stopping.

Raised a real question: does this project actually demonstrate ML/AI
understanding, given most of today was "Claude proposes a tool + explains
why, I approve or redirect"? Worked through it honestly rather than
brushing it off:

- Distinguished *knowing a tool exists* from *recognizing when output is
  wrong and directing a fix* — the second is what actually happened
  (e.g. noticing the artwork-complaint cluster was missing after already
  seeing the signal existed via the nearest-neighbor check).
- Best evidence understanding actually transferred, not just got
  demonstrated: unprompted, recognized that a personal project's "topic
  memory" feature likely has the same k-means-style flaw (rare themes
  getting absorbed into big generic clusters) and is considering applying
  UMAP+HDBSCAN there. That's independent pattern transfer to a different
  system, not just following along here.
- Bigger question underneath: since this project is meant to prove
  hireable skill (per `project-boundaries-guide_1.md`'s "freelance
  portfolio" framing), what's actually being sold if AI wrote most of the
  code? Landed on: the sellable skill isn't "typed the code unaided," 
  it's catching wrong output before it ships, making the product/scope
  judgment calls AI won't make unprompted, and verifying rather than
  trusting results (GPU actually computing, not just detected; junk
  threshold checked against real examples, not guessed) — decided to be
  upfront about AI-assisted development rather than obscure it.

**Wrote `README.md`** — portfolio-facing (different audience than
`CLAUDE.md`), with the k-means → HDBSCAN → UMAP+HDBSCAN debugging journey
as the centerpiece "here's what the collaboration actually produced"
evidence, an honest "built in collaboration with AI" section, and results
pulled from the real Watchmen run.

**Flagged a licensing problem with using this dataset for portfolio
purposes**, not just today's exploration: UCSD Book Graph is licensed
"academic use only... should not be used for commercial purposes" — in
real tension with a project meant to attract paid work. Fine for the
technical validation already done (genuinely academic/exploratory, not
redistributed), but decided this shouldn't be what a portfolio demo runs
on going forward. Options logged in `CLAUDE.md`, not decided which one
yet — next-session item.
