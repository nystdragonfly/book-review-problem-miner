# Keeping Projects Separate in Claude Code — Reference Guide

A working reference for avoiding project bleed-together, especially useful while building out portfolio projects for freelance work.

---

## 1. One Folder, One Session, Per Project

**The rule:** Every project — portfolio piece, freelance gig, personal project — gets its own folder and its own git repo. Never work on two projects in the same open Claude Code session.

**Folder location (Ubuntu):** `~/prof_projects/` — kept separate from personal projects, one subfolder per project (e.g. `~/prof_projects/ai-ml-demo/`).

**How to do it:**
- Before starting new work, ask: *"Am I opening a folder that already belongs to a different project?"* If yes, stop and open the right one.
- Close out a session fully (or open a fresh window/tab) before switching projects. Don't just keep typing into the same running session about a different thing.
- If you catch yourself mid-session realizing you've drifted into a different project's territory, stop, note where you left off, and restart clean in the right folder.

**Why it matters:** Context in a session carries forward — code patterns, assumptions, half-finished ideas. If two projects share a session, that context leaks between them, which is likely the biggest source of the "bleeding together" you noticed.

---

## 2. Explicit Scope File (CLAUDE.md) Per Project

**The rule:** Every project folder gets a short `CLAUDE.md` file at its root, stating clearly what the project is, what it is *not*, and its current status.

**Template to reuse:**

```markdown
# Project: [Name]

## What this is
[1-2 sentences — the actual scope of this project]

## What this is NOT
[Explicitly rule out scope creep — e.g. "not a full app, just a script,"
"not connected to the game project," "no UI needed"]

## Current status
[e.g. "Just started," "core logic done, needs testing,"
"finished, polishing README"]

## Notes
[Anything else worth remembering next session]
```

**How to use it:**
- Create this file *before* the first real work session on a project, not after.
- Update "Current status" at the end of each session (ties into practice #3 below).
- If a session with me starts drifting off-scope, you can point back at this file and say "stay inside this."

**Why it matters:** This gives me an explicit boundary to check against, instead of me guessing or improvising based on whatever's freshest in context. It directly targets the "wrong direction for an hour" problem — a clear scope file makes it easier to catch drift early instead of an hour in.

---

## 3. Session Start/End Ritual

**The rule:** Say the project and the goal out loud (or type it) at the start of a session. Note what's done and what's left at the end.

**Start of session — say or type something like:**
> "Working on [project name] today. Goal: [specific thing you want done by end of session]."

**End of session — even one line:**
> "Stopping here. Done: [what got finished]. Left: [what's next]."

**Why it matters:**
- Starting: forces a clear boundary before work begins, so neither of us drifts.
- Ending: makes picking back up next time fast and clean — no re-explaining, no guessing what state things are in. This matters especially with your rotation-based work style (long focused stretches, then switching) since the gap between sessions on the same project could be days.

---

## 4. Tool Permissions for Professional Projects

**The rule:** Personal projects can run with mostly unrestricted tool access. Professional/client-facing projects (like anything in `~/prof_projects/`) should run with stricter permissions, since mistakes there carry more real-world weight (client trust, licensing exposure, accidental data leaks).

**Starter config** (`settings.json`, place in `.claude/settings.json` at the root of `~/prof_projects/` or per-project):

```json
{
  "permissions": {
    "deny": [
      "Read(./.env)",
      "Read(./**/secrets/**)",
      "Read(**/*credentials*)"
    ],
    "ask": [
      "Bash(git push:*)",
      "Bash(git commit:*)",
      "Bash(rm -rf:*)",
      "Bash(curl:*)",
      "Bash(wget:*)",
      "Bash(pip install:*)"
    ]
  }
}
```

**What this does:**
- **Secrets stay off-limits** — `.env`, credentials, and secrets files can't be read by Claude at all.
- **Destructive/network/install actions pause for confirmation** — git commits and pushes, forceful deletes, network fetches, and new package installs all require a yes/no before running, instead of happening silently.
- **Everything else stays smooth** — reading, editing, exploring, and running tests aren't gated, so this doesn't slow down normal work.

**Why it matters for professional work specifically:**
- **Secrets/credentials:** client trust depends on nothing leaking. Note: a deny rule on the `Read` tool blocks Claude's built-in file reading, but doesn't stop a Bash subprocess like `cat .env` — for stronger enforcement, OS-level sandboxing (Claude Code's sandbox feature) closes that gap too.
- **Network/data access:** scraping sites (e.g. Goodreads, Amazon) often violates their Terms of Service. Prefer existing public datasets built for this purpose over live scraping, and keep network calls gated behind a confirmation so this doesn't happen by default.
- **Git/destructive ops:** you want to be the one deciding what gets committed/pushed to a professional repo, not have it happen automatically.
- **Package licensing:** watch for copyleft licenses (GPL, AGPL) in dependencies — they can force you to open-source code that touches them, which is a real problem if the code is meant for a client. MIT/Apache/BSD-licensed packages are generally safe.

---

## 5. Tools vs. Dependencies vs. Assets — Three Different License Questions

**The distinction:** using a free tool to create your work is not the same as your work depending on licensed code, and neither is the same as using a "free for personal use" asset in paid work. Three separate questions, worth keeping straight:

**1. Tools you work in (editors, IDEs, creative software)**
Always safe to use for professional work, regardless of the tool's own license — you're not redistributing the tool itself.
- Examples: GIMP (GPL), PyCharm Community (Apache 2.0), Godot (MIT), VS Code, Blender — all fine for commercial use.

**2. Libraries/dependencies your code imports and ships with**
The license can follow your code if you redistribute it. Watch for copyleft (GPL, AGPL) in anything your project actually imports — permissive licenses (MIT, Apache, BSD) are safe.

**3. "Free for personal/non-commercial use" assets and models**
A genuinely separate, restricted category — free to use, but explicitly barred from commercial/paid work by license terms:
- Fonts (many free fonts online are personal-use-only)
- Icon packs, stock assets, sound effects (some free tiers restrict commercial use)
- Pretrained ML models/datasets — some carry "research only" or "non-commercial" license terms, distinct from open-source ones. Worth checking directly relevant to portfolio/freelance ML work.
- Freemium software with Terms of Service restricting commercial use on the free tier

**Practical habit:** before using any free asset, font, model, or dataset in something meant to earn money, check its license/terms for "non-commercial," "personal use," or "research only." Plain MIT/Apache/BSD/GPL (with the redistribution caveat above) is fine. No stated license at all is a yellow flag — find the source's actual terms rather than assuming.

---

## Quick Checklist (Once This Becomes Habit)

- [ ] New project → new folder → new repo → new session
- [ ] CLAUDE.md written before first real work session
- [ ] Session opens with stated project + goal
- [ ] Session closes with a done/left-off note
- [ ] CLAUDE.md status updated to match
- [ ] Professional project? → `.claude/settings.json` permissions in place before real work starts
- [ ] Using a free asset/font/model/dataset? → check for "non-commercial"/"personal use"/"research only" terms first

---

*Built as part of the freelance portfolio project plan — August 2026.*
