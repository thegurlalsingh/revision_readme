# AI INTERN INTERVIEW — MASTER PREPARATION
**For: Gurlal Singh | Interview: Tomorrow**

> Built from your actual resume, the live READMEs of `cv_tailor`, `botfolio`, `creator_discovery_automation`, and `scholarship_extractor` on GitHub, and the fully verified IEEE Xplore listing for your paper (Section 11).

---

## 1. Interview Strategy

You have **4 real, working, non-trivial full-stack + AI projects** and a co-authored IEEE paper. That's a genuinely strong intern profile — the risk isn't "did you build it," it's **can you justify every architectural choice under pressure**. Interviewers go deepest where resumes use big words: "multi-agent," "autonomous," "eliminated hallucinations," "scalable." Expect the most pressure on:

1. **CV Tailor** — multi-agent orchestration (LangGraph) is your most sophisticated AI claim → expect the deepest drilling here.
2. **Botfolio** — real-time video/audio/speech pipeline + proctoring → systems-design heavy.
3. **Creator Discovery** — scraping + engagement math + LLM classification → practical/ethical questions (Instagram ToS, rate limits).
4. **Scholarship Extractor** — anti-hallucination design, confidence scoring → this is actually your *most defensible* AI-engineering story because it's deterministic-first with clear guardrails.
5. **Research paper** — now fully verified against the actual IEEE Xplore text (Section 11): a 40-combination grid search (5 DL models × 2 optimizers × 4 facial-feature inputs) for ASD detection, ViT+full-face+Adam hitting 90% accuracy. Know the numbers and the stated limitations cold — they're your own paper's honest self-critique, which is exactly what a "defend your research" question is looking for.

**Golden rule for every answer:** WHAT → HOW → WHY → ALTERNATIVES → TRADE-OFFS. Never stop at "what."

---

## 2. Resume Breakdown — Claim Risk Map

| Claim | Risk | Why | What you need to defend it |
|---|---|---|---|
| "Autonomous multi-agent AI platform" (CV Tailor) | 🔴 High | "Autonomous" and "multi-agent" are buzzwords interviewers test hardest | Exact LangGraph node structure, state passing, what "autonomous" actually means here (loop until approved, not truly unsupervised) |
| "Eliminating hallucinated content" (CV Tailor) | 🔴 High | Absolute claims ("eliminating") invite "how do you *know* you eliminated it, not just reduced it?" | Be ready to soften verbally to "the Auditor agent flags and forces rewrites when it detects unsupported claims" — describe the *mechanism*, not the marketing verb |
| "Improving evaluation consistency" (Botfolio) | 🟠 Medium | Vague metric — no number given | Explain qualitatively: structured rubric (Relevance/Clarity/Confidence) vs. ad-hoc grading |
| "Contributed directly to a paper accepted at an IEEE conference" | 🟠 Medium | Interviewer may ask for exact contribution % and paper title/link | Know precisely what *you* did (benchmarking, grid search, documentation) vs. co-authors |
| "600+ LeetCode, 1500+ rating" | 🟢 Low | Verifiable, believable | Have 1-2 recent problems ready to discuss casually |
| "88th/650+ Smart BU Hackathon" | 🟢 Low | Believable, specific | Know what you built there in one sentence, in case asked |
| "Improving classification accuracy" (AI Research Intern) | 🟠 Medium | No number → "by how much?" | Be honest: this was comparative/benchmarking work across architectures (VGG-style/EfficientNet-class CNNs are typical for this literature), not a single model you trained from scratch alone — say so if that's true |
| "Deterministic-first extraction... minimizing hallucinated data" (Scholarship Extractor) | 🟢 Low | Actually your *strongest* claim — the README shows a real evidence-based scoring rubric | This is your best "did you actually think about AI safety/reliability" answer — lean into it |
| "280+ targeted search queries" (Scholarship Extractor) | 🟢 Low | Specific, verifiable number | Know roughly how the query set is constructed (site: filters × category × state, per the README) |

---

## 3. "Tell Me About Yourself" / Intro Questions

### Q: "Tell me about yourself."
**Strong answer (natural, ~45–60 sec):**
"I'm a final-year-track CS undergrad at Bennett University specializing in AI. Over the last year I've focused on building agentic AI systems end-to-end — not just calling an LLM API, but orchestrating multiple specialized agents that check each other's work. My flagship project is CV Tailor, where five agents — a strategist, writer, auditor, ATS evaluator, and revision planner — iterate on a resume until it's both truthful and optimized against a job description. I've also built a full recruitment platform with live video interviews and speech evaluation, an influencer-outreach automation tool, and a scholarship-discovery pipeline that's built specifically to avoid hallucinating facts. Alongside that, I spent a summer as an AI research intern benchmarking deep learning architectures for autism detection, which became a paper accepted at an IEEE conference. I like problems where AI has to be *reliable*, not just impressive in a demo."

**Follow-up:** "Which of those projects would you want to keep working on after this internship, and why?"
**Follow-up answer:** Pick one honestly (CV Tailor is the best choice — it's your most technically ambitious) and give one concrete next step you'd build (e.g., moving LangGraph execution to a Celery queue, per your own README roadmap).

### Q: "Why AI/ML specifically?"
**Strong answer:** Ground it in something concrete from your work — e.g., you noticed most "AI features" people build are single-prompt wrappers, and you got interested in the harder problem of making LLM outputs *verifiable* (auditor agents, deterministic-first extraction, confidence scoring) rather than just fluent.

### Q: "Which project was hardest?"
**Strong answer:** Botfolio, honestly — because it isn't just LLM calls, it's a real-time media pipeline (webcam capture → GCP Storage → ffmpeg audio extraction → Speech-to-Text → LLM grading) with proctoring/security guards layered on top, and getting that whole chain to work reliably end-to-end (state guards for pipeline steps, subprocess integration between Node and Python) was the most "distributed systems"-flavored problem you've solved.

### Q: "What did you personally contribute?" (if asked generally)
Be ready to speak in first person about specific pieces (the agent prompts, the SSE streaming layer, the DB schema, the scraping/engagement logic) rather than "we built." If any project had collaborators, know exactly your slice.

---

## 4. PROJECT 1 — CV Tailor (Deepest prep needed)

**What it is (from your own README):** A full-stack, multi-agent AI system (React/Vite frontend, FastAPI backend) that rewrites a resume against a job description using a **stateful LangGraph pipeline**: Strategist → Writer → Auditor → ATS Evaluator → Revision Planner, looping back to Writer until approved.

### Architecture Q&A

**Q: Walk me through what happens when a user submits a resume + JD.**
Strong answer: "The frontend uploads the resume PDF and JD text via two POST endpoints. The backend uses PyMuPDF to extract raw text from the PDF, parses both resume and JD into structured JSON (stored in Postgres via SQLAlchemy), then the frontend opens a Server-Sent Events connection to `/api/tailor/run`. That triggers the LangGraph graph: the Strategist compares resume vs JD and builds a 3-point plan, the Writer rewrites the resume JSON per that plan, the Auditor fact-checks the draft against the *original* resume to catch fabricated skills, the ATS Evaluator scores it like a real ATS, and the Revision Planner decides whether to loop back to the Writer or approve. Every state transition streams a log line and a UI summary block over SSE so the user watches the agents 'think' live. On approval, the backend compiles final JSON into a PDF (WeasyPrint/PDFKit), uploads it to Supabase, and returns a download URL."

**Follow-up: Why LangGraph over just chaining LangChain calls yourself?**
Strong answer: "LangGraph gives you an explicit state machine with conditional edges — the Planner→Writer loop-back is a cycle, which is awkward to express in a simple linear chain. LangGraph also makes the state (draft, audit notes, ATS score) a first-class typed object passed between nodes, instead of me hand-rolling a loop with prompt-stuffed history, which gets messy and token-expensive fast."

**Follow-up: Why not a single mega-prompt that does strategy+writing+auditing in one shot?**
Strong answer: "Single-prompt approaches conflate generation and verification — the same model call that writes the content is a bad judge of whether it hallucinated something, because it has no adversarial incentive to catch its own mistakes. Splitting Writer and Auditor into separate calls with different system prompts and different framing (Auditor is explicitly told to compare against the *original* source of truth) gives you a much stronger check, similar to why generator/discriminator or self-consistency setups outperform single-pass generation for factuality."

### WHY Questions / Trade-offs

- **Why FastAPI over Django/Flask?** Async-native, Pydantic validation for the JSON schemas moving between agents, auto OpenAPI docs — a good fit for an API that's mostly I/O-bound (LLM calls, DB, Supabase uploads).
- **Why PostgreSQL + SQLAlchemy over a NoSQL store?** The domain is genuinely relational — users → resumes → JDs → tailored_resume, with foreign keys tying a tailored output back to *both* its source resume and target JD. You want referential integrity here, not schema flexibility.
- **Why SSE instead of WebSockets for streaming agent logs?** SSE is one-directional (server→client), which is all you need here — the client isn't sending data mid-stream, just listening. Simpler than WS: plain HTTP, auto-reconnect in browsers, no separate protocol upgrade handling.
- **Why JWT over sessions?** Stateless auth fits a FastAPI/React SPA better than server-side session storage; scales horizontally without sticky sessions.
- **What happens if the Auditor keeps rejecting the Writer's draft?** *(You should decide/know your actual max-loop behavior — if your graph has no cap, this is a real bug to acknowledge: "Currently the loop could in theory run indefinitely; a production version needs a max-iteration guard with a graceful degrade to 'best draft so far.'")* This is a great "what would you improve" answer — say it proactively if asked about weaknesses.
- **What if the OpenAI API call fails mid-pipeline?** Be honest about current state per your own roadmap (no mention of retry/backoff in the README) — a good answer: "Right now a failure would break the SSE stream; the fix is wrapping each agent node call in a retry-with-backoff and surfacing a partial-failure state to the frontend rather than silently dying."

### Deep Attack Tree — CV Tailor
```
Why LangGraph over manual orchestration?
        ↓
Why does the Planner node need to exist separately from the Auditor?
        ↓
What's actually stored in the LangGraph "state" object between nodes?
        ↓
What happens if two revision loops in a row don't improve the ATS score?
        ↓
How would you cap infinite loops in production?
        ↓
How would you scale this to 1,000 concurrent tailoring runs?
        ↓ (→ your own README roadmap: Celery + Redis + RabbitMQ workers)
What would you cache to cut cost/latency? (→ parsed JD/resume schemas, per repeated JDs)
```

### System Design (from your own roadmap — use it, it's honest and shows self-awareness)
- **Bottleneck today:** the LangGraph execution runs synchronously inside the FastAPI request/SSE connection → long-running HTTP connections don't scale.
- **Your own fix (stated in README):** move execution to Celery workers with Redis/RabbitMQ brokers, add a Redis cache for repeated LLM calls on the same JD/resume pair, and eventually a vector DB (Pinecone/Weaviate) so agents can pull a user's *other* past achievements that didn't make it into the current resume.
- **Say this out loud** if asked "how would you scale it" — it demonstrates you've actually thought about production, not just demo-ware.

---

## 5. PROJECT 2 — Botfolio (AI Recruitment Platform)

**What it is:** Node.js/Express + Prisma/PostgreSQL backend, React frontend, 5-stage sequential candidate pipeline (Info → MCQ → Video → Coding → Completed), each stage gated by a `currentStep` field in the DB.

### Architecture Q&A

**Q: Walk me through the video interview stage specifically — it's your most technically involved piece.**
Strong answer: "The frontend uses the browser's MediaRecorder API to capture webcam video for each of 5 AI-generated questions, uploads the raw `.webm` to GCP Storage. A Python subprocess (`video_to_audio.py`) uses ffmpeg to extract the audio track to MP3. That's sent to Google Cloud Speech-to-Text for transcription, and the transcript is graded by an LLM (via OpenRouter, using a Gemini/Claude-class model) on three axes — Relevance, Clarity, Confidence — with per-step and aggregate scores stored back in Postgres via Prisma."

**Follow-up: Why call Python as a subprocess from a Node.js backend instead of doing everything in one language?**
Strong answer: "ffmpeg tooling and a lot of media/ML tooling has better-supported Python bindings, and I didn't want to rewrite PDF-parsing/audio-extraction logic in JS when mature Python libraries already exist. The trade-off is process-spawn overhead and weaker type safety across the language boundary — for a larger system I'd wrap that in a small dedicated Python microservice with its own API instead of shelling out from Express."

**Follow-up: Why Prisma over raw SQL or another ORM?**
Strong answer: Type-safe query builder, migrations, and schema-as-code that's easy to reason about for a schema this relational (User → ExperienceTimeline/EducationTimeline/JobDescription/McqAttempt→McqQuestion/VideoInterview→VideoInterviewStep/CodingAttempt) — the generated Prisma client also gives you compile-time safety in a JS codebase that otherwise has none.

**Follow-up: Why JDoodle instead of building your own code sandbox?**
Strong answer: Building a secure, multi-language (JS/Python/C++/Java) sandboxed execution environment yourself means solving process isolation, resource limits, and security hardening — that's a huge scope increase for what's fundamentally a "run this code against test cases" feature. JDoodle offloads that risk. Trade-off: vendor dependency, rate limits, and less control over execution environment/timeouts.

### WHY / Trade-offs
- **Why step-guards (`StepGuard.jsx`) instead of trusting the frontend routing?** Client-side guards are for UX; actual security has to be server-enforced too — say explicitly whether your backend *also* validates `currentStep` before accepting a submission for a stage (this is an important honesty check — if it's only frontend-enforced today, say "the frontend blocks navigation, but the source of truth and real enforcement should also happen server-side on every mutating endpoint, which is something I'd want to double check/harden").
- **Why proctoring measures (fullscreen enforcement, clipboard blocking, dev-tools blocking)?** Basic integrity signal for an assessment platform — but be ready for "how would a determined candidate bypass this?" (Answer honestly: client-side JS checks are fundamentally beatable — a second monitor, a phone, or disabling JS defeats most of them; real proctoring needs server-side signals too, like webcam presence checks, which is a good "what I'd add" answer.)
- **Why rate limiting on auth endpoints specifically?** Brute-force/credential-stuffing protection — standard hardening for any exposed login endpoint.

### Attack Tree — Botfolio
```
Why OpenRouter (multi-model gateway) instead of calling OpenAI directly?
        ↓
Why does that matter for a recruitment platform specifically?
        ↓ (cost control / model flexibility per task — MCQ gen vs. grading vs. code feedback might benefit from different models)
How do you keep MCQ generation *consistent* with a candidate's actual resume/skills?
        ↓
What happens if the LLM generates an MCQ with no correct answer, or an ambiguous one?
        ↓
How would you validate/QA AI-generated assessment content at scale?
        ↓
What happens if Speech-to-Text mis-transcribes a technical answer, unfairly lowering a score?
        ↓
How would you add a human-review safety net without breaking the "automated" pitch?
```

### System Design
- **Scale to 1,000 concurrent candidates:** the video pipeline (ffmpeg subprocess + Speech-to-Text + LLM grading) is the obvious bottleneck — this is CPU/IO heavy per submission. You'd move this off the request path into a job queue (same Celery/BullMQ-style pattern as CV Tailor) so uploads are accepted immediately and graded asynchronously, with the frontend polling/websocket-notified on completion.
- **What would you cache?** Nothing about a *specific* interview response is cacheable, but generated MCQ question banks per (role, skill-set) combination could be cached/reused instead of regenerating from scratch every time — real cost lever.

---

## 6. PROJECT 3 — Creator Discovery Automation

**What it is:** FastAPI + Playwright backend, React frontend. Discovers Instagram micro-influencers via the Socialcrawl API, scrapes profile + latest 5 reels with Playwright, computes engagement rate, classifies niche via an LLM (OpenRouter), and generates personalized cold emails/DMs, storing everything in Supabase Postgres, sending via Resend.

### Architecture Q&A

**Q: Walk me through the full pipeline.**
Strong answer: "User submits campaign keywords + target creator count from the React panel. FastAPI calls the Socialcrawl API for keyword- or similarity-based creator discovery, producing candidate handles. Playwright then opens each profile, scrapes username/bio/follower count/verification/contact email, and filters to a 5K–100K follower range to focus on micro-influencers. For creators that pass the filter, Playwright also scrapes their latest 5 reels — captions, likes, comments — which lets me compute engagement rate as `(likes+comments)/followers × 100`. Bio + reel captions go to an LLM which classifies category and content themes, and that context feeds a second LLM call that drafts a personalized cold email (60–90 words) and a shorter Instagram DM (15–30 words), grounded in the creator's actual content rather than a generic template. Everything is stored across three related tables — creators, reels, outreach — and emails are sent through Resend with status tracked as generated → sending → sent/failed."

**Follow-up: Why Playwright over the official Instagram API?**
Strong answer: Instagram's official Graph API is heavily gated (requires a Business/Creator account, app review, and doesn't expose the kind of open discovery/scraping this workflow needs). Playwright automates a real browser, so it can read what a logged-in user session sees. **Be ready for the honest follow-up: this is fragile and against Instagram's Terms of Service** — a good, mature answer: "This was built as a learning/portfolio project to explore browser automation and LLM personalization pipelines; a production version would need to go through Meta's official Graph API/Partner program or a licensed data provider like a creator-marketing platform's API, both for reliability (Playwright scraping breaks whenever Instagram changes its DOM) and for ToS compliance."

**Follow-up: Why compute engagement rate yourself instead of relying on Socialcrawl's data?**
Strong answer: Socialcrawl gives discovery/candidate handles, but the engagement number needs to be computed from *fresh* content (latest 5 reels) at the time of the campaign — using a stale, cached engagement number from a third-party discovery API would be less accurate and wouldn't let you tie the personalization directly to specific recent posts you can reference in outreach copy.

**Follow-up: Why filter 5K–100K followers specifically?**
Strong answer: That's the accepted range for "micro-influencers" — large enough to have a real audience, small enough that engagement rate tends to be meaningfully higher than mega-influencers, and importantly, small enough that they're more likely to respond personally to outreach (a macro-influencer/celebrity account won't).

### WHY / Trade-offs
- **Why store `contact_email = "Not Found"` instead of skipping the record?** Keeps the creator in the dataset for manual/Instagram-DM outreach even when email scraping fails (a real limitation, honestly documented in your own README) — better than silently dropping potentially good creators.
- **Why Supabase/Postgres over a simpler flat file or NoSQL store?** Same reasoning as CV Tailor — genuinely relational data (`creators 1—N reels`, `creators 1—1 outreach`), and you get Postgres's querying power for the dashboard (filter/sort by engagement, category, status).
- **Why generate *both* an email and a shorter DM instead of one message reused everywhere?** Different channels have different norms — email tolerates more formal structure and length, Instagram DM culture is casual/short. Reusing one message on both channels reads as generic/spammy on the platform where it's used the "wrong" length.

### Attack Tree — Creator Discovery
```
Why an LLM for niche classification instead of a rules-based keyword matcher?
        ↓
Why might that be risky? (hallucinated/inconsistent categories across runs)
        ↓
How would you validate the LLM's category labels are consistent?
        ↓
What happens if Instagram changes its page structure and Playwright selectors break?
        ↓
How would you detect that failure quickly instead of silently returning empty data?
        ↓
What's your plan for ToS/legal risk at production scale?
        ↓
How would you redesign this around an official API/licensed data provider?
```

### System Design / Failure Cases
- **What if Instagram rate-limits/blocks the scraping IP?** Not addressed in current implementation — good honest answer: "would need IP rotation/proxy pooling and randomized delays, but at that point you're fighting an anti-bot system, which reinforces why an official API is the right long-term path."
- **What's the biggest bottleneck at scale?** Playwright browser automation is inherently slow and sequential-ish per profile — real headless-browser scraping doesn't parallelize cheaply (memory/CPU per browser instance). You'd want a worker pool with a capped concurrency and possibly a lighter-weight HTTP-based scraper for creators where DOM interaction isn't required.

---

## 7. PROJECT 4 — Scholarship Extractor (your strongest "AI reliability" story)

**What it is:** Python/FastAPI backend, discovers scholarships via 280+ targeted DuckDuckGo search queries (site:gov.in, site:ac.in etc.), crawls with Requests+BeautifulSoup, extracts with a **deterministic-first, LLM-fallback** engine, scores confidence via an explicit evidence-based rubric (0–100), stores in Supabase Postgres, and continuously rechecks via `pg_cron`/`pg_net` for change/staleness detection. React dashboard on top.

### Architecture Q&A

**Q: Walk me through discovery → verification.**
Strong answer: "Discovery runs 280+ targeted search queries combining site-restriction operators (gov.in, ac.in, edu.in, org.in) with scholarship-related terms, spanning government/university/CSR/NGO source types. Results are deduplicated by URL and classified into GOVERNMENT/UNIVERSITY/CORPORATE/NGO-TRUST/INTERNATIONAL/OTHER by domain and page signals. A recursive crawler pulls HTML and PDF text with SSL/retry handling. Extraction is deterministic-first: I try to pull each field — title, amount, eligibility, deadlines, documents — with rule-based parsing first, and only fall back to an LLM call when deterministic extraction can't find a field. Anything still missing is stored as `NULL`, never guessed. Each record then gets scored against a 100-point evidence rubric — official domain (30pts), reachability (20pts), guidelines page present (15pts), etc. — and anything ≥95 is marked VERIFIED, otherwise REVIEW REQUIRED."

**Follow-up: Why deterministic-first instead of just using an LLM to extract everything — wouldn't that be simpler?**
Strong answer: "An LLM extracting every field from raw HTML/PDF text is more flexible but also more prone to *inventing* plausible-sounding values when a field is genuinely absent from the page — which is exactly the failure mode I'm trying to avoid for something students will rely on (a scholarship deadline or eligibility criterion). Deterministic parsing (looking for structured patterns, labeled sections, tables) is more brittle across differently-formatted sites, but when it *does* find a value, it's guaranteed to be grounded in the actual text — no hallucination risk. The LLM is only invoked as a fallback for the harder unstructured cases, and even then it's constrained to the actual crawled source text, never asked to fill gaps from its own knowledge."

**Follow-up: Why is confidence scoring rule-based instead of asking the LLM 'how confident are you'?**
Strong answer: "LLM self-reported confidence is notoriously uncalibrated — models will say '95% confident' about something wrong just as easily as something right. A deterministic rubric based on *external, checkable evidence* (is this an official domain? is the source reachable right now? is there a guidelines/FAQ page?) gives a score that's actually auditable — and the dashboard exposes each individual check, so a human reviewer can see *why* a score landed where it did, not just trust a black-box number."

**Follow-up: Why pg_cron/pg_net inside Supabase instead of an external cron service (e.g., a separate worker + Celery beat)?**
Strong answer: Keeps the scheduling colocated with the data it's operating on — no separate infra to run/monitor for a relatively lightweight periodic job, and Supabase's `pg_net` lets Postgres itself fire the HTTP POST to the FastAPI recheck endpoint. Trade-off: less flexible than a dedicated task queue for retry policies/backoff/complex scheduling, and it ties your scheduling logic to your specific DB provider.

### WHY / Trade-offs
- **Why "3 consecutive verification failures → NO_LONGER_VERIFIABLE" instead of removing the record immediately?** Avoids false negatives from a single transient failure (site down for maintenance, temporary SSL issue) — requires a pattern of failure before demoting a record's status.
- **Why keep full field-level change history (`scholarship_changes`) instead of just overwriting the current value?** Auditability — if a deadline silently changes, you want to know *when* it changed and what it used to be, both for trust and for debugging the pipeline itself.
- **Why free-tier tools (DuckDuckGo search, free OpenRouter model) specifically?** Constraint stated directly in your own requirements table — this was built to prove the pipeline works without paid APIs; a production version would likely swap in a paid search API for higher query volume/reliability and a stronger LLM for the fallback extraction cases.

### Attack Tree — Scholarship Extractor
```
Why deterministic-first extraction?
        ↓
Why not trust the LLM fallback fully once it's invoked?
        ↓ (LLM only sees the actual crawled text — anti-hallucination guardrail #3 in your README)
What happens when a scholarship page is JS-rendered and BeautifulSoup can't see the content?
        ↓ (Requests+BeautifulSoup can't execute JS — a real limitation vs. Playwright-based crawling)
How would you detect that failure mode (empty/near-empty extraction) automatically?
        ↓
How would you decide whether to add a headless-browser fallback for JS-heavy sites, given the cost/latency trade-off?
        ↓
How does the system know when NOT to trust its own confidence score?
```

### System Design
- **Biggest current gap:** Requests+BeautifulSoup crawling can't render JavaScript — any scholarship portal that loads content client-side would show up as an empty/low-confidence page. A production fix would add a headless-browser (Playwright) fallback specifically for pages that fail the initial static crawl, at the cost of latency/compute per page.
- **What would you change with another month?** Add the JS-rendering fallback, move discovery queries + crawling into a proper job queue for parallelism (280+ queries run serially today unless you already parallelize — check and be honest), and add a lightweight second LLM "critic" pass similar to CV Tailor's Auditor agent to sanity-check extracted fields before they're marked VERIFIED.

---

## 8. Cross-Project Questions

**Q: Which project has the best architecture, and why?**
Strong answer: "Scholarship Extractor, for reliability engineering — the deterministic-first/LLM-fallback split and the transparent evidence-based confidence score are a more disciplined pattern than 'call an LLM and trust it,' which is the trap a lot of AI projects fall into. CV Tailor is more *ambitious* architecturally (multi-agent, stateful graph), but Scholarship Extractor is more *production-minded* about the actual failure mode that matters in AI systems — hallucination."

**Q: Which would scale the worst, and why?**
Strong answer: "Creator Discovery — it depends on Playwright browser automation against a platform (Instagram) that actively resists scraping. Browser instances are expensive per-unit (memory/CPU), don't parallelize cheaply, and the whole approach is fragile against DOM changes and anti-bot detection. The others depend on APIs/LLMs that scale more predictably with money, not with evading detection."

**Q: Where did you use similar concepts differently across projects?**
Strong answer: "LLM-as-fallback appears twice with opposite emphasis — CV Tailor's Auditor agent is an LLM *checking* another LLM's output (verification via a second model call), while Scholarship Extractor's LLM fallback only fires when deterministic *non-LLM* parsing fails (verification via avoiding the LLM in the first place when possible). Both are trying to reduce hallucination risk, just from different ends: one adds an LLM as a guard, the other removes the LLM from the default path entirely."

**Q: Why different tech stacks across projects (FastAPI vs Node/Express, Prisma vs SQLAlchemy)?**
Strong answer: Botfolio used Node/Express/Prisma likely because the frontend-heavy real-time-state application (step guards, proctoring) benefited from a unified JS ecosystem; the AI-heavy projects (CV Tailor, Creator Discovery, Scholarship Extractor) used FastAPI/Python because that's where the ML/LLM tooling (LangGraph, LangChain, PyMuPDF, BeautifulSoup, Playwright's Python bindings) naturally lives. Be honest if the real reason was "I wanted to practice both stacks" — that's a perfectly fine answer too, framed as intentional breadth.

---

## 9. Technology Cross-Examination (rapid-fire per stack item)

| Tech | What you should be able to say |
|---|---|
| **LangGraph / LangChain** | LangChain = building blocks (prompts, chains, tool-calling); LangGraph = explicit stateful graph for multi-step/cyclical agent flows (needed for CV Tailor's revision loop) |
| **FastAPI** | Async Python framework, Pydantic-based validation, auto OpenAPI docs — used across CV Tailor, Creator Discovery, Scholarship Extractor |
| **SSE (Server-Sent Events)** | One-way server→client stream over plain HTTP; used to stream live agent logs in CV Tailor and crawler logs in Scholarship Extractor |
| **JWT** | Stateless auth token, used across CV Tailor and Botfolio for session-less authentication |
| **Playwright** | Headless-browser automation; used to log into/scrape Instagram profile+reel data in Creator Discovery |
| **PyMuPDF** | Python PDF parsing library — extracts raw text from uploaded resume PDFs in CV Tailor |
| **BeautifulSoup** | HTML parsing for Scholarship Extractor's crawler (static HTML only — can't execute JS, a real limitation) |
| **pg_cron / pg_net** | Postgres extensions (via Supabase) for in-database scheduled jobs and outbound HTTP calls — powers Scholarship Extractor's daily recheck |
| **Prisma** | Type-safe ORM/migration tool for Botfolio's Node backend |
| **SQLAlchemy** | Python ORM used in CV Tailor's FastAPI backend |
| **OpenRouter** | Multi-provider LLM gateway (lets you swap Gemini/Claude/free models without changing API integration code) — used in Botfolio, Creator Discovery, and Scholarship Extractor's fallback |
| **JDoodle** | Third-party sandboxed code execution API used for Botfolio's coding-challenge stage |
| **Google Cloud Speech-to-Text** | Transcribes candidate video-interview audio in Botfolio |
| **Resend** | Transactional email delivery API used for outreach in Creator Discovery |
| **Docker** | Containerization — listed for CV Tailor/Botfolio; be ready to say honestly whether you *deployed* with it or just containerized locally |

**A note on integrity:** if an interviewer asks about something on your skills list you've only briefly touched (e.g., NestJS, TensorFlow, if not clearly used in a specific project) — don't overclaim. Say plainly what you've used it for vs. what you've only studied. This scores better than bluffing.

---

## 10. AI/ML Fundamentals Most Relevant To You

Focus revision here — these map directly to your projects, don't waste time on CV/NLP internals you didn't use (no evidence of CNNs/transformers you personally trained beyond the benchmarking internship):

**Generative AI / LLM engineering (highest priority — this is your actual specialty):**
- **RAG** — you haven't fully implemented it yet (it's in CV Tailor's *roadmap*, not built) — say so honestly if asked "tell me about your RAG pipeline." Know the concept: chunk → embed → store in vector DB → retrieve top-k by similarity → stuff into prompt context.
- **Prompt engineering** — you've done this practically across all 4 projects (agent role prompts, extraction prompts, classification prompts). Be ready to describe one real prompt-design decision (e.g., constraining the Auditor to only see the original resume as ground truth).
- **Hallucination** — you have a *very* strong, concrete answer here from Scholarship Extractor: deterministic-first extraction + evidence-based scoring + "missing → NULL, never guessed" + LLM only sees actual source text.
- **Agents / multi-agent orchestration** — CV Tailor is your canonical example; know the difference between "agent" (LLM + role + can loop/decide) vs. "chain" (fixed sequence).
- **Temperature / top-k / top-p** — standard definitions: temperature scales logit randomness before sampling; top-k restricts sampling to k highest-probability tokens; top-p (nucleus) restricts to the smallest set of tokens whose cumulative probability ≥ p. Know these even if you didn't explicitly tune them — you likely used defaults, and it's fine to say that.
- **Structured output / function calling** — relevant to how CV Tailor and Botfolio get parsed JSON back from LLMs reliably (JSON-mode / schema-constrained prompting).
- **Context windows** — relevant to why CV Tailor's roadmap wants a vector DB (pulling in a user's *other* past achievements without stuffing everything into every prompt).

**Machine Learning fundamentals (for your research-internship questions):**
- Overfitting/underfitting, train/val/test split, cross-validation, precision/recall/F1, grid search hyperparameter tuning (you *explicitly* used this — know it cold: exhaustively searches a specified hyperparameter grid, evaluating each combination via cross-validation to find the best-performing configuration).
- Transfer learning — your paper used VGG16, ResNet152, MobileNet, Xception, and ViT, which in this kind of small-dataset (1,268 images/class) medical-imaging setup are almost always used via pretrained ImageNet weights fine-tuned on the target task, rather than trained from scratch. Confirm this is how you actually ran it before stating it as fact, but it's the standard, expected approach here and a very likely follow-up question ("did you train from scratch or fine-tune a pretrained model?").

---

## 11. Research Paper — Fully Verified

**✅ Confirmed via IEEE Xplore (full text reviewed):** https://ieeexplore.ieee.org/document/11383733

- **Title:** "Comparative Analysis of Various Deep Learning Models for Autism Detection Using Grid Search"
- **Published in:** 2025 Second International Conference on Pioneering Developments in Computer Science & Digital Technologies (IC2SDT)
- **Conference dates:** 04–06 December 2025, Delhi, India
- **Added to IEEE Xplore:** 20 February 2026
- **DOI:** 10.1109/IC2SDT68218.2025.11383733
- **Publisher:** IEEE
- **Authors:** Prashant K. Gupta, Bireshwar Dass Mazumdar, Shallu Sharma, **Gurlal Singh**
- **Contributions statement (in the paper itself):** *"All authors contributed equally to the work."* — this overrides my earlier speculation about author-order significance; you can state plainly that authorship was equal, no need to downplay your role relative to co-authors.

### What the paper actually does
- **Problem:** ASD (Autism Spectrum Disorder) detection from facial images using deep learning is usually treated as a "black box" — a single DL model is trained on full-face images and reported as "best," with no insight into *what* the model is actually learning from, and no systematic comparison across models/optimizers/features.
- **Core method — a grid search across three axes:**
  1. **5 DL models:** Vision Transformer (ViT), Xception, VGG16, ResNet152, MobileNet
  2. **2 optimizers:** Adam, Lion
  3. **4 facial-feature inputs:** Eyes, Nose, Lips, Full face (extracted via `dlib` facial landmarking)
  - Total combinations tested: **5 × 2 × 4 = 40**
- **Dataset:** A publicly available facial-image dataset of children, pre-labeled autistic / non-autistic. Split 80:10:10 train/valid/test. Training set: 1,268 images per class; validation: 50 per class; test: 150 per class. Images resized to 224×224. Augmentation used: random scaling (2–4×), flips, random rotation (15°–45°).
- **Framework:** PyTorch.
- **Novelty claim (their own words):** first work to systematically grid-search DL model × optimizer × facial-feature-region combinations for ASD detection, rather than reporting one "best" model on full-face images alone — explicitly framed as an attempt to partially "white-box" what these models rely on.

### Key results (know these numbers cold)
| Optimizer | Best result per feature |
|---|---|
| **Adam** | ViT was best across all 4 features: **82%** (Eyes), **77%** (Nose), **77%** (Lips), **90%** (Full face) |
| **Lion** | Eyes: ViT & Xception tied at **82%**; Nose: ResNet152 & MobileNet tied at **73%**; Lips: Xception **71%**; Full face: ViT **86.6%** |

**Headline takeaway:** No single model wins across every feature/optimizer combination — full-face input consistently outperformed any single facial region under both optimizers, meaning the models likely use cues beyond eyes/nose/lips alone. Eyes performed better than nose/lips, hinting gaze-related features may matter (flagged as future work — e.g. retina/eye-tracking).

### Stated limitations (from the paper's own Discussion section — use these verbatim, they're honest and show maturity)
- Dataset, while balanced, comes from a specific demographic setting — generalizability is uncertain.
- Labels were taken as ground truth without independent verification (no clustering/consensus check on the labels themselves).
- Uni-modal (images only) — multi-modal data (e.g. video) is suggested as a way to improve accuracy/generalizability in future work.
- Accuracy was the only evaluation metric used (no F1/precision/recall reported in the abstract/results discussed) — be ready for "why not F1, especially with a class-imbalance-sensitive medical task?" Honest answer: the dataset here is class-balanced by construction (equal autistic/non-autistic counts), which is part of why accuracy alone was used as the primary metric; a more rigorous follow-up could still add F1/precision/recall for completeness.

### Interview Q&A — Basic
**Q: What is your paper about, in one sentence?**
"We ran a systematic grid search — 5 deep learning models × 2 optimizers × 4 facial-feature regions, 40 combinations total — to see which combination best classifies children's facial images as autistic or non-autistic, instead of just reporting one 'best' model."

**Q: What's the main contribution?**
"Showing empirically that no single DL model or optimizer wins universally — performance depends on the interaction between model architecture, optimizer, and which part of the face is used as input — and using facial-region inputs (eyes/nose/lips vs. full face) as a step toward interpreting *what* these black-box models rely on."

### Interview Q&A — Technical
**Q: Why these 5 models specifically (ViT, Xception, VGG16, ResNet152, MobileNet)?**
Strong answer: "They span meaningfully different architecture families — ViT is transformer-based/attention-driven with no convolutional inductive bias; Xception uses depthwise-separable convolutions; VGG16 is a classic deep plain-CNN; ResNet152 adds residual/skip connections to train very deep networks; MobileNet is a lightweight, efficiency-oriented CNN. That spread lets the comparison say something about architecture *families*, not just tweak one architecture's hyperparameters."

**Q: Why Adam vs. Lion as the two optimizers?**
Strong answer: "Adam is the default, well-understood adaptive-learning-rate optimizer known for efficient convergence. Lion is a newer, more memory-efficient optimizer (from Google's symbolic-search work) known for strong generalization in some settings. Comparing a well-established default against a newer alternative tests whether optimizer choice interacts with model architecture — and the results show it does: Adam favored ViT everywhere, but Lion gave other architectures (Xception, ResNet152, MobileNet) a competitive edge."

**Q: Why extract eyes/nose/lips specifically instead of other facial regions?**
Strong answer: "These are well-established regions of interest in facial-analysis literature for expression/attention-related conditions, and they're reliably extractable via `dlib`'s facial landmark detector. The goal was to test whether the model's classification signal is concentrated in specific regions (which would help interpretability) versus needing the full face — and the answer was: full face still wins, meaning there's information beyond these three isolated regions that matters."

**Q: Why is accuracy the only metric reported?**
See "Stated limitations" above — dataset is class-balanced (equal autistic/non-autistic samples in train/val/test), which is the honest justification for leading with accuracy; acknowledge precision/recall/F1 would strengthen it further.

### Interview Q&A — Deep / Defense
**Q: "Isn't 'grid search across model+optimizer+feature' just a big hyperparameter sweep — what's actually novel versus prior comparative-DL-for-ASD papers?"**
Strong answer: "Prior work (cited in our literature review) mostly compares DL models on full-face images alone and crowns one winner. Our novelty is two-fold: first, treating the *facial region used as input* as itself a variable in the grid, which prior work doesn't do — that's the attempt to partially interpret what the black-box model relies on. Second, showing the optimizer isn't a neutral background choice — it interacts with which architecture wins, which undercuts the common practice of reporting 'model X is best' without controlling for optimizer."

**Q: "82% and 90% accuracy — is that actually good? How do you know it's not overfitting to a narrow dataset?"**
Strong answer: "We used data augmentation (random scaling, flips, rotation) specifically to reduce overfitting risk, and we report results on a held-out test split, not training accuracy. But I'll be honest about the limitation we stated ourselves: the dataset comes from a specific demographic setting, so generalization to other populations isn't established by this work alone — that's explicitly flagged as a limitation, not glossed over."

**Q: "Your own paper says 'all authors contributed equally' — as a student intern among senior co-authors, what did you specifically do day to day?"**
Answer this honestly and specifically from your own memory of the internship — e.g., which of the 40 combinations you personally ran, whether you built the `dlib` landmark-extraction pipeline, ran the PyTorch training loops, or handled the results/statistics writeup. This document can't know that part — you do. Don't let "equally contributed" become a vague dodge; have one concrete, specific thing you personally did.

**Q: "What would you change if you repeated this study?"**
Strong, paper-grounded answer: "Add F1/precision/recall alongside accuracy, verify the ground-truth labels independently (the paper notes we took the dataset's labels at face value), and test on a second, demographically different dataset to check whether the ViT/full-face result generalizes — all three are limitations we named ourselves in the Discussion section."

---

## 12. Resume Red Flags — Self-Check Before the Interview

| Claim | Why it invites scrutiny | Safer, defensible framing |
|---|---|---|
| "eliminating hallucinated content" (CV Tailor) | Absolute language | "the Auditor agent is designed to catch and force revision of unsupported claims" |
| "improving factual accuracy" | No number | Describe the mechanism (fact-check against original resume), not a metric you don't have |
| "improving evaluation consistency" (Botfolio) | No number | Describe the structured rubric replacing ad-hoc judgment |
| "accelerating recruiter time-to-decision at scale" | "at scale" implies real usage data you may not have | Frame as the *design intent* of the automation, not a measured outcome, unless you actually have usage data |
| "increased volunteer participation" (NSS) | Vague | Have a rough sense of before/after if asked, or reframe around specific initiatives you ran |

**General rule:** if a bullet has no number and you get asked "by how much," don't invent one. Say "I don't have a precise measured figure — here's the qualitative improvement and why I believe it helped."

---

## 13. "Did You Actually Build It?" — Rapid Comprehension Checks

Be ready for blunt, code-level questions like these (answer from genuine understanding, not memorized lines):

- In CV Tailor: *"What's actually in the LangGraph state object passed between agents?"* / *"What does the Auditor do if it can't find any hallucinations — does it just pass through?"*
- In Botfolio: *"What field in the DB actually gates whether a candidate can access the coding round?"* (→ `currentStep` on `User`) / *"What happens if the ffmpeg subprocess crashes mid-extraction?"*
- In Creator Discovery: *"How exactly do you compute engagement rate — walk me through the formula with real numbers"* (→ `(likes+comments)/followers × 100`) / *"What does the `outreach` table's foreign key relationship to `creators` actually look like — one-to-one or one-to-many?"* (→ one-to-one)
- In Scholarship Extractor: *"Walk me through the exact point breakdown of the confidence score"* (→ title present 10, official domain 30, reachable 20, guidelines 15, FAQ 10, terminology 10, valid dates 5 = 100) / *"What happens to a record after 3 consecutive verification failures?"* (→ marked `NO_LONGER_VERIFIABLE`)

These are the kind of exact-detail questions that separate "built it" from "described it."

---

## 14. System Design — Cross-Cutting Answers

**"How would you handle 1M users across your busiest project (CV Tailor)?"**
1. Move LangGraph execution off the request thread into Celery/RQ workers.
2. Redis cache for repeated LLM calls (same JD text hit by multiple users, or re-runs).
3. Horizontal-scale FastAPI behind a load balancer; Postgres read replicas for the dashboard/history queries.
4. Move PDF compilation (WeasyPrint) to a background worker too — it's CPU-bound.
5. Rate-limit per user to control LLM cost, with a queue + estimated-wait UI instead of blocking.

**"What would you monitor?"** LLM call latency/cost per agent node, Auditor rejection rate (proxy for Writer quality), SSE connection drop rate, end-to-end pipeline completion time, DB query latency on the hot paths.

---

## 15. Rapid-Fire Round (10–30 sec answers)

- **RAG:** Retrieve relevant chunks from an external knowledge store via embedding similarity, then inject them into the LLM prompt as context before generation.
- **Embedding:** A dense vector representation of text (or other data) such that semantically similar inputs are close together in vector space.
- **Cosine similarity:** Measures the angle between two vectors, not their magnitude — common for comparing embeddings since it's scale-invariant.
- **Hallucination:** An LLM generating fluent but factually unsupported or fabricated content.
- **Temperature:** Sampling parameter controlling randomness — higher = more diverse/less deterministic output.
- **Function/tool calling:** Structured mechanism where the LLM outputs a schema-conforming call to an external function instead of free text, so the app can execute it deterministically.
- **Agent:** An LLM given a role, tools, and the ability to decide/loop, rather than a single fixed prompt→response.
- **Grid search:** Exhaustive search over a specified hyperparameter grid, evaluating each combination (typically via cross-validation) to find the best config.
- **Overfitting:** Model fits training data (including its noise) too closely and generalizes poorly to unseen data.
- **Precision vs. Recall:** Precision = of predicted positives, how many were correct; Recall = of actual positives, how many were caught.
- **JWT:** A signed, self-contained token encoding claims (like user ID) — verified via signature, no server-side session lookup needed.
- **SSE vs WebSocket:** SSE is one-way (server→client) over plain HTTP; WebSocket is full-duplex over a separate protocol upgrade.
- **ORM:** A layer that maps database rows to language objects (e.g., Prisma, SQLAlchemy), letting you query/manipulate data without raw SQL.

---

## 16. Top Priority Ranking

### 🔴 MUST KNOW (very likely)
- Full walkthrough of CV Tailor's agent pipeline, in your own words, with the exact 5 agent names/roles.
- Why LangGraph, why multi-agent, what "autonomous" actually means here.
- Botfolio's video pipeline chain (browser → GCS → ffmpeg → Speech-to-Text → LLM grading).
- Engagement rate formula (Creator Discovery) — cold, exact.
- Scholarship Extractor's confidence-score rubric and anti-hallucination design — this is your best answer, use it whenever asked "tell me about a time you thought about AI reliability/safety."
- The paper's 40-combination grid search (5 models × 2 optimizers × 4 features), the headline result (ViT + Adam + full face = 90%), and what *you specifically* did day-to-day despite the "all authors contributed equally" byline — have one concrete, specific answer ready, not a vague deflection.

### 🟠 HIGH PRIORITY
- Trade-offs: FastAPI vs Node, SQL vs NoSQL choices, JWT vs sessions, Playwright vs official APIs (and the ToS honesty point).
- "What would you change/scale/improve" for each project — you already have real, README-backed answers (Redis/Celery for CV Tailor, headless-browser fallback for Scholarship Extractor).
- Grid search explanation, tied to your internship work.

### 🟡 MEDIUM PRIORITY
- Exact DB schema field names (good to skim once, not memorize perfectly).
- NSS leadership story (operations, stakeholder management) as a behavioral answer.
- LeetCode/DSA — expect at most a light sanity-check question given the "600+ solved" claim, possibly a live problem.

### 🟢 LOWER PRIORITY
- Deep transformer/attention internals (you haven't personally trained a transformer from scratch based on available evidence — don't overclaim depth you don't have here; a surface-correct definition is enough).

---

## 17. Final Revision Plan (Interview Tomorrow)

**30 minutes — must-know cold:**
- CV Tailor's 5-agent pipeline names + flow + why LangGraph.
- Engagement rate formula + Scholarship Extractor's confidence rubric numbers.
- Your "tell me about yourself" answer, spoken out loud twice.

**1 hour — project explanations + key decisions:**
- Re-read this doc's Sections 4–7 (all four projects' Architecture Q&A + WHY questions) once, out loud if possible.
- Practice the Botfolio video-pipeline walkthrough until it's fluent without notes.

**2 hours — deep technical + attack trees:**
- Go through every Attack Tree in Sections 4–7 and answer each arrow yourself before checking this doc.
- Review Section 10 (AI/ML fundamentals) — especially RAG, hallucination, agents, grid search.

**3–4 hours — full pass:**
- Read every section of this document once end-to-end.
- Re-open your own 4 GitHub repos and skim each README fresh — some details here are dense; seeing them again in your own repo context will cement recall.
- Re-read Section 11 (now fully verified) once more and rehearse your specific-contribution answer for the "all authors contributed equally" question out loud.

**Final 30 minutes before the interview:**
- Re-read Section 16 (🔴 MUST KNOW) only.
- Say your "tell me about yourself" out loud once more.
- Remind yourself: WHAT → HOW → WHY → ALTERNATIVES → TRADE-OFFS, on every technical answer.

Good luck — your project set is genuinely strong, especially Scholarship Extractor's anti-hallucination design. Lean into that as your differentiator; it shows you think about AI *reliability*, which is exactly what separates a wrapper-builder from an AI engineer.
