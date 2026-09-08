# Project Requirements — SAG Vietnam Legal Agent

## 1. Project Overview

Build a **reproducible legal-information agent for Vietnamese law** using **SAG** as the core retrieval/reasoning approach, wrapped with a **hybrid retrieval + reranking front-end** and a **mandatory Hindsight (self-verification) layer** before any answer is shown to a user.

Reference SAG repository:
https://github.com/Zleap-AI/SAG

> **Correction / clarification on SAG.** SAG stands for **SQL-Retrieval Augmented Generation**, not "Search-Augmented Generation." It is *not* a drop-in vector-RAG replacement you call once — it is an indexing + reasoning architecture that:
> - Converts each ingested chunk into one semantically complete **event** plus a set of indexing **entities** (an event-entity index), instead of storing only raw text chunks.
> - At query time, uses **SQL join queries to dynamically link events that share entities into local hyperedges** — i.e., it builds a small, query-specific relational structure on the fly rather than maintaining one static global knowledge graph.
> - Supports **fast (vector) and precise (multi-hop) retrieval modes**, source-traceable citations (every result maps back to the original chunk), and an agent-orchestration core with MCP support.
> - Is implemented as a FastAPI backend (`sag-api`) plus a Python package (`zleap-sag`), embeddable locally (SQLite + LanceDB with built-in BM25) or backed by Postgres/pgvector or Elasticsearch in production.
>
> The intern must read the actual SAG repo/paper before building anything — do not assume SAG is "just embeddings + an LLM." Its main value for this project is **multi-hop relational reasoning across legal provisions** (e.g., linking a Circular's implementing article back to the Law article it interprets, or linking amendment/replacement chains), which plain vector RAG handles poorly.

The system is intended as an internal research/prototype tool for:
- Legal/compliance department staff
- Board of Directors (BoD)

The primary goal is **not to replace lawyers**. The system must retrieve relevant legal sources, reason over them, provide traceable citations, and clearly communicate uncertainty.

---

## 1.1 Reference Implementations (use, but do not copy blindly)

Two external repos are relevant. Neither replaces the requirements below — both are accelerants, and the intern should be explicit in the weekly report about what was reused vs. built from scratch.

### a) SAG — https://github.com/Zleap-AI/SAG
The retrieval/reasoning core this project is required to study and adopt (see corrected description above). Also review the `zleap-sag` PyPI package and the underlying paper ("SAG: SQL-Retrieval Augmented Generation with Query-Time Dynamic Hyperedges") for the event-entity model and hyperedge construction logic — this is what Week 1's "SAG study notebook" should demonstrate an understanding of.

### b) https://github.com/Paparusi/legal-ai-agent (MIT licensed)
A pre-existing, fairly full-featured Vietnamese legal AI product (FastAPI + Postgres/pgvector + a VSCode-style web UI) covering legal search over a large indexed law corpus, AI contract review with a 10-category risk taxonomy, clause drafting, and a crawler for Vietnamese legal sources (thuvienphapluat.vn, vbpl.vn, congbao.chinhphu.vn).

**What can reasonably be reused/adapted from it:**
- Its **crawler patterns** for the three named Vietnamese legal-document sources — a major head start on Week 1–2's corpus-building.
- Its **contract risk-category taxonomy** (one-sided clauses, excessive penalty >8%, unreasonable deadlines, missing protective clauses, clauses contradicting law, auto-renewal traps, liability limits, IP/confidentiality, dispute resolution, force majeure) as a starting point for Section 13's "constrained set of contract issue categories."
- General project scaffolding ideas (FastAPI route layout, Docker Compose setup) if useful for the MVP's backend.

**What must NOT be reused as-is, because it does not meet this project's scope:**
- Its retrieval layer is **full-text search + synonym expansion + TF-IDF ranking** — this is *not* the hybrid BM25 + dense embedding + reranker + SAG pipeline this project requires (Section 6). Do not treat its search quality as a baseline to match and stop there.
- Its law data / citations should be **treated as unverified content to re-derive from primary sources**, not copied wholesale — the repo is a commercial product, not a research artifact, and its legal citations have not been validated against this project's Source-of-Truth requirements (Section 4).
- Any reused code must keep MIT license attribution, and the intern should note in the weekly report exactly which files/ideas were adapted from it.

---

## 2. Core Use Cases

### Use Case A — Legal validity / legality Q&A

Example questions:

- "Công ty có được phép kinh doanh sản phẩm X không?"
- "Hoạt động này có phù hợp với ngành nghề kinh doanh đã đăng ký không?"
- "Điều kiện pháp lý để cung cấp dịch vụ X là gì?"
- "Có cần giấy phép/chứng nhận/đăng ký bổ sung không?"
- "Quy định hiện hành nằm ở điều, khoản, điểm nào?"

Expected answer structure:

1. Short conclusion
2. Applicable legal conditions
3. Relevant legal documents
4. Exact citation:
   - Văn bản
   - Điều
   - Khoản
   - Điểm
   - Mục/tiết when applicable
5. Reasoning / application to the company's situation
6. Assumptions
7. Uncertainty / missing information
8. Sources
9. Verification status (SUPPORTED / PARTIALLY_SUPPORTED / INSUFFICIENT_EVIDENCE / CONFLICTING_SOURCES / REQUIRES_HUMAN_REVIEW), assigned or confirmed by the Hindsight layer (Section 6.5) — not just asserted by the generation LLM.

The system must avoid presenting unsupported conclusions as facts.

---

### Use Case B — Contract review

Input:
- Uploaded contract/document

Output:

1. Executive summary
2. Potential legal issues
3. Risk classification
   - Critical
   - High
   - Medium
   - Low
4. Relevant contract clause
5. Relevant Vietnamese legal provision
6. Explanation of the conflict/risk
7. Suggested review/action
8. Missing information required for a definitive assessment
9. Source citations
10. Hindsight-verified confidence note per flagged issue (see Section 6.5 and Section 13)

The system should distinguish between:
- Explicit legal violation
- Potential legal risk
- Commercial/business risk
- Ambiguous wording
- Missing contractual protection
- Issue requiring human lawyer review

---

## 3. Legal Knowledge Scope

### Framework 1 — Financial Institution

Primary legal domains:

1. Luật Các tổ chức tín dụng (TCTD)
2. Banking regulations / Luật Ngân hàng and relevant banking legislation
3. Luật Kế toán
4. Luật Doanh nghiệp
5. Luật Chứng khoán and listing regulations
6. Personal data protection legislation
7. Luật Thương mại
8. Intellectual property / trademark protection
9. Luật Lao động
10. Luật Phòng, chống rửa tiền

Also include relevant subordinate/implementing legislation:

- Nghị định
- Thông tư (TT)
- Quyết định
- Văn bản hướng dẫn
- Other legally relevant instruments

The system should model relationships between:
- Law
- Decree
- Circular
- Guidance
- Amendments
- Replacements
- Effective dates

This is exactly the kind of cross-document relationship SAG's event-entity hyperedges are meant to capture — e.g., an entity "Điều 126 Luật TCTD" linking to events describing implementing Thông tư articles.

---

### Framework 2 — Publishing / Media Organization

Primary legal domains:

1. Luật Xuất bản
2. Luật Báo chí
3. Personal data protection legislation
4. Luật Doanh nghiệp
5. Luật Kế toán

Also include relevant:
- Nghị định
- Thông tư
- Guidance documents
- Amendments/replacement documents

---

### Company-Specific Data

The system must support private organizational knowledge, including:

- Giấy chứng nhận đăng ký doanh nghiệp / business registration
- Điều lệ công ty
- Other company policies
- Internal legal/compliance documents
- Uploaded contracts

Company-specific documents must be logically separated from public legal sources.

---

## 4. Source-of-Truth Requirements

Legal answers must prioritize authoritative sources.

Preferred source hierarchy:

1. Official Vietnamese government/legal databases
2. Official ministry/regulator websites
3. Official legal-document repositories
4. Company-provided authoritative documents
5. Reputable secondary legal sources only when necessary

Secondary sources must never silently replace primary legal sources. This explicitly includes the Paparusi/legal-ai-agent repo's bundled law data (Section 1.1) — its indexed content is a convenience reference at best, never a primary source.

Every legal conclusion should be traceable to the source document.

---

## 5. Legal Document Representation

Each legal document should ideally contain structured metadata:

```text
document_id
title
document_type
document_number
issuing_authority
issued_date
effective_date
expiration_date
status
amends
replaces
replaced_by
domain
source_url
content
article
clause
point
section
```

The system should preserve legal hierarchy:

```text
Document
 └── Chapter
      └── Section
           └── Article (Điều)
                └── Clause (Khoản)
                     └── Point (Điểm)
                          └── Sub-point / item
```

Do not flatten this hierarchy if doing so would make legal citations inaccurate.

---

## 6. Retrieval Architecture

The initial architecture should be modular. The pipeline order below is **mandatory**, not just a suggestion: candidates must first come from BM25 **and** embedding search (hybrid), and only the resulting candidate set is passed to the reranker to select the final evidence set.

```text
User Query
    ↓
Query Analysis / Router
    ↓
Query Reformulation
    ↓
Hybrid Retrieval (mandatory: run both, then fuse)
    ├── BM25 / lexical search
    ├── Embedding / dense semantic search
    └── Metadata filtering (domain, framework, effective/expiration dates)
    ↓
Fused Candidate Set (e.g. Reciprocal Rank Fusion of BM25 + dense results)
    ↓
Reranker — Voyage AI (see 6.1)
    ↓
Reranked Top-K Candidates
    ↓
SAG Retrieval / Multi-hop Reasoning (event-entity hyperedge expansion over the reranked set)
    ↓
Evidence Selection
    ↓
LLM Generation (draft answer)
    ↓
Hindsight / Self-Verification Layer (see 6.5 — mandatory)
    ↓
Citation Validation
    ↓
Structured Answer
```

The intern should experiment with:
- BM25
- Dense embeddings
- Hybrid retrieval (fusion method: RRF, weighted score fusion, etc. — document the choice)
- Reranking (Voyage AI)
- Query expansion
- Metadata filtering
- SAG
- Hindsight / self-verification
- LLM-based reasoning

The project must document why each component is included, and must run the ablation comparisons listed in Week 4 (Section 17) to justify keeping each stage.

### 6.1 Reranker — Voyage AI

Use the Voyage AI reranker (https://docs.voyageai.com/docs/reranker) as the mandatory reranking stage between hybrid retrieval and SAG/evidence selection. Voyage rerankers are cross-encoders: they jointly score a query against each candidate document, which is more accurate than the separately-encoded query/document vectors used in stage-1 embedding search — this is why it sits **after** BM25+embedding retrieval and narrows, rather than replaces, that candidate set.

Implementation notes:
- **Recommended model:** `rerank-2.5` (or `rerank-2.5-lite` for latency-sensitive experiments) — both are multilingual with instruction-following support, which matters for Vietnamese legal text; avoid the older `rerank-1`/`rerank-2` family unless doing a deliberate cost/quality comparison.
- **API shape:** `voyageai.Client().rerank(query, documents, model, top_k, truncation)` (Python) or `POST https://api.voyageai.com/v1/rerank` (REST). Returns a list of `{index, document, relevance_score}` sorted by descending relevance.
- **Constraints to design around:** max 1,000 candidate documents per call; combined query+document context length up to 32,000 tokens for `rerank-2.5`/`rerank-2.5-lite`; a total-token budget cap per call (see Voyage's FAQ) — so the hybrid stage's candidate count (`top_n` before reranking) must be tuned to stay within these limits, and this tuning should be recorded as an experiment parameter.
- Never call the API key directly from client-side code; load `VOYAGE_API_KEY` from environment/secrets, matching the "never hard-code API keys" principle in Section 18.
- Record reranker `top_k`, model name, and latency/cost per query in every retrieval experiment (Section 15).

### 6.2 Where SAG sits relative to the reranker

SAG's multi-hop hyperedge expansion should run **after** reranking, over the reranked top-K set — i.e., use the reranker to cheaply cut noise from the hybrid candidate pool, then let SAG's event-entity linking pull in related provisions (implementing Thông tư, amendment/replacement chain) that the reranker alone would not surface. Document any experiment where this ordering is reversed and why.

---

## 6.5 Hindsight / Self-Verification Layer (mandatory)

This is a required pipeline stage, not an optional enhancement. Before any answer reaches the user, the agent must review its own draft output against the evidence it actually retrieved.

**What it does:**
1. Take the LLM's draft answer (from the main generation step) plus the exact evidence chunks that were provided to it (post-reranker, post-SAG expansion).
2. Check, claim by claim, whether each substantive statement in the draft is actually supported by the cited evidence — not just plausible-sounding.
3. Check whether every citation (Điều/Khoản/Điểm/document) in the draft actually exists in the retrieved evidence, catching invented article numbers or mismatched documents.
4. Check for temporal validity issues (Section 9) — e.g., the draft citing a provision whose `expiration_date` has passed without flagging it as historical.
5. Based on the above, either:
   - Approve the draft as-is,
   - Revise the draft (removing/softening unsupported claims, fixing citations), or
   - Downgrade/assign the verification status (`SUPPORTED` / `PARTIALLY_SUPPORTED` / `INSUFFICIENT_EVIDENCE` / `CONFLICTING_SOURCES` / `REQUIRES_HUMAN_REVIEW` from Section 11).
6. Log what was changed and why — this log is part of the reproducibility requirement (Section 14) and should be inspectable per answer.

**Why this is a "must, not a plus":** the project's core safety requirement (Section 11 — never invent provisions, never fabricate citations, prefer abstention) is very hard to guarantee from generation-time prompting alone. A second, evidence-grounded pass that can catch and correct its own model's mistakes is the mechanism that actually makes that guarantee credible enough to show a BoD member.

**Implementation approaches to compare (pick at least one for the MVP, document trade-offs):**
- A second LLM call with a strict verification prompt (same or a different/cheaper model) that only sees the draft + evidence, not the original generation prompt — reduces the chance it just re-confirms its own reasoning.
- A programmatic citation-matching pass (regex/structured check that every cited Điều/Khoản/Điểm string appears in the evidence set) combined with an LLM check only for claims that pass citation matching but might still be unsupported in substance.
- A lightweight critique-and-revise loop (Reflexion-style): generate → critique → regenerate once if the critique finds problems, capped at one revision to bound latency/cost.

**Evaluation** (extends Section 8):
- Revision rate (% of drafts the Hindsight layer changes)
- Catch rate on a seeded error set (deliberately inject a wrong article number or unsupported claim and measure how often Hindsight catches it)
- False-flag rate (Hindsight incorrectly downgrading a correct answer)
- Added latency/cost per query from this stage

---

## 7. Chunking Strategy

Legal documents require structure-aware chunking.

Do NOT rely only on arbitrary token/character chunking.

Preferred approach:

```text
Document
→ Chapter
→ Section
→ Article
→ Clause
→ Point
```

A chunk should preserve enough parent context to understand the provision.

Example metadata:

```json
{
  "document_id": "...",
  "article": "Điều 24",
  "clause": "Khoản 2",
  "point": "Điểm a",
  "chapter": "...",
  "text": "...",
  "effective_date": "...",
  "status": "active"
}
```

---

## 8. Retrieval Requirements

The retrieval system should answer:

> "Which exact legal provisions should the LLM use to answer this question?"

Evaluation should measure:

### Retrieval

- Recall@K
- Precision@K
- MRR
- nDCG where appropriate
- Report each metric per stage: BM25 alone, dense alone, hybrid (fused), hybrid+reranker, hybrid+reranker+SAG — so the reranker's and SAG's marginal contribution is each independently visible.

### Answer quality

- Citation accuracy
- Citation completeness
- Groundedness
- Correctness
- Hallucination rate
- Abstention quality
- Hindsight metrics (Section 6.5): revision rate, catch rate, false-flag rate

### Contract review

- Issue detection precision
- Issue detection recall
- Severity classification accuracy
- Correct legal citation rate

Do not optimize only for LLM-generated answer quality. Evaluate retrieval independently.

---

## 9. Temporal Legal Validity

This is a critical requirement.

Vietnamese legal documents change over time.

The system must distinguish:

- Current/effective provisions
- Expired provisions
- Amended provisions
- Replaced documents
- Future-effective provisions

The system should support queries such as:

> "Quy định hiện hành tại ngày X là gì?"

Retrieval should consider `effective_date` and `expiration_date`.

Never use an expired provision as the current legal basis without explicitly stating its historical status. The Hindsight layer (Section 6.5) is the last checkpoint responsible for catching this before an answer ships.

---

## 10. Citation Requirements

Every substantive legal claim should have a traceable citation.

Preferred citation:

```text
[Luật/ Nghị định/ Thông tư]
Điều X, Khoản Y, Điểm Z
Source: <URL>
```

Citations should point to the smallest practical legal unit.

Bad:

> "Theo Luật TCTD..."

Better:

> "Theo Điều X, Khoản Y, Điểm Z của [văn bản], ..."

The final answer should make it possible for a human reviewer to verify the claim.

---

## 11. Hallucination / Safety Requirements

The system must:

- Never invent legal provisions
- Never invent article numbers
- Never fabricate sources
- Never fabricate citations
- Explicitly state when evidence is insufficient
- Distinguish legal certainty from model inference
- Prefer abstention over unsupported conclusions

Recommended answer states:

```text
SUPPORTED
PARTIALLY_SUPPORTED
INSUFFICIENT_EVIDENCE
CONFLICTING_SOURCES
REQUIRES_HUMAN_REVIEW
```

These states are assigned/confirmed by the Hindsight / Self-Verification Layer (Section 6.5), not asserted directly by the generation LLM.

---

## 12. Company-Specific Compliance Reasoning

The agent should eventually combine:

```text
Public Legal Knowledge
        +
Company Registration
        +
Company Charter
        +
Internal Policies
        +
User Scenario
        ↓
Compliance Assessment
```

Example:

```text
Can Company X provide Service Y?
```

The system should check:

1. Whether the activity is legally permitted
2. Whether the activity is within registered business lines
3. Whether sector-specific conditions apply
4. Whether a license/approval is required
5. Whether the company's charter creates additional restrictions
6. Whether relevant personal-data, labor, AML, tax/accounting, securities, etc. obligations apply
7. Whether the evidence is sufficient

---

## 13. Contract Review Architecture

Suggested pipeline:

```text
Contract Upload
    ↓
Document Parsing
    ↓
Contract Structure Detection
    ↓
Clause Segmentation
    ↓
Issue Detection
    ↓
Legal Retrieval (Hybrid → Voyage Reranker → SAG, per Section 6)
    ↓
Legal/Clause Comparison
    ↓
Risk Classification
    ↓
Hindsight / Self-Verification (Section 6.5) — re-check each flagged issue's citation and severity before reporting
    ↓
Citation Validation
    ↓
Review Report
```

Do not initially attempt full autonomous legal review.

Start with a constrained set of contract issue categories. The Paparusi/legal-ai-agent repo's 10-category taxonomy (Section 1.1) is a reasonable starting checklist to adapt:

- Missing mandatory terms
- Illegal/prohibited clauses
- Liability
- Termination
- Payment (incl. excessive penalty clauses — note the >8% ceiling referenced in Vietnamese commercial law as one concrete check to validate against the current Luật Thương mại text, not assumed from an external repo)
- Confidentiality
- Personal data
- Intellectual property
- Dispute resolution
- Governing law
- Labor-related provisions
- Regulatory compliance
- Auto-renewal traps
- Force majeure (missing or weak)

---

## 14. Reproducibility Requirements

Every major experiment must be reproducible.

Repository should contain:

```text
README.md
REQUIREMENTS.md
docs/
notebooks/
src/
tests/
data/
configs/
evaluation/
reports/
scripts/
```

Recommended structure:

```text
project/
├── README.md
├── REQUIREMENTS.md
├── pyproject.toml
├── .env.example
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── docs/
├── notebooks/
├── src/
│   ├── ingestion/
│   ├── chunking/
│   ├── retrieval/          # BM25, dense, hybrid fusion
│   ├── reranking/          # Voyage AI reranker client + config
│   ├── sag/                # SAG integration (event-entity indexing, hyperedge queries)
│   ├── hindsight/          # Self-verification / critique-and-revise layer
│   ├── generation/
│   ├── citation/
│   ├── contract_review/
│   └── evaluation/
├── tests/
├── evaluation/
├── reports/
└── scripts/
```

`.env.example` should list at minimum: `ANTHROPIC_API_KEY` (or chosen LLM provider key), `VOYAGE_API_KEY`, plus whatever DB/vector-store credentials the chosen stack needs. Secrets/API keys must never be committed.

---

## 15. Experiment Tracking

Each experiment should document:

- Hypothesis
- Dataset
- Retrieval method
- Embedding model
- Reranker (model name, e.g. `rerank-2.5`, and `top_k`)
- SAG configuration (retrieval mode: vector / atomic / multi-hop)
- Hindsight configuration (on/off, method used — Section 6.5)
- LLM
- Prompt/version
- Parameters
- Evaluation metrics
- Results
- Cost
- Latency
- Failure cases
- Conclusions

Example:

```text
Experiment: BM25 vs Hybrid Retrieval

Hypothesis:
Hybrid retrieval improves legal provision recall.

Dataset:
100 manually-created Vietnamese legal questions.

Metrics:
Recall@5
Recall@10
MRR
Citation accuracy

Result:
...

Conclusion:
...
```

---

## 16. Weekly Deliverables

Every Friday at **16:00**, submit a reproducible weekly package containing:

### 1. Notebook(s)

Demonstrate experiments and results.

### 2. Report

Explain:

- What was implemented
- Why it was implemented
- Results
- Problems
- Failure cases
- Next steps

### 3. Demo

Demonstrate the latest working functionality.

### 4. GitHub

Commit all relevant source code and documentation.

### 5. Reproducibility

Another developer should be able to understand:

- What changed
- How to run it
- What data was used
- What configuration was used
- How results were produced

---

# 17. Suggested Internship Roadmap

Assumption: approximately **8 weeks**. If the actual internship duration differs, scale the phases proportionally.

## Week 1 — Understand SAG + Legal RAG Baseline

Objectives:

- Study SAG (the actual event-entity/hyperedge architecture — Section 1) and skim the Paparusi/legal-ai-agent repo (Section 1.1) for corpus-building and taxonomy ideas
- Understand legal RAG
- Study Vietnamese legal document structure
- Define dataset
- Define evaluation methodology
- Build a minimal ingestion pipeline

Deliverables:

- SAG study notebook (must reflect the event-entity/hyperedge model, not a generic RAG description)
- Architecture document
- Initial legal corpus
- Basic chunking prototype
- Baseline BM25 retrieval
- Week 1 report/demo

---

## Week 2 — Legal Document Ingestion + Structure

Objectives:

- Build legal document parser
- Preserve Điều/Khoản/Điểm hierarchy
- Normalize metadata
- Implement structure-aware chunking
- Track effective dates/status

Deliverables:

- Ingestion pipeline
- Structured document schema
- Chunking notebook
- Example processed corpus
- Tests

---

## Week 3 — Retrieval Baseline

Implement and compare:

- BM25
- Dense embeddings
- Hybrid retrieval (fusion of the two — mandatory per Section 6)

Create a small manually verified evaluation dataset.

Deliverables:

- Retrieval benchmark
- Recall@K
- MRR
- Failure analysis
- Recommendation for retrieval architecture

---

## Week 4 — Reranking (Voyage AI) + SAG + Hindsight

Objectives:

- Add the Voyage AI reranker (Section 6.1) on top of hybrid retrieval
- Integrate SAG (Section 6.2)
- Build a first version of the Hindsight / self-verification layer (Section 6.5)
- Compare:
  - BM25
  - Dense
  - Hybrid
  - Hybrid + Voyage reranker
  - Hybrid + Voyage reranker + SAG
  - Hybrid + Voyage reranker + SAG + Hindsight (measure Hindsight's catch rate on a seeded error set)

Deliverables:

- Architecture update
- Benchmark
- Voyage reranker + SAG experiment notebook
- Hindsight catch-rate experiment notebook
- Failure analysis
- Cost/latency comparison

---

## Week 5 — Legal Q&A Agent

Build the first usable agent.

Capabilities:

- Query classification
- Retrieval (hybrid + reranker + SAG)
- Evidence selection
- LLM generation
- Hindsight self-verification pass before responding
- Citation generation
- Abstention

Deliverables:

- End-to-end Q&A demo
- Evaluation dataset
- Citation accuracy results
- Hallucination/failure analysis (with and without Hindsight, to show its measurable effect)

---

## Week 6 — Company-Specific Compliance

Add:

- Business registration
- Company charter
- Company-specific documents

Implement:

```text
Legal rules + Company facts → Compliance assessment
```

Deliverables:

- Company knowledge ingestion
- Compliance assessment prototype
- Example scenarios
- Evaluation results

---

## Week 7 — Contract Review

Build constrained contract review.

Focus on several high-value categories rather than trying to detect everything (starting checklist in Section 13).

Deliverables:

- Contract parser
- Clause segmentation
- Issue detection
- Legal retrieval
- Risk classification
- Hindsight-checked, citation-backed report

---

## Week 8 — Evaluation + Final Demo

Objectives:

- Consolidate architecture
- Improve reliability
- Run complete benchmark
- Document limitations
- Clean repository
- Reproduce experiments from clean environment

Final deliverables:

- Final demo
- Final report
- Evaluation report
- Notebooks
- Reproducible code
- Architecture documentation
- GitHub repository
- Known limitations
- Future roadmap

---

# 18. Development Principles

The AI coding assistant must act as a **development/research assistant**, not blindly implement the entire project.

When working on this repository:

1. Explain the reasoning before major architectural changes.
2. Prefer small, testable increments.
3. Do not introduce unnecessary frameworks.
4. Keep components modular.
5. Write tests for important retrieval/citation logic.
6. Record experiments.
7. Never hard-code API keys (including `VOYAGE_API_KEY`).
8. Do not silently fabricate legal data.
9. Make assumptions explicit.
10. Keep the system reproducible.
11. Prefer measurable experiments over subjective claims.
12. Preserve raw/source provenance.
13. Separate research code from production-like code.
14. When an implementation choice is uncertain, present alternatives and trade-offs before proceeding.
15. When adapting code or ideas from the Paparusi/legal-ai-agent repo (Section 1.1), note the source file and keep MIT attribution; never assume its bundled legal data or citations are correct without independent verification.

---

# 19. Definition of Done

A feature is considered complete only when:

- [ ] Code works
- [ ] Tests exist where appropriate
- [ ] Documentation exists
- [ ] Example usage exists
- [ ] Experiment/configuration is recorded
- [ ] Results are reproducible
- [ ] Legal sources are traceable
- [ ] Failure cases are documented

For retrieval changes:

- [ ] Evaluation dataset exists
- [ ] Baseline comparison exists
- [ ] Metrics are reported
- [ ] Failure cases are inspected

For legal answers:

- [ ] Source is traceable
- [ ] Citation is precise
- [ ] Current legal status is considered
- [ ] Unsupported claims are avoided
- [ ] Answer has passed the Hindsight / self-verification layer and carries a verification status (Section 6.5, Section 11)

---

# 20. AI Coding Assistant Operating Instructions

When the user asks you to work on this repository:

### Before coding

1. Read `REQUIREMENTS.md`.
2. Inspect the current repository structure.
3. Identify the current implementation stage.
4. Identify relevant existing modules.
5. State the proposed change.
6. Explain the reason and trade-offs.
7. Ask for confirmation only when the change is materially architectural or destructive.

### While coding

- Make small commits/changes.
- Keep interfaces stable.
- Add tests.
- Do not rewrite unrelated code.
- Preserve reproducibility.

### After coding

Report:

```text
Implemented:
- ...

Changed files:
- ...

Tests:
- ...

Experiment:
- ...

Known limitations:
- ...

Next recommended step:
- ...
```

The assistant should help the intern **learn and build the system**, not simply dump a complete implementation without explanation.

---

# 21. Initial MVP Priority

Do NOT attempt to implement the entire project simultaneously.

Recommended MVP:

```text
Vietnamese legal corpus
        ↓
Structure-aware ingestion
        ↓
BM25
        ↓
Dense retrieval
        ↓
Hybrid retrieval (fuse BM25 + dense)
        ↓
Voyage AI Reranker
        ↓
SAG (event-entity multi-hop expansion)
        ↓
LLM
        ↓
Hindsight / Self-Verification
        ↓
Citation validation
        ↓
Legal Q&A
```

Then expand to:

```text
Company-specific compliance
        ↓
Contract review
        ↓
Multi-framework support
        ↓
Production-quality agent
```

The primary research question is:

> **Can SAG + hybrid retrieval + Voyage AI reranking + a Hindsight self-verification pass produce more accurate, traceable, and reliable answers for Vietnamese legal questions than a conventional RAG pipeline?**

Every major experiment should help answer this question.
