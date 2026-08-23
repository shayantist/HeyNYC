# HeyNYC safety

_Last updated: 2026-08-23._

HeyNYC answers questions where a confident mistake can cost someone money, housing, food, or time. Its public promise is narrow: **show which source supports each claim, clearly explain what could not be confirmed, and keep useful partial evidence and links instead of replacing the answer with generic failure copy.** High-stakes claims need authoritative evidence. A directly supporting editorial excerpt can support a low-stakes event claim; truly unsourced material must be described as something HeyNYC could not confirm ([search trust grading](heynyc/core/tools/web_search.py), [runtime validation](heynyc/core/pydantic_runtime/runtime.py)).

Looking for how to report a security vulnerability (a bug, a leak, an injection exploit)? That's a different doc: see [SECURITY.md](SECURITY.md). This one is about AI safety, whether the assistant gives safe, grounded, honest answers.

## The design: verify, cite, and label

The answer model produces small internal claim and framing units. Each factual unit names the source IDs that should support it. The runtime checks those units separately, then joins related units into normal paragraphs and places the citation after the supported sentence. The resident sees prose, not the internal checklist ([structured output](heynyc/core/pydantic_runtime/projection.py), [runtime validation](heynyc/core/pydantic_runtime/runtime.py)).

This is a local implementation of a common production pattern, not a claim that HeyNYC copied one vendor's schema. [Anthropic citations](https://platform.claude.com/docs/en/build-with-claude/citations) bind generated text to source passages, while [Gemini grounding](https://ai.google.dev/gemini-api/docs/google-search) returns citation metadata for spans of ordinary text. [OpenAI recommends structured intermediate outputs](https://developers.openai.com/api/docs/guides/latest-model) when later code must validate or transform model output. Medical question-answering research likewise finds that a valid source link alone does not establish that the source supports the statement beside it ([Nature Communications, 2025](https://www.nature.com/articles/s41467-025-58551-6)).

**A failed claim-source check does not erase the work already done.** The runtime keeps useful text and retrieved records, asks the answer model to restate only what the source supports, and retains the relevant source links. If correction fails, it preserves the material and explains the specific limitation instead of replacing the answer with generic copy ([claim-source boundary](heynyc/core/pydantic_runtime/runtime.py)). This directly addresses the failure pattern documented in reporting on New York City's MyCity chatbot, including guidance that businesses could take workers' tips or refuse tenants using housing vouchers ([The Markup and THE CITY](https://themarkup.org/artificial-intelligence/2024/03/29/nycs-ai-chatbot-tells-businesses-to-break-the-law)).

Two boundaries fall out of this design:

- **It never makes an eligibility decision.** The screener runs the city's own official Benefits Screening API and reports "likely eligible per the city's engine, this is an estimate, the agency decides." It does not compute its own yes/no.
- **It routes specialized questions to a human.** Legal, medical, and immigration questions get routed (Right to Counsel, ActionNYC, 911, a caseworker) with a clear "I'm an AI, not a lawyer" disclaimer. It does not self-authorize as your lawyer.

## How we test it

### Offline tests and live evals are separate gates

Every service module ships with its own eval, and a module isn't "done" until that eval is green. The offline pytest suite exercises deterministic code, injected tools, and scripted traces without calling a model. The live `heynyc eval` command runs the configured model through the real agent, then applies the deterministic floor (attribution, faithfulness, grounding, link liveness, and the other structural checks defined in [`heynyc/eval/README.md`](heynyc/eval/README.md)) to the captured trace. The two are complementary. A green offline suite does not prove that the current model selected the right tools, stayed in language, or produced a useful answer. Live testing is selective and risk-triggered to control spend: changed modules and failure cases run after scoped changes, a compact cross-cutting set gates prompt, model, routing, memory, and guard changes, and the full golden and adversarial suites are reserved for public releases and major safety-boundary changes. Channel tests use the direct agent path first and only the smallest WhatsApp smoke needed to prove transport behavior. Design, commands, rubric, and the exact cadence are in [`heynyc/eval/README.md`](heynyc/eval/README.md).

### The Part C cited-claim check

On top of that floor is a **mechanical citation check** (we call it Part C). The runtime rejects unknown citation IDs, discovery-only evidence, internal markup, and exact address, date, money, phone, or unit-number values that disagree with the cited structured snapshot. High-stakes guidance uses typed claim blocks with explicit citation ownership, and cited web evidence must be authoritative. Unknown-domain excerpts remain discovery-only for high-stakes or ambiguous capability sets; only an explicitly low-stakes capability can retain one as a labeled unverified excerpt. The comparison uses the record captured at query time rather than a re-fetch ([search trust grading](heynyc/core/tools/web_search.py), [runtime validator](heynyc/core/pydantic_runtime/runtime.py), [grounding implementation](heynyc/core/grounding.py)).

### The runtime verification guard

Part C above started as an offline evaluation check. The [configured Pydantic runtime](heynyc/core/pydantic_runtime/runtime.py) now applies its conclusive mechanical checks before a cited answer ships.

The boundary is **grounding, not truth**. Mechanical checks can establish that an exact structured value matches its captured record. They cannot establish that arbitrary natural-language prose follows from a passage, and they do not decide whether the source is true or current. Faithfulness to a source and truth in the world are separate axes ([Maynez et al., ACL 2020](https://arxiv.org/abs/2005.00661)). Keeping the source fresh is a separate job.

The default configured Pydantic resident path always applies mechanical checks. It adds the third check below only when a high-stakes situation capability participates. Low-stakes turns skip that extra model call, and the selectable legacy runtime does not use it ([runtime selection](README.md#current-status), [configured runtime](heynyc/core/pydantic_runtime/__init__.py)).

- **Mechanical validation.** The runtime checks citation IDs, source classes, URLs, and exact structured values such as dates, phone numbers, amounts, and addresses ([grounding implementation](heynyc/core/grounding.py)).
- **Correction.** A conclusive mechanical failure returns the specific problem to the answer model for up to two complete-answer retries. External correction signals are more reliable than asking a model to reconsider without new evidence ([RARR, ACL 2023](https://arxiv.org/abs/2210.08726), [Huang et al., ICLR 2024](https://arxiv.org/abs/2310.01798)).
- **Claim-source check for high-stakes capabilities.** A separate configured checker compares each cited claim with the source excerpt. This adds model cost and latency, so ordinary low-stakes turns do not run it. If it finds weak, partial, or contradictory support, the answer model gets one chance to rewrite the answer naturally using only what the source supports. If that correction cannot complete, HeyNYC preserves the useful material and source link and explains the limitation ([claim-source checker](heynyc/core/nli.py), [runtime integration](heynyc/core/pydantic_runtime/runtime.py)).

The claim-source checker is a guard, not an oracle. It judges whether retrieved text supports a claim, not whether the source is correct, current, or complete. Complete traces are still reviewed during live evaluations, including the resident request, tool calls, retrieved evidence, answer, and citations ([evaluation conventions](heynyc/eval/README.md)).

This is not a mathematical truth guarantee. The guard enforces citation ownership and conclusive exact-value checks; complete live traces are reviewed for broader claim support. Source freshness, authority, and completeness remain separate responsibilities.

Full design, rationale, and the research behind each choice are kept in the project's internal design specs.

### Adversarial and failure testing

The public test set includes the documented MyCity failures from both the business and affected-resident perspective, along with prompt injection, privacy, false-premise, citation, high-stakes, and non-English cases. After correcting the original grading, the first 137-query review found six grounding failures. It found no successful jailbreak, PII mishandling, fabricated citation, or repetition of MyCity's unlawful guidance. These are historical results, not proof that the larger current suite is clean; that suite has not received a complete owner-approved live rerun. The exact results, current method, and known failures remain in the [red-team record](docs/testing/red-team.md#results-to-date), [benchmark method](docs/testing/benchmarks.md), and [failure register](docs/testing/failure-db.md).

## Keeping the source current

The guard checks an answer against a source. It does not prove that the source itself is current.

- **Live data stays live.** Service tools query their responsible datasets at request time where possible and retain the source's own update signal ([source catalog](docs/source-capability-catalog.md)).
- **Static facts carry dates.** Cached rules, thresholds, and phone numbers use freshness metadata and surface a caveat when stale ([freshness implementation](heynyc/core/freshness.py)).
- **Discovery is not a whitelist.** Web search retains unknown domains as candidates. Source tier affects what a result may verify, especially for high-stakes guidance, but does not decide whether the result may be discovered ([search implementation](heynyc/core/tools/web_search.py)). Retrieved text remains untrusted because poisoned retrieval is a documented attack surface ([PoisonedRAG](https://arxiv.org/abs/2402.07867), [OWASP guidance](https://owasp.org/www-project-top-10-for-large-language-model-applications/)).

Automated content-drift alerts and a fixed human review cadence for high-stakes static facts remain planned. HTTP conditional requests provide the intended low-cost change signal ([MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Conditional_requests)), while government content practice provides the owner and review-date model ([GOV.UK](https://www.gov.uk/guidance/content-design/content-maintenance)).

## Conversation continuity is a safety and privacy boundary

HeyNYC keeps bounded conversation history so follow-up questions work. The application replaces the channel sender with a [salted pseudonymous key](heynyc/channels/identity.py), requires encryption for hosted serving, and expires conversation and draft records on a schedule ([channel service](heynyc/channels/app.py)). Previous assistant text may provide conversational context, but it is never evidence for a new factual claim ([memory implementation](heynyc/core/memory.py)).

Twilio messages enter an encrypted inbox before acknowledgement, replies are staged before delivery, and completed queue content is scrubbed ([Twilio adapter](heynyc/channels/twilio.py)). Confirmed `DELETE MY DATA` removes the resident's transcript, queued messages, draft, and pending report flags while retaining only non-identifying operational totals ([channel orchestrator](heynyc/channels/orchestrator.py)). The short explanation is in [PRIVACY.md](PRIVACY.md), and the formal [Privacy Notice](docs/legal/HEYNYC-PRIVACY.md) controls.

This design uses [NIST's Privacy Framework](https://www.nist.gov/privacy-framework), [GDPR data-minimization and storage-limitation principles](https://eur-lex.europa.eu/eli/reg/2016/679/art_5/oj), and the [GOV.UK Chat privacy notice](https://www.gov.uk/government/publications/govuk-chat-privacy-notice/govuk-chat-privacy-notice) as references. HeyNYC is not certified or independently audited against them. Self-service export, automated cross-host migration, and a supervised live restart test remain unfinished ([formal gap list](docs/legal/HEYNYC-PRIVACY.md#4-standards-evidence-and-current-gaps)).

## The human-in-the-loop boundary

HeyNYC is built to get people *through the door*, not to replace the person who decides. Concretely:

- It **never decides eligibility.** The screener is the city's official engine and reports an estimate; the agency makes the determination.
- It **never submits anything on your behalf.** It prepares; you act.
- It **routes the high-stakes call to a human:** 311 for city services, a caseworker for benefits, Right to Counsel for eviction, ActionNYC for immigration, 911 for emergencies.

A wrong answer to someone who is least able to catch it is the failure mode we care most about, so the human path is always there, and the assistant is upfront that it's an AI, not a City employee.

Human review for consequential decisions follows the direction of the [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework), the [EU AI Act's human-oversight rule](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-14), and [New York City's AI Action Plan](https://www.nyc.gov/assets/oti/downloads/pdf/reports/artificial-intelligence-action-plan.pdf). These are design references, not compliance or certification claims.

## Limitations (what we haven't proven yet)

These are real and current as of 2026-08-18:

- **Non-English safety is not proven at production grade.** Spanish and Bengali tests have found errors absent from the English answer. Different languages can produce different factual error rates ([multilingual hallucination research](https://arxiv.org/abs/2410.18270), [Multi-FAct](https://arxiv.org/abs/2402.18045)). Dedicated cases show that we test the surface, not that every language is safe.
- **Language access and accessibility have not been independently audited.** HeyNYC builds toward [NYC's language-access requirements](https://www.nyc.gov/site/civicengagement/about/language-access-plan.page) and [WCAG 2.1 Level AA](https://www.w3.org/TR/WCAG21/), but does not claim certification.
- **No external third-party rating yet.** The independent fresh-context grader is a genuine improvement over grading our own homework, and it earned its keep by catching the overclaim. But it's still our own second agent, not an outside human or organization holding the red pen. A truly external rater is a stated next step, not something we've done.
- **Coverage is a growing subset, and gaps stay visible.** Measured against a frozen 170-query benchmark on 2026-07-05, about 78 were a clean yes, 49 partial, and 43 a gap (roughly 46%). Four more grounded modules (clinics, Housing Connect, WIC, childcare) have shipped since, so today's coverage is higher, though we have not re-measured against the frozen set. On questions we do not cover, HeyNYC labels the gap, preserves any useful source material, and offers a direct route to continue. The live experience is still thinner than the ambition, and we would rather say that than oversell it.
- **Grounding reduces risk; it does not guarantee correctness.** A source can be wrong, stale, incomplete, or misread. HeyNYC promises traceability and honest labels, not perfect accuracy.
- **Live data can go stale, and live legal facts move.** We date every dataset citation with a "valid as of" and flag staleness, but between refreshes we can lag, and litigation-live items (like the Section 8 ruling) need re-checking against current law.

## Where to read more

The testing records behind these claims live in [`docs/testing/`](docs/testing/), generated from internal sources by `scripts/export_testing_docs.py`:

- No-hallucination eval design and rubric: [`heynyc/eval/README.md`](heynyc/eval/README.md)
- MyCity safety subset (the traps and the real law behind each): [`docs/testing/benchmarks.md`](docs/testing/benchmarks.md)
- The historical 137-query red-team, results, and reconciliation: [`docs/testing/red-team.md`](docs/testing/red-team.md#results-to-date)
- How we red-team (the current 205-case method): [`docs/testing/red-team.md`](docs/testing/red-team.md#how-we-red-team)
- Benchmark methodology (how we measure at scale): [`docs/testing/benchmarks.md`](docs/testing/benchmarks.md)
- Security vulnerability reporting (a different thing): [SECURITY.md](SECURITY.md)

HeyNYC is an open-source project and is not affiliated with the City of New York. If you find an unsafe or ungrounded answer, open an issue. Report sensitive vulnerabilities privately through [SECURITY.md](SECURITY.md).
