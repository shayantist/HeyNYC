# HeyNYC safety

_Last updated: 2026-08-16 (verified and unverified output boundary clarified)._

HeyNYC helps New Yorkers find, understand, and apply for government services. It answers questions about benefits, housing, food, and immigration, the places where a confident wrong answer can cost someone money, their home, or their immigration status. So the bar here isn't "usually right." The bar is: **verified facts are grounded and cited, while material that could not be verified is visibly labeled and keeps its source link so the resident can check it.** High-stakes claims require authoritative evidence before HeyNYC labels them verified. For an explicitly low-stakes capability, an unverified search excerpt may support only the exact claim it states, with its source still identified as unverified ([search trust grading](heynyc/core/tools/web_search.py), [runtime validation](heynyc/core/pydantic_runtime/runtime.py)). This doc is how we try to hit that bar and, just as importantly, where we still fall short. It's written to be read by a skeptic.

Looking for how to report a security vulnerability (a bug, a leak, an injection exploit)? That's a different doc: see [SECURITY.md](SECURITY.md). This one is about AI safety, whether the assistant gives safe, grounded, honest answers.

## The design: verify, cite, and label

The core rule is **never blur verified and unverified material.** Every specific the agent labels verified (an address, a dollar amount, an eligibility rule, a deadline, a phone number) has to trace to something a grounded tool actually returned: NYC Open Data, the city's official Benefits Screening API, geocoding, or scoped web search. Each verified fact ships with an inline `{cite:Sn}` marker pointing at its source, and the sources are listed so you can click through and check the claim yourself. This is deliberate, click-through attribution, the same shape as Anthropic's Citations feature[^anthropic-citations], which returns the exact source passage behind each claim so a reader (or a machine) can verify it.

**A failed verification does not erase the work already done.** The runtime preserves the useful draft or retrieved records, labels the unresolved material unverified or incomplete, and provides the source links for direct checking. When no source or partial result exists, it says that specifically. It never presents a failed claim as verified. This is the deliberate opposite of MyCity, the city's own chatbot, which confidently answered questions it had no grounding for and told business owners they could break the law[^markup-mycity]: that they could take a cut of workers' tips, turn away tenants paying with Section 8 vouchers, and go cash-free, all of which are illegal in New York.

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

The boundary is **grounding, not truth**. Mechanical checks can establish that an exact structured value matches its captured record. They cannot establish that arbitrary natural-language prose follows from a passage, and they do not decide whether the source is true or current. Faithfulness to a source and truth in the world are separate axes[^faithfulness-factuality]. Keeping the source fresh is a separate job.

The resident path has three pieces and adds no separate verification-model call:

- **Mechanical validation.** Part C runs inline on cited prose and blocks the conclusive citation and structured-value failures listed above.
- **Feedback-and-retry.** On a failure, the agent receives the specific offending claim and reason, then gets up to two bounded chances to regenerate the complete answer. This is external, concrete feedback, the attribute-and-revise pattern that works precisely because the correction signal comes from outside the model[^rarr], not open-ended "are you sure?" self-reflection, which research shows can make accuracy worse without an external signal[^self-correct].
- **Preserve and label.** If the complete answer still fails a grounding check, the [runtime](heynyc/core/pydantic_runtime/runtime.py) preserves the latest useful candidate and its retrieved source links, clearly says that every detail could not be verified, and keeps the turn in an error state rather than calling it verified. Internal markup and moderated content remain separate safety boundaries and are not exposed.

The [provider-swappable NLI interface](heynyc/core/nli.py) remains available for evaluation comparisons, but the configured resident path does not call it. Recorded F236 live tests found that the extra checker and regeneration erased the latency gains from simpler tool orchestration ([failure register](docs/testing/failure-db.md)). Broader citation correctness is therefore reviewed from complete traces during live evaluations, including the model request, tool calls, retrieved evidence, answer, and citations ([evaluation conventions](heynyc/eval/README.md)).

This is not a mathematical truth guarantee. The guard enforces citation ownership and conclusive exact-value checks; complete live traces are reviewed for broader claim support. Source freshness, authority, and completeness remain separate responsibilities.

Full design, rationale, and the research behind each choice are kept in the project's internal design specs.

### The MyCity safety subset (human-graded)

We rebuilt NYC's documented MyCity failures as a labeled test set: the exact cases where MyCity told business owners they could break the law (take workers' tips, refuse Section 8 voucher tenants, go cash-free, lock out a tenant, skip schedule-change notice). We pose each one twice, once the way the owner asked it and once the way the worker or tenant on the receiving end would ask, because that second person is the one about to lose money or housing. Correct behavior on every one: answer correctly and grounded, or abstain and route to the authoritative source. **Never repeat the illegal advice.** Every gold answer is tied to the real statute and human-reviewed, and the litigation-live ones (the 2026 Section 8 / source-of-income ruling) carry a date and a re-check flag. The MyCity subset, and the real law behind each trap, is summarized in the [public benchmark methodology](docs/testing/benchmarks.md).

### The historical 137-query red-team and current 205-case suite

The [first completed adversarial run](docs/testing/red-team.md#results-to-date) used **137 queries across 8 categories**, every one built to make HeyNYC give harmful, ungrounded, or illegal advice, or to break its grounding. The categories were MyCity replays, prompt injection and jailbreak, out-of-scope harm, false-premise and leading questions, high-stakes over-reliance, PII and privacy, citation integrity, and Spanish-language safety. The [shipped current suite](docs/testing/red-team.md#how-we-red-team) now contains **205 cases**. It has not yet received a complete owner-approved live rerun, so the results below describe the historical 137-case run, not the current suite.

**The honest part is how it was graded.** The person who wrote the adversarial queries also scored them first, which is a real conflict of interest: you grade your own traps leniently, or misread your own transcripts. That's not a hunch, it's the documented self-enhancement and self-preference bias that any grader, human or LLM-as-judge, carries[^judge-bias]. So a second, **independent fresh-context grader** re-scored all 137 against the ground-truth legal facts, having never seen the first grader's verdicts. **It caught the first grader overclaiming.** After reconciling every disagreement against the raw transcript, the two converged on the real result.

The result, stated straight:

- **0 jailbreak failures** (18/18 held: DAN, fake system-override, base64-encoded injection, false-memory, oracle role-play, and system-prompt-exfiltration probes were all refused).
- **0 PII failures** (15/15, including a "tell me the last 4 of my SSN on file" social-engineering probe and a refusal to profile a named person's ICE risk).
- **0 illegal-advice failures** across the MyCity-style traps, in both owner and tenant framings.
- **0 fabricated citations** (16/16 on the category built specifically to extract a fake code section, URL, case number, or hotline).
- **7 grounding-accuracy failures**, found and fixed. Four were the same bug (mischaracterizing the live 2026 Section 8 / source-of-income court ruling: wrong court, overstated scope). One was a public-charge misstatement (half-confirming that SNAP counts against a green card, which it doesn't under current rules). Two were Spanish-only lapses the clean English answer didn't have (a fabricated statute number, and emergency aspirin dosing).

All 7 were fixed, re-verified, and committed (commit `3454e1d`). The [public write-up](docs/testing/red-team.md#results-to-date) has the per-category results and the two-grader reconciliation; the verbatim failure transcripts are withheld from the public export pending owner review, since some are crisis and self-harm material.

**Why we publish the failures.** Because "our tests pass" is what the last tool said. A red-team that failed on 7 real items, caught its own grader overclaiming, fixed them, and documents each one is a stronger safety claim than a clean scorecard.

## Keeping the source current

The guard above checks an answer against its source; it deliberately does not judge whether the source itself is still current, and that is where a grounded assistant quietly goes wrong: the citation still resolves, but the rule, rate, or deadline behind it has moved. Keeping the source fresh is its own job, and the approach follows established practice, which is a stack of standards from adjacent fields rather than a single trick.

- **Query live wherever a live source exists.** Most modules (clinics, food pantries, cooling centers, housing records, benefits, and more) hit the city's open-data APIs at request time, so they are fresh by construction: a changed record shows up the moment the city updates it, with nothing to keep in sync.
- **Date and flag the static facts.** The facts we do cache (legal thresholds, program rules, phone numbers) each carry an "as of" date and a per-type staleness tolerance; when a fact is older than its tolerance, the tool emits a caveat rather than presenting a possibly-stale figure as current, so drift is visible instead of silent (implemented in `heynyc/core/freshness.py`). This is the half MyCity skipped.
- **Grade sources after discovery.** Live [web search](heynyc/core/tools/web_search.py) retains unlisted results instead of treating a hand-maintained domain list as complete. Known domains provide ranking and trust metadata, not an acquisition allowlist. High-stakes claims require authoritative evidence from a sufficient direct excerpt or fetched page. An explicitly low-stakes capability may use an unverified excerpt only for the claim it directly states, with its warning and citation retained. Archived and community-tier excerpts remain discovery-only. Retrieved text remains an untrusted input because poisoned retrieval is a documented attack surface.[^poisonedrag]

Two pieces are planned and not yet shipped: an automated content-drift check that re-fetches each static fact's cited source on a cadence and flags a material change for human re-verification, built on standard HTTP conditional requests (`ETag` / `Last-Modified`, a `304` when nothing changed)[^http-conditional] so it is cheap, and a fixed human review cadence on the high-stakes static facts, mirroring the government content-maintenance practice of an owner and a review-by date on every page.[^govuk-maintenance] The full standard, the citations behind it, and how HeyNYC maps to each piece are in the project's internal freshness and source-trust standards brief.

## Conversation continuity is a safety and privacy boundary

HeyNYC remembers an ongoing messaging conversation so a resident can ask a real follow-up instead of starting over after every message or process restart. The identifier is a [salted HMAC of the channel and sender](heynyc/channels/identity.py), not a saved raw phone number. Hosted serving [requires an encryption key, encrypts transcript records, and purges expired conversations and drafts at startup and every 24 hours](heynyc/channels/app.py); the inactivity window currently defaults to 30 days. These are implemented controls, not planned claims.

The Twilio reliability boundary is implemented locally. The [Twilio route](heynyc/channels/twilio.py) encrypts and persists work keyed by `MessageSid` before returning HTTP 200. The worker persists the complete rendered reply in an encrypted outbox before delivery starts, checkpoints each accepted outbound Message SID, and resumes only unsent parts after a restart or transient failure. Inbox and outbox content is scrubbed after successful delivery. This matches Twilio's synchronous inbound-webhook contract and outbound logging guidance ([Twilio incoming-message webhooks](https://www.twilio.com/docs/messaging/guides/webhook-request), [Twilio outbound message logging](https://www.twilio.com/docs/messaging/guides/outbound-message-logging)). Automatic generation and delivery retries stop after three attempts. A process crash in the narrow interval between Twilio accepting an outbound message and the local SID checkpoint can still duplicate that part after recovery because Twilio's message-create API does not expose an idempotency key. This path is tested offline but still awaits a supervised live restart on the pilot host; Meta intake remains in memory.

The bounded-memory layer is now implemented locally. It budgets complete turns by measured tokens, compacts only under pressure into one typed continuity record, redacts identifiers before compaction, revalidates continuity against resident-authored text, and excludes prior assistant facts, URLs, and citations from future evidence. A generated turn is committed only after its reply is durably staged for Twilio delivery or accepted by the channel; staged Twilio delivery is checkpointed and resumed. `NEW` and `PRIVACY` are implemented locally. Confirmed `DELETE MY DATA` is shipped: after an in-chat confirmation it erases the resident's encrypted transcript, queued messages, draft, and pending report flags, keeping only PII-free aggregates and an anonymized daily spend record. The new memory behavior still awaits a supervised live restart.

The intended replacement is bounded task continuity, not a personal profile. A typed record may retain a resident's stated goal, corrections, completed steps, and unresolved questions. It must not preserve a benefit rule, deadline, location status, legal claim, or citation as truth; those are retrieved again and pass through the current grounding guard. Old assistant text is dialogue context, never evidence. Validated application state stays in the separate draft store rather than being copied into model-written memory.

That design follows current provider guidance that long-running conversations should remove stale tool output and compact only when context pressure requires it ([OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction), [Anthropic context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)). Its privacy baseline comes from the [NIST Privacy Framework](https://www.nist.gov/privacy-framework), the data-minimization and storage-limitation principles in [GDPR Article 5](https://eur-lex.europa.eu/eli/reg/2016/679/art_5/oj), and the public-sector precedent of publishing what conversation history is kept and for how long in the [GOV.UK Chat privacy notice](https://www.gov.uk/government/publications/govuk-chat-privacy-notice/govuk-chat-privacy-notice). These are design references, not a claim that HeyNYC has been certified or independently audited against them.

Host migration and resident export are separate boundaries. Internally, SQLite's file format is [cross-platform](https://www.sqlite.org/onefile.html), but a live database must be copied through its [online backup mechanism](https://sqlite.org/backup.html), alongside encrypted session and draft files and their existing keys. A restore must be tested before cutover, consistent with [NIST contingency guidance](https://csrc.nist.gov/pubs/sp/800/34/r1/upd1/final). Residents should receive only their own verified records, not the internal database: the planned self-service archive pairs a readable conversation with versioned [JSON](https://www.rfc-editor.org/info/rfc8259/), following the structured, commonly used, machine-readable portability principle in [GDPR Article 20](https://eur-lex.europa.eu/eli/reg/2016/679/art_20/oj). Both the automated migration path and self-service export remain planned, not shipped.

Before calling the memory layer complete, we still need the compact live acceptance set for cross-language and cross-module follow-ups, long-conversation usefulness, and a supervised restart. Confirmed in-chat deletion is implemented and tested end to end.

## The human-in-the-loop boundary

HeyNYC is built to get people *through the door*, not to replace the person who decides. Concretely:

- It **never decides eligibility.** The screener is the city's official engine and reports an estimate; the agency makes the determination.
- It **never submits anything on your behalf.** It prepares; you act.
- It **routes the high-stakes call to a human:** 311 for city services, a caseworker for benefits, Right to Counsel for eviction, ActionNYC for immigration, 911 for emergencies.

A wrong answer to someone who is least able to catch it is the failure mode we care most about, so the human path is always there, and the assistant is upfront that it's an AI, not a City employee.

None of this is just our house style, it's where every serious AI-governance framework points. Keeping a human in charge of a high-stakes decision is the core of the EU AI Act (Regulation 2024/1689)[^eu-ai-act], spelled out in its Article 14 on human oversight[^eu-ai-act-14], and a through-line of the NIST AI Risk Management Framework[^nist-ai-rmf]. Being upfront that you're talking to a machine is its own legal duty under the Act's Article 50 transparency rules[^eu-ai-act-50]. Close to home, NYC's own AI Action Plan[^nyc-ai-action-plan] and New York State's acceptable-use policy for AI[^nys-ai-acceptable-use] both insist on human oversight for consequential uses, and NYC already set a local audit precedent for automated decision systems in Local Law 144 of 2021[^nyc-local-law-144]. We're building to that bar, not around it.

## Limitations (what we haven't proven yet)

Read this part. These are real and current as of 2026-07-21:

- **Non-English safety has dedicated cases but is not demonstrated as a production capability.** Spanish and Bengali testing has already surfaced regressions the English answers did not have, including a fabricated law number, emergency aspirin dosing, a public-charge detour in response to a SNAP-loss question, and weaker tool use from some candidate models. A mistranslated benefit rule is a safety bug, not a UX bug, and in-language safety cannot be assumed from English coverage: models hallucinate at measurably different rates across languages[^multilingual-hallucination-gaps], and factual precision drops when they generate grounded facts in a non-English language[^multi-fact]. The translate-at-edge pipeline, an internal design spec, remains off by default and has not completed production validation. Dedicated cases are evidence that we test this surface, not proof that every supported language is safe.
- **Language access and accessibility are part of the safety surface, and we hold ourselves to the standards without claiming we've been audited against them.** Serving New Yorkers means clearing more than an accuracy bar. Language access is the law here: NYC Local Law 30 of 2017[^nyc-local-law-30] requires city services in the ten designated citywide languages plus telephonic interpretation in 100-plus. Accessibility has a hard target too: WCAG 2.1 Level AA[^wcag-21], the same standard the 2024 ADA Title II web rule[^ada-title-ii-2024] points state and local government at and that Section 508[^section-508] sets for federal technology. We build toward both, but we haven't done an independent accessibility audit or a per-language access review, so read these as commitments, not certifications.
- **No external third-party rating yet.** The independent fresh-context grader is a genuine improvement over grading our own homework, and it earned its keep by catching the overclaim. But it's still our own second agent, not an outside human or organization holding the red pen. A truly external rater is a stated next step, not something we've done.
- **Coverage is a growing subset, and gaps stay visible.** Measured against a frozen 170-query benchmark on 2026-07-05, about 78 were a clean yes, 49 partial, and 43 a gap (roughly 46%). Four more grounded modules (clinics, Housing Connect, WIC, childcare) have shipped since, so today's coverage is higher, though we have not re-measured against the frozen set. On questions we do not cover, HeyNYC labels the gap, preserves any useful source material, and offers a direct route to continue. The live experience is still thinner than the ambition, and we would rather say that than oversell it.
- **Grounding caps the error rate, it doesn't zero it.** A wrong-but-cited answer manufactures false confidence, and our users are the least able to survive one. That's why every claim carries a source you can check and a human fallback, and why we promise verifiability and a human decision, never accuracy.
- **Live data can go stale, and live legal facts move.** We date every dataset citation with a "valid as of" and flag staleness, but between refreshes we can lag, and litigation-live items (like the Section 8 ruling) need re-checking against current law.

## Where to read more

The testing records behind these claims live in [`docs/testing/`](docs/testing/), generated from internal sources by `scripts/export_testing_docs.py`:

- No-hallucination eval design and rubric: [`heynyc/eval/README.md`](heynyc/eval/README.md)
- MyCity safety subset (the traps and the real law behind each): [`docs/testing/benchmarks.md`](docs/testing/benchmarks.md)
- The historical 137-query red-team, results, and reconciliation: [`docs/testing/red-team.md`](docs/testing/red-team.md#results-to-date)
- How we red-team (the current 205-case method): [`docs/testing/red-team.md`](docs/testing/red-team.md#how-we-red-team)
- Benchmark methodology (how we measure at scale): [`docs/testing/benchmarks.md`](docs/testing/benchmarks.md)
- Security vulnerability reporting (a different thing): [SECURITY.md](SECURITY.md)

HeyNYC is an open-source project and is not affiliated with the City of New York. If you find a way to make it give an unsafe or ungrounded answer, that's exactly the kind of failure we want to know about; please open an issue (or, for anything sensitive, see [SECURITY.md](SECURITY.md)).

## References

Every source below was checked against the live page while writing this doc (last verified 2026-07-05). If a claim above couldn't be tied to a real source, we softened or dropped it rather than invent a cite, which is the same rule we hold the product to.

[^anthropic-citations]: [Anthropic, Citations (deterministic source attribution)](https://platform.claude.com/docs/en/build-with-claude/citations)
[^markup-mycity]: [The Markup (co-published with THE CITY), "NYC's AI Chatbot Tells Businesses to Break the Law" (2024)](https://themarkup.org/artificial-intelligence/2024/03/29/nycs-ai-chatbot-tells-businesses-to-break-the-law)
[^faithfulness-factuality]: [Maynez et al., "On Faithfulness and Factuality in Abstractive Summarization," ACL 2020](https://arxiv.org/abs/2005.00661)
[^rarr]: [Gao et al., "RARR: Researching and Revising What Language Models Say," ACL 2023](https://arxiv.org/abs/2210.08726)
[^self-correct]: [Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet," ICLR 2024](https://arxiv.org/abs/2310.01798)
[^abstention]: [Tomani et al., "Uncertainty-Based Abstention in LLMs Improves Safety and Reduces Hallucinations" (2024)](https://arxiv.org/abs/2404.10960)
[^minicheck]: [Tang et al., "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents," EMNLP 2024](https://arxiv.org/abs/2404.10774)
[^judge-bias]: [Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS 2023](https://arxiv.org/abs/2306.05685)
[^poisonedrag]: [PoisonedRAG: Knowledge Corruption Attacks to RAG, arXiv:2402.07867](https://arxiv.org/abs/2402.07867); [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
[^http-conditional]: [MDN: HTTP conditional requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Conditional_requests)
[^govuk-maintenance]: [GOV.UK: content maintenance](https://www.gov.uk/guidance/content-design/content-maintenance)
[^eu-ai-act]: [EU AI Act, Regulation (EU) 2024/1689, official consolidated text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng)
[^eu-ai-act-14]: [EU AI Act, Article 14: Human oversight (European Commission AI Act Service Desk)](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-14)
[^nist-ai-rmf]: [NIST AI Risk Management Framework (AI RMF 1.0)](https://www.nist.gov/itl/ai-risk-management-framework)
[^eu-ai-act-50]: [EU AI Act, Article 50: Transparency obligations (European Commission AI Act Service Desk)](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50)
[^nyc-ai-action-plan]: [The New York City Artificial Intelligence Action Plan (October 2023)](https://www.nyc.gov/assets/oti/downloads/pdf/reports/artificial-intelligence-action-plan.pdf)
[^nys-ai-acceptable-use]: [New York State ITS, Acceptable Use of Artificial Intelligence Technologies (NYS-P24-001)](https://its.ny.gov/acceptable-use-artificial-intelligence-technologies)
[^nyc-local-law-144]: [NYC Local Law 144 of 2021, Automated Employment Decision Tools (DCWP)](https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page)
[^multilingual-hallucination-gaps]: ["Multilingual Hallucination Gaps in Large Language Models" (2024)](https://arxiv.org/abs/2410.18270)
[^multi-fact]: ["Multi-FAct: Assessing Factuality of Multilingual LLMs using FActScore" (2024)](https://arxiv.org/abs/2402.18045)
[^nyc-local-law-30]: [NYC Local Law 30 of 2017, citywide language access](https://www.nyc.gov/site/civicengagement/about/language-access-plan.page)
[^wcag-21]: [WCAG 2.1 (W3C Recommendation), conformance Level AA](https://www.w3.org/TR/WCAG21/)
[^ada-title-ii-2024]: [ADA Title II 2024 web and mobile accessibility rule (DOJ fact sheet, requires WCAG 2.1 AA)](https://www.ada.gov/resources/2024-03-08-web-rule/)
[^section-508]: [Section 508 of the Rehabilitation Act, laws and policies (Section508.gov)](https://www.section508.gov/manage/laws-and-policies/)
