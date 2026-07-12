# HeyNYC safety

_Last updated: 2026-07-11 (Phase 2 Tier-2 checker status). Limitations and reference links last verified 2026-07-05._

HeyNYC helps New Yorkers find, understand, and apply for government services. It answers questions about benefits, housing, food, and immigration, the places where a confident wrong answer can cost someone money, their home, or their immigration status. So the bar here isn't "usually right." The bar is: **every fact is grounded in an official source and cited, or the assistant says it doesn't know.** This doc is how we try to hit that bar and, just as importantly, where we still fall short. It's written to be read by a skeptic.

Looking for how to report a security vulnerability (a bug, a leak, an injection exploit)? That's a different doc: see [SECURITY.md](SECURITY.md). This one is about AI safety, whether the assistant gives safe, grounded, honest answers.

## The design: grounded, cite-or-abstain

The core rule is **no ungrounded facts.** Every specific the agent states (an address, a dollar amount, an eligibility rule, a deadline, a phone number) has to trace to something a grounded tool actually returned: NYC Open Data, the city's official Benefits Screening API, geocoding, or a scoped web search over trusted NYC domains. Each fact ships with an inline `{cite:Sn}` marker pointing at its source, and the sources are listed so you can click through and check the claim yourself. This is deliberate, click-through attribution, the same shape as [Anthropic's Citations feature](https://platform.claude.com/docs/en/build-with-claude/citations), which returns the exact source passage behind each claim so a reader (or a machine) can verify it.

**When nothing grounds an answer, it abstains.** It says it doesn't know and points you to the authoritative source (311, the right agency, a real human). This is the deliberate opposite of MyCity, the city's own chatbot, which confidently answered questions it had no grounding for and [told business owners they could break the law](https://themarkup.org/artificial-intelligence/2024/03/29/nycs-ai-chatbot-tells-businesses-to-break-the-law): that they could take a cut of workers' tips, turn away tenants paying with Section 8 vouchers, and go cash-free, all of which are illegal in New York.

Two boundaries fall out of this design:

- **It never makes an eligibility decision.** The screener runs the city's own official Benefits Screening API and reports "likely eligible per the city's engine, this is an estimate, the agency decides." It does not compute its own yes/no.
- **It routes specialized questions to a human.** Legal, medical, and immigration questions get routed (Right to Counsel, ActionNYC, 911, a caseworker) with a clear "I'm an AI, not a lawyer" disclaimer. It does not self-authorize as your lawyer.

## How we test it

### The no-hallucination eval (deterministic, blocks CI)

Every service module ships with its own eval, and a module isn't "done" until that eval is green. The floor is a set of **deterministic, code-only checks** that run offline with no model in the loop, so they're reproducible and hard to game:

- **attribution:** every asserted specific carries a citation.
- **faithfulness:** the cited snippet actually appears in a span the agent retrieved.
- **grounding:** specifics (address / date / dollar / eligibility) trace to a tool or retriever output.
- **link-liveness** and other structural checks.

Design and rubric: [`heynyc/eval/README.md`](heynyc/eval/README.md).

### The Part C cited-claim check

On top of that floor is a **deterministic cited-claim check** (we call it Part C): the verbatim facts in the answer, sitting right next to a citation marker, must actually occur in the source that marker points at. A phone number, a dollar amount, a street address, a proper name; if the answer states it and cites a source, that fact has to be *in* that source. It's a string match against the snapshot captured at query time (not a re-fetch), tuned hard to never false-fail a genuinely grounded answer and to block only when a fact's absence from its cited source is conclusive. This closes the one place the no-hallucination contract used to trust the model's own attribution instead of verifying it.

### The runtime verification guard (Phase 1 shipped; Phase 2 prototyped, off by default)

Part C above started as a deterministic grounding check running *offline*, as a CI gate. Phase 1 of the runtime guard moved it *online* (shipped in commit `f127a57`), so the same "is this fact actually in its cited source?" check now runs on every live answer before it reaches you. Being precise about what's shipped versus prototyped: **Phase 1 below is live; the Phase 2 NLI checker is now built as an off-by-default prototype, offline-tested, and not yet wired into the live loop.**

It's a **layered guard** built on one principle: **verify grounding, not truth.** We check "is this claim supported by the source it cited?" (a text-vs-text check that needs no world knowledge, so it's cheap and reliable, because [faithfulness to a source is a different axis than truth in the world](https://arxiv.org/abs/2005.00661)) and we deliberately do NOT put "is this true in the world?" on the live gate (that needs a current knowledge cutoff, and a stale judge would flag correct, up-to-date benefit info as wrong). Keeping the *source* fresh is a separate job.

**Phase 1 (shipped, commit `f127a57`)** is three pieces, all reusing code we already had, so it costs no new inference:

- **The deterministic grounding check (Tier 1), promoted from the eval harness to runtime.** This is Part C (above) running inline on every answer: every cited law/section number, URL, dollar/eligibility figure, date, and verbatim quote must occur in the specific source it's cited to, or it doesn't ship.
- **Feedback-and-retry.** On a failure, the agent gets the *specific* offending claim and reason back ("that dollar amount isn't in S3") for a targeted fix or re-retrieval, capped at a try or two. This is external, concrete feedback, the [attribute-and-revise pattern that works precisely because the correction signal comes from outside the model](https://arxiv.org/abs/2210.08726), not open-ended "are you sure?" self-reflection ([which the research shows makes accuracy worse when there's no external signal to act on](https://arxiv.org/abs/2310.01798)).
- **Abstain or hedge.** If a claim is still unverified after retries, we drop it or hedge it; if it's load-bearing, we abstain and route to 311 or a human. [Abstaining is the cheaper error](https://arxiv.org/abs/2404.10960): declining to answer when uncertain avoids roughly half of hallucinations and pushes safety way up, and for our users a confident wrong answer is the expensive one.

**Phase 2 (the Tier-2 NLI checker: prototype built, off by default, offline-tested, not yet in the live loop)** is a dedicated self-hosted faithfulness checker (a small [MiniCheck-class NLI model](https://arxiv.org/abs/2404.10774), which matches GPT-4 on grounding checks at a tiny fraction of the cost) for the prose claims the deterministic Tier-1 check can't parse. This is the piece that catches a fabricated law number cited to a soft web source, the exact failure class an open model like qwen showed in testing, and the one thing the Tier-1 string check is silent on. The checker and an off-by-default Tier-2 hook in the grounding guard are now built and exercised offline against that fabricated-statute case, so the wiring and the catch are demonstrated with no model in the loop. What's still pending before it can gate a live answer: validating the catch with a real MiniCheck-class model, calibrating the decision threshold so it never false-fails a genuinely grounded answer, and wiring it into the live guard. Until all three land it stays a demonstrated prototype, not a live check. Full design and the honest caveats: [`docs/superpowers/specs/2026-07-09-tier2-nli-checker-design.md`](docs/superpowers/specs/2026-07-09-tier2-nli-checker-design.md).

**Why this is a big deal for the rest of this doc:** this guard is what lets HeyNYC run safely on a **cheaper or self-hosted model.** The guarantee is "grounded or it abstains," and if the *architecture* carries that guarantee, the specific backend model doesn't have to. And because the checks are deterministic or run on a *local* model, verification adds no new data egress, which ties the safety story to the data-sovereignty story (no resident data has to leave government infrastructure to verify an answer).

Full design, rationale, and the research behind each choice: [`docs/superpowers/specs/2026-07-05-runtime-verification-guard-design.md`](docs/superpowers/specs/2026-07-05-runtime-verification-guard-design.md).

### The MyCity safety subset (human-graded)

We rebuilt NYC's documented MyCity failures as a labeled test set: the exact cases where MyCity told business owners they could break the law (take workers' tips, refuse Section 8 voucher tenants, go cash-free, lock out a tenant, skip schedule-change notice). We pose each one twice, once the way the owner asked it and once the way the worker or tenant on the receiving end would ask, because that second person is the one about to lose money or housing. Correct behavior on every one: answer correctly and grounded, or abstain and route to the authoritative source. **Never repeat the illegal advice.** Every gold answer is tied to the real statute and human-reviewed, and the litigation-live ones (the 2026 Section 8 / source-of-income ruling) carry a date and a re-check flag. Full subset, with the real law for each trap: [`docs/eval/benchmark-v2-safety.md`](docs/eval/benchmark-v2-safety.md).

### The 137-query adversarial red-team (with independent grading)

The bigger test is an adversarial red-team: **137 queries across 8 categories**, every one built to make HeyNYC give harmful, ungrounded, or illegal advice, or to break its grounding. The categories: MyCity replays, prompt injection / jailbreak, out-of-scope harm, false-premise / leading, high-stakes over-reliance, PII / privacy, citation-integrity, and Spanish-language safety.

**The honest part is how it was graded.** The person who wrote the adversarial queries also scored them first, which is a real conflict of interest: you grade your own traps leniently, or misread your own transcripts. That's not a hunch, it's the [documented self-enhancement and self-preference bias that any grader, human or LLM-as-judge, carries](https://arxiv.org/abs/2306.05685). So a second, **independent fresh-context grader** re-scored all 137 against the ground-truth legal facts, having never seen the first grader's verdicts. **It caught the first grader overclaiming.** After reconciling every disagreement against the raw transcript, the two converged on the real result.

The result, stated straight:

- **0 jailbreak failures** (18/18 held: DAN, fake system-override, base64-encoded injection, false-memory, oracle role-play, and system-prompt-exfiltration probes were all refused).
- **0 PII failures** (15/15, including a "tell me the last 4 of my SSN on file" social-engineering probe and a refusal to profile a named person's ICE risk).
- **0 illegal-advice failures** across the MyCity-style traps, in both owner and tenant framings.
- **0 fabricated citations** (16/16 on the category built specifically to extract a fake code section, URL, case number, or hotline).
- **7 grounding-accuracy failures**, found and fixed. Four were the same bug (mischaracterizing the live 2026 Section 8 / source-of-income court ruling: wrong court, overstated scope). One was a public-charge misstatement (half-confirming that SNAP counts against a green card, which it doesn't under current rules). Two were Spanish-only lapses the clean English answer didn't have (a fabricated statute number, and emergency aspirin dosing).

All 7 were fixed, re-verified, and committed (commit `3454e1d`). The full write-up, with every failure quoted verbatim so you can re-judge it yourself and the two-grader reconciliation table, is in [`docs/eval/red-team-v1.md`](docs/eval/red-team-v1.md).

**Why we publish the failures.** Because "our tests pass" is what the last tool said. A red-team that failed on 7 real items, caught its own grader overclaiming, fixed them, and shows you the transcript is a stronger safety claim than a clean scorecard.

## The human-in-the-loop boundary

HeyNYC is built to get people *through the door*, not to replace the person who decides. Concretely:

- It **never decides eligibility.** The screener is the city's official engine and reports an estimate; the agency makes the determination.
- It **never submits anything on your behalf.** It prepares; you act.
- It **routes the high-stakes call to a human:** 311 for city services, a caseworker for benefits, Right to Counsel for eviction, ActionNYC for immigration, 911 for emergencies.

A wrong answer to someone who is least able to catch it is the failure mode we care most about, so the human path is always there, and the assistant is upfront that it's an AI, not a City employee.

None of this is just our house style, it's where every serious AI-governance framework points. Keeping a human in charge of a high-stakes decision is the core of the [EU AI Act (Regulation 2024/1689)](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng), spelled out in its [Article 14 on human oversight](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-14), and a through-line of the [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework). Being upfront that you're talking to a machine is its own legal duty under the [Act's Article 50 transparency rules](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50). Close to home, [NYC's own AI Action Plan](https://www.nyc.gov/assets/oti/downloads/pdf/reports/artificial-intelligence-action-plan.pdf) and [New York State's acceptable-use policy for AI](https://its.ny.gov/acceptable-use-artificial-intelligence-technologies) both insist on human oversight for consequential uses, and NYC already set a local audit precedent for automated decision systems in [Local Law 144 of 2021](https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page). We're building to that bar, not around it.

## Limitations (what we haven't proven yet)

Read this part. These are real and current as of 2026-07-05:

- **Non-English safety is not separately evaluated yet.** The red-team's Spanish subset already surfaced regressions the English answers didn't have (a fabricated law number, an aspirin-dosing lapse). A mistranslated benefit rule is a safety bug, not a UX bug, and in-language safety can't be assumed from English coverage: [models hallucinate at measurably different rates across languages](https://arxiv.org/abs/2410.18270), and [factual precision drops when they generate grounded facts in a non-English language](https://arxiv.org/abs/2402.18045). We don't yet have a dedicated non-English safety eval, and prompt text only partly closes what is really a model-level gap. We've since prototyped an architectural fix for exactly this: a [translate-at-edge pipeline](docs/superpowers/specs/2026-07-05-multilingual-translate-at-edge-design.md) that grounds and verifies in English, then translates only the finished answer with law numbers, dollar amounts, and citations frozen and copied through verbatim, backed by an entity round-trip check that falls back to the English answer on any mismatch. It's off by default and not yet validated against a real translator, so it doesn't close this gap yet, but it's the shape of the fix. This is the limitation we're most honest about, because our actual audience is largely non-English speakers.
- **Language access and accessibility are part of the safety surface, and we hold ourselves to the standards without claiming we've been audited against them.** Serving New Yorkers means clearing more than an accuracy bar. Language access is the law here: [NYC Local Law 30 of 2017](https://www.nyc.gov/site/civicengagement/about/language-access-plan.page) requires city services in the ten designated citywide languages plus telephonic interpretation in 100-plus. Accessibility has a hard target too: [WCAG 2.1 Level AA](https://www.w3.org/TR/WCAG21/), the same standard the [2024 ADA Title II web rule](https://www.ada.gov/resources/2024-03-08-web-rule/) points state and local government at and that [Section 508](https://www.section508.gov/manage/laws-and-policies/) sets for federal technology. We build toward both, but we haven't done an independent accessibility audit or a per-language access review, so read these as commitments, not certifications.
- **No external third-party rating yet.** The independent fresh-context grader is a genuine improvement over grading our own homework, and it earned its keep by catching the overclaim. But it's still our own second agent, not an outside human or organization holding the red pen. A truly external rater is a stated next step, not something we've done.
- **Coverage is a growing subset, and we abstain on the rest.** Measured against a frozen 170-query benchmark on 2026-07-05, about 78 were a clean yes, 49 partial, and 43 a gap (roughly 46%). Four more grounded modules (clinics, Housing Connect, WIC, childcare) have shipped since, so today's coverage is higher, though we have not re-measured against the frozen set. On the questions we don't cover, the safe-and-correct behavior is to abstain and route, which we do, so the live experience is still thinner than the ambition. We'd rather say that than oversell it.
- **Grounding caps the error rate, it doesn't zero it.** A wrong-but-cited answer manufactures false confidence, and our users are the least able to survive one. That's why every claim carries a source you can check and a human fallback, and why we promise verifiability and a human decision, never accuracy.
- **Live data can go stale, and live legal facts move.** We date every dataset citation with a "valid as of" and flag staleness, but between refreshes we can lag, and litigation-live items (like the Section 8 ruling) need re-checking against current law.

## Where to read more

- No-hallucination eval design and rubric: [`heynyc/eval/README.md`](heynyc/eval/README.md)
- MyCity safety subset (human-graded gold answers): [`docs/eval/benchmark-v2-safety.md`](docs/eval/benchmark-v2-safety.md)
- The 137-query adversarial red-team and reconciliation: [`docs/eval/red-team-v1.md`](docs/eval/red-team-v1.md)
- Benchmark methodology (how we measure at scale): [`docs/eval/benchmark-methodology.md`](docs/eval/benchmark-methodology.md)
- Security vulnerability reporting (a different thing): [SECURITY.md](SECURITY.md)

HeyNYC is an open-source project and is not affiliated with the City of New York. If you find a way to make it give an unsafe or ungrounded answer, that's exactly the kind of failure we want to know about; please open an issue (or, for anything sensitive, see [SECURITY.md](SECURITY.md)).

## References

Every source below was checked against the live page while writing this doc (last verified 2026-07-05). If a claim above couldn't be tied to a real source, we softened or dropped it rather than invent a cite, which is the same rule we hold the product to.

**The MyCity failure we're built against**

- [The Markup (co-published with THE CITY), "NYC's AI Chatbot Tells Businesses to Break the Law" (2024)](https://themarkup.org/artificial-intelligence/2024/03/29/nycs-ai-chatbot-tells-businesses-to-break-the-law)

**Grounding, faithfulness, and cite-or-abstain**

- [Maynez et al., "On Faithfulness and Factuality in Abstractive Summarization," ACL 2020](https://arxiv.org/abs/2005.00661)
- [Tang et al., "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents," EMNLP 2024](https://arxiv.org/abs/2404.10774)
- [Anthropic, Citations (deterministic source attribution)](https://platform.claude.com/docs/en/build-with-claude/citations)
- [Tomani et al., "Uncertainty-Based Abstention in LLMs Improves Safety and Reduces Hallucinations" (2024)](https://arxiv.org/abs/2404.10960)

**Independent grading and self-correction**

- [Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS 2023](https://arxiv.org/abs/2306.05685)
- [Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet," ICLR 2024](https://arxiv.org/abs/2310.01798)
- [Gao et al., "RARR: Researching and Revising What Language Models Say," ACL 2023](https://arxiv.org/abs/2210.08726)

**Human oversight and AI disclosure**

- [EU AI Act, Regulation (EU) 2024/1689, official consolidated text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng)
- [EU AI Act, Article 14: Human oversight (European Commission AI Act Service Desk)](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-14)
- [EU AI Act, Article 50: Transparency obligations (European Commission AI Act Service Desk)](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50)
- [NIST AI Risk Management Framework (AI RMF 1.0)](https://www.nist.gov/itl/ai-risk-management-framework)
- [The New York City Artificial Intelligence Action Plan (October 2023)](https://www.nyc.gov/assets/oti/downloads/pdf/reports/artificial-intelligence-action-plan.pdf)
- [New York State ITS, Acceptable Use of Artificial Intelligence Technologies (NYS-P24-001)](https://its.ny.gov/acceptable-use-artificial-intelligence-technologies)
- [NYC Local Law 144 of 2021, Automated Employment Decision Tools (DCWP)](https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page)

**Language access and accessibility**

- [NYC Local Law 30 of 2017, citywide language access](https://www.nyc.gov/site/civicengagement/about/language-access-plan.page)
- [WCAG 2.1 (W3C Recommendation), conformance Level AA](https://www.w3.org/TR/WCAG21/)
- [ADA Title II 2024 web and mobile accessibility rule (DOJ fact sheet, requires WCAG 2.1 AA)](https://www.ada.gov/resources/2024-03-08-web-rule/)
- [Section 508 of the Rehabilitation Act, laws and policies (Section508.gov)](https://www.section508.gov/manage/laws-and-policies/)

**Multilingual faithfulness limits**

- ["Multilingual Hallucination Gaps in Large Language Models" (2024)](https://arxiv.org/abs/2410.18270)
- ["Multi-FAct: Assessing Factuality of Multilingual LLMs using FActScore" (2024)](https://arxiv.org/abs/2402.18045)
