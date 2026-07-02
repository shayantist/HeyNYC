# Security Policy

HeyNYC is a civic project that helps New Yorkers find, understand, and apply for
government services. As it grows toward handling resident personal information
(PII) — for example, when assisting with a benefits application — we take
security and privacy seriously and welcome responsible disclosure.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security bugs.** A public report
tips off attackers before a fix is available and, for a tool that touches
resident data, could put real people at risk.

Instead, report vulnerabilities **privately** to `<maintainer email>`. If
GitHub private vulnerability reporting is enabled on this repository, you may
also use **Security → Report a vulnerability**.

When you report, please include (as much as you can):

- a description of the issue and its potential impact,
- steps to reproduce or a proof of concept,
- affected version/commit, and
- any suggested remediation.

We will acknowledge your report, investigate, and keep you updated on
remediation. Please give us a reasonable window to release a fix before any
public disclosure, and act in good faith — avoid privacy violations, data
destruction, or service disruption while testing.

## Scope

In scope: the code in this repository (the agent core, service modules, the eval
harness, and the messaging channels), and any handling of secrets or resident
PII in those paths.

Especially valuable:

- exposure or logging of PII (the design intent is that the screening path is
  PII-free and application PII is never logged, never sent to third parties),
- prompt-injection or grounding bypasses that could cause the agent to emit
  ungrounded/fabricated guidance on regulated topics (benefits, eligibility),
- secret/credential leakage, authentication/webhook-signature bypasses, and
  injection or arbitrary-file-access issues.

Out of scope: third-party services HeyNYC integrates with (e.g. NYC Open Data,
the NYC Benefits Screening API, Ticketmaster, messaging providers) — report
those to the respective owners. Missing hardening on a purely local dev setup is
lower priority; note it, but it is not treated as an active vulnerability.

## A note on data

HeyNYC is not affiliated with the City of New York. It never submits anything on
a resident's behalf and never stores raw phone numbers (senders are reduced to a
salted, non-reversible key). If a report involves resident data, please handle
any examples with care and redact real PII from your write-up.
