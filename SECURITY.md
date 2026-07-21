# Security Policy

HeyNYC is a civic project that helps New Yorkers find, understand, and apply for government services. The optional benefits application-draft workflow can handle resident personal information (PII), so we take security and privacy seriously and welcome responsible disclosure. The default screening path is PII-free, and forms are off unless explicitly enabled.

This is the security policy, how to report a vulnerability. For how the assistant itself stays grounded and safe (guardrails, red-team, abstention), see [SAFETY.md](SAFETY.md).

## Reporting a vulnerability

**Please do not open a public GitHub issue for security bugs.** A public report tips off attackers before a fix is available and, for a tool that touches resident data, could put real people at risk.

Instead, report vulnerabilities **privately** to **shayan@reach4help.org**. If GitHub private vulnerability reporting is enabled on this repository, you may also use **Security → Report a vulnerability**.

When you report, please include (as much as you can):

- a description of the issue and its potential impact,
- steps to reproduce or a proof of concept,
- affected version/commit, and
- any suggested remediation.

We will acknowledge your report, investigate, and keep you updated on remediation. Please give us a reasonable window to release a fix before any public disclosure, and act in good faith: avoid privacy violations, data destruction, or service disruption while testing.

## Scope

In scope: the code in this repository (the agent core, service modules, the eval harness, and the messaging channels), and any handling of secrets or resident PII in those paths.

Especially valuable:

- exposure or logging of PII (the screening path is PII-free; the optional
  application draft is encrypted at rest when `HEYNYC_PII_KEY` is configured, and uses `HEYNYC_PII_RETENTION_DAYS` for cleanup),
- prompt-injection or grounding bypasses that could cause the agent to emit
  ungrounded/fabricated guidance on regulated topics (benefits, eligibility),
- secret/credential leakage, authentication/webhook-signature bypasses, and
  injection or arbitrary-file-access issues.

Out of scope: third-party services HeyNYC integrates with (e.g. NYC Open Data, the NYC Benefits Screening API, Ticketmaster, messaging providers). Report those to the respective owners. Missing hardening on a purely local dev setup is lower priority; note it, but it is not treated as an active vulnerability.

## A note on data

HeyNYC is not affiliated with the City of New York. It does not submit anything on a resident's behalf. Channel senders are reduced to a salted, non-reversible key, and the optional draft workflow is intended for transient, consented application data. If a report involves resident data, please handle any examples with care and redact real PII from your write-up.
