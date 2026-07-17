# HeyNYC Privacy Notice

**Effective date:** July 17, 2026

This notice explains how [Reach4Help](https://reach4help.org/) handles information when you use HeyNYC by SMS, WhatsApp, command line, or another supported channel. It supplements the general [Reach4Help Privacy Policy](https://reach4help.org/privacy/). If the two notices conflict about HeyNYC, this HeyNYC notice controls.

## 1. Information HeyNYC handles

Depending on how you use the service, HeyNYC may handle:

- Your phone number or messaging address, profile name supplied by the messaging provider, message identifier, channel, and delivery metadata
- The questions, feedback, and other content you send, including locations or personal circumstances you choose to include
- HeyNYC's responses, cited sources, tool activity, model usage, latency, and error information
- Information you deliberately provide while preparing an application or service request

Do not send Social Security numbers, immigration numbers, payment-card numbers, passwords, medical records, or other sensitive identifiers through ordinary chat.

## 2. How we use information

Reach4Help uses this information to:

- Respond to your questions and maintain the context of your conversation
- Find relevant public services, locations, rules, alerts, and events
- Prepare information or application drafts when you explicitly request that feature
- Prevent duplicate messages, enforce rate limits, secure the service, diagnose failures, and measure cost and performance
- Review feedback you deliberately submit and improve safety and accuracy
- Comply with law and protect users, Reach4Help, and the public

Reach4Help does not sell personal information or use HeyNYC messages for advertising. Mobile information, text-message consent, and opt-in records are not shared with third parties or affiliates for their own marketing or promotional purposes.

## 3. Phone-number protection and conversation records

HeyNYC's application [converts a sender address into a salted pseudonymous identifier](../heynyc/channels/identity.py) before saving application-level sessions, telemetry, drafts, or feedback. The application does not save the raw phone number in those local records. The messaging provider still receives and processes the phone number so it can deliver messages.

HeyNYC saves conversation text so the assistant can understand follow-up questions and avoid appearing to forget an ongoing conversation. Before a resident-answer request reaches the answer model, the current pilot [uses a measured context budget](../heynyc/core/memory.py), keeps complete recent turns, and may replace older turns with a typed continuity record. The smaller scope-classification preflight is a separate model call and is not covered by that answer-context budget. Prior assistant prose is not treated as evidence for new factual claims. In-progress application drafts may also be saved so you can return and finish them. Public hosted deployments [require encryption and run a scheduled deletion process](../heynyc/channels/app.py) for local conversation and draft files after the configured inactivity period, which defaults to 30 days. Pseudonymous operational metrics and [encrypted, pattern-redacted feedback](../heynyc/channels/analytics.py) may be kept longer when needed for security, evaluation, and service improvement.

No security or deletion method is perfect. Reach4Help limits what the application stores, restricts access, and uses technical safeguards appropriate to the pilot, but cannot guarantee absolute security.

## 4. Standards, evidence, and current gaps

HeyNYC uses the [NIST Privacy Framework](https://www.nist.gov/privacy-framework) for data minimization and lifecycle management, the storage-limitation principle in [GDPR Article 5](https://eur-lex.europa.eu/eli/reg/2016/679/art_5/oj), and the [GOV.UK Chat privacy notice](https://www.gov.uk/government/publications/govuk-chat-privacy-notice/govuk-chat-privacy-notice) as public-sector design references. Current model-provider guidance also recommends removing stale context and compacting long conversations when needed ([OpenAI](https://developers.openai.com/api/docs/guides/compaction), [Anthropic](https://platform.claude.com/docs/en/build-with-claude/context-editing)). These references guide the pilot; they do not mean HeyNYC is certified or independently audited against those frameworks.

What the current code demonstrates:

- Raw sender addresses are replaced with a [salted pseudonymous identifier](../heynyc/channels/identity.py) before application-level persistence
- Hosted serving fails without a valid encryption key and [runs expiry at startup and daily](../heynyc/channels/app.py)
- Conversation records are [encrypted per record with authenticated encryption](../heynyc/core/session.py)
- Resident-authored feedback is [pattern-redacted before encrypted storage](../heynyc/channels/analytics.py)

What is not implemented yet:

- In-chat `DELETE MY DATA`; [`NEW` starts a fresh conversation and `PRIVACY` explains current practices](../heynyc/channels/orchestrator.py), while deletion currently requires email
- A settled retention and deletion policy for longer-lived pseudonymous telemetry
- Live semantic acceptance of memory compaction across long multilingual conversations
- An external privacy, security, accessibility, or standards-compliance audit

## 5. Service providers and public data sources

Reach4Help uses service providers to operate HeyNYC. Depending on the deployed configuration, message content and related data may be processed by:

- [Twilio](https://www.twilio.com/en-us/legal/privacy) or [Meta](https://www.facebook.com/privacy/policy/) to receive and deliver SMS or WhatsApp messages
- [OpenAI](https://openai.com/policies/privacy-policy/) or [Anthropic](https://www.anthropic.com/legal/privacy) to generate or evaluate automated responses
- Hosting, monitoring, mapping, geocoding, search, and document-generation providers needed to answer a request
- New York City, New York State, federal, transit, cultural, or other identified data sources queried to answer your question

These providers may keep their own service logs under their privacy notices and legal obligations. HeyNYC may send a query, voluntarily supplied location, or other necessary request details to a relevant provider. Reach4Help does not authorize these providers to use your text-message consent for their own marketing.

## 6. Government agencies

Using HeyNYC does not create an account with a government agency. Under the project's [resident-control rule](../SAFETY.md), Reach4Help does not silently submit applications, complaints, or service requests. If a future feature prepares information for an agency, you will receive a review step and must expressly authorize any submission. An agency that receives information you choose to submit will handle it under that agency's own privacy rules.

## 7. Your choices

You can stop SMS messages by replying **STOP**. Reply **HELP** for messaging help. STOP is a messaging opt-out, not a deletion request. You can also stop using the service at any time.

To request access to or deletion of information held directly by Reach4Help, email [privacy@reach4help.org](mailto:privacy@reach4help.org). Include enough information for us to locate the relevant record without sending sensitive identifiers by email. Reach4Help may need to retain limited information for security, legal compliance, or an unresolved request. Requests concerning a messaging, model, or government provider may also be subject to that provider's procedures.

## 8. Children

HeyNYC is not designed to knowingly collect sensitive personal information from children under 13 without involvement from a parent or guardian. If you believe a child has provided such information, contact [privacy@reach4help.org](mailto:privacy@reach4help.org).

## 9. Changes and contact

Reach4Help may update this notice as HeyNYC's providers or features change. The effective date above will identify the current version.

For privacy questions or requests, email [privacy@reach4help.org](mailto:privacy@reach4help.org). For service help, email [support@reach4help.org](mailto:support@reach4help.org).
