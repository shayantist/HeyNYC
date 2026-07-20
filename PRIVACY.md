# Privacy

HeyNYC answers questions about NYC services over SMS and WhatsApp. Here is what happens to your messages, stated plainly.

**What we process.** Your messages are carried by the SMS/WhatsApp network (Twilio) like any text you send, and the conversation context needed for a reply is sent to our configured AI model provider to generate the answer. That processing is what makes the service work; no service like this can truthfully promise your words touch no infrastructure.

**What we keep.** An encrypted conversation transcript and any in-progress application draft, for the configured retention period, on our infrastructure. Known sensitive identifiers are redacted before storage. We don't build a profile on you, and the eligibility-screening flow is PII-free by design. Don't paste an SSN or similar sensitive ID into the chat; the assistant will tell you the same.

**Who reads it.** No person reads conversations in the normal course of operating the service. A human sees an exchange in exactly two cases: you send it to us with the REPORT command, which asks you to confirm first and shares only that one exchange; or a safety or abuse situation requires review. We state this as policy rather than impossibility, because the operator of any encrypted service holds its keys, and we would rather be straight about that.

**Operator access.** The people who run HeyNYC decrypt a real resident's conversation in exactly two cases: you sent an exchange to us with REPORT and confirmed, or a specific safety or abuse review requires it. Debugging and development use the operators' own test conversations, never yours. Curiosity is not a case.

**Your controls.** NEW starts a fresh conversation the assistant no longer sees earlier messages of (the audit record is retained). PRIVACY returns this policy in short form. REPORT flags your last exchange for human review, with your confirmation. DELETE MY DATA erases your data: after you confirm, it deletes your encrypted conversation transcript, any in-progress application draft, and any pending report flags. What survives is only PII-free aggregate service statistics and an anonymized daily spend record kept for abuse control, neither of which identifies you. It cannot be undone.

**Spend and abuse limits.** Per-person daily usage caps protect the service without cutting anyone off from the deterministic emergency guidance, which always works.

_Last updated: 2026-07-18. This document changes only to become more accurate, never more flattering._
