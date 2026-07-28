# Immigration support: verified source and design record

_Verified 2026-07-27. Status: WOVEN._

## Resident need

The Supreme Court decided `Mullin v. Doe` on June 25, 2026. The case concerned
challenges to the termination of Temporary Protected Status for people from
Haiti and Syria, and the Court reversed the interim relief that had postponed
the terminations. The decision did not decide any resident's individual status
or next legal step. Those are separate questions for an immigration lawyer or
the responsible federal agency. [Supreme Court opinion](https://www.supremecourt.gov/opinions/25pdf/25-1083_f204.pdf)

NYC responded the same day by directing residents with status or legal-help
questions to the MOIA Immigration Legal Support Hotline at 800-354-0365. The
city describes that help as free and confidential. [Mayor's statement](https://www.nyc.gov/mayors-office/news/2026/06/statement-from-mayor-mamdani-on-supreme-court-decision-allowing-)

MOIA's current legal-resource page also directs immigrant New Yorkers to the
hotline or to call 311 and say "Immigration Legal." Its Know Your Rights page
publishes material in Arabic, Bangla, Haitian Creole, Spanish, and other
languages, including resources for encounters at home or work and for a
detained friend or family member. [MOIA legal help](https://www.nyc.gov/site/immigrants/legal-resources/legal-resources.page)
[MOIA Know Your Rights](https://www.nyc.gov/site/immigrants/legal-resources/know-your-rights.page)

The landing page is a directory, not the evidence for the individual rights
claims. Runtime retrieval therefore targets the city's claim-bearing
[English booklet](https://www.nyc.gov/assets/immigrants/downloads/pdf/KYR-with-ICE_February-2025_English.pdf),
[Haitian Creole booklet](https://www.nyc.gov/assets/immigrants/downloads/pdf/KYR-with-ICE-2026-Haitian-Creole.pdf),
and [Arabic booklet](https://www.nyc.gov/assets/immigrants/downloads/pdf/KYR-with-ICE-2026-Arabic.pdf)
directly.

New York State's July 10, 2026 support notice says residents affected by the
Haiti and Syria TPS changes can call the Office for New Americans hotline at
1-800-566-7636 for anonymous, multilingual legal and service referrals.
[New York State support notice](https://opwdd.ny.gov/news/important-resources-available-nyers-affected-recent-federal-temporary-status-revocation)

The previously current USCIS Haiti and Syria TPS URLs redirected to archived
pages during verification. Archived pages remain historical evidence, but the
runtime labels them archived and must not present them as current agency
posture. [USCIS Haiti archive](https://www.uscis.gov/archive/temporary-protected-status-designated-country-haiti)
[USCIS Syria archive](https://www.uscis.gov/archive/temporary-protected-status-designated-country-syria)

## Reporting and sanctuary limits

NYC Executive Order 13 and the later audit address how city agencies handle
city resources, property, data, and requests involving civil immigration
enforcement. They do not stop federal agents from acting throughout the city,
so HeyNYC must not describe sanctuary policy as a guarantee of safety.
[Executive Order 13](https://www.nyc.gov/mayors-office/news/2026/02/executive-order-13)
[Executive Order 13 audit](https://www.nyc.gov/mayors-office/news/2026/05/mayor-mamdani-releases-executive-order-13-report-of-audit-findin)

New York's Attorney General provides a form for reporting federal government
action in the state. The form says that filing does not create a complaint or
lawsuit, contact information is optional, and submitted information or media
may be used in public documents or legal proceedings or shared for other lawful
reasons. HeyNYC should explain those terms briefly and link the form. It should
not collect the report, media, or identifying details itself.
[NY Attorney General reporting form](https://ag.ny.gov/federal-actions-form)

## Implementation ruling

The module uses the repository's existing `SituationHint` and
`official_sources` mechanisms. It adds no module-specific tool, classifier, or
static legal-fact table.

- TPS changes, court posture, legal-help routes, rights, and sanctuary scope
  are retrieved from official sources during the turn.
- The assistant separates a court decision, agency implementation, and an
  individual's status.
- It does not decide a resident's status, predict deportation, coach evasion,
  identify live enforcement locations, or call any place safe.
- It can explain and link an official reporting form, but it does not collect,
  store, map, publish, or submit a report.
- Haitian Creole and Arabic cases accompany English cases for TPS changes and
  live enforcement encounters.

This follows the project's retrieval-first rule and avoids a second legal
workflow that would drift independently of the official form.

## Source register

| Source | Runtime purpose | Verified |
|---|---|---|
| [Supreme Court opinion](https://www.supremecourt.gov/opinions/25pdf/25-1083_f204.pdf) | Court holding and date | 2026-07-27 |
| [Mayor's statement](https://www.nyc.gov/mayors-office/news/2026/06/statement-from-mayor-mamdani-on-supreme-court-decision-allowing-) | City response and current legal-help route | 2026-07-27 |
| [MOIA legal resources](https://www.nyc.gov/site/immigrants/legal-resources/legal-resources.page) | Free legal-help route | 2026-07-27 |
| [MOIA Know Your Rights](https://www.nyc.gov/site/immigrants/legal-resources/know-your-rights.page) | Current multilingual rights material | 2026-07-27 |
| [MOIA English Red Card](https://www.nyc.gov/assets/immigrants/downloads/pdf/EN-Red-Card-Cutout-Printable.pdf) | Short rights script for an enforcement encounter | 2026-07-27 |
| [MOIA Haitian Creole Red Card](https://www.nyc.gov/assets/immigrants/downloads/pdf/kyr_red_card_haitian_creole.pdf) | Short Haitian Creole rights script | 2026-07-27 |
| [MOIA English Know Your Rights booklet](https://www.nyc.gov/assets/immigrants/downloads/pdf/KYR-with-ICE_February-2025_English.pdf) | Claim-bearing home and workplace rights | 2026-07-27 |
| [MOIA Haitian Creole Know Your Rights booklet](https://www.nyc.gov/assets/immigrants/downloads/pdf/KYR-with-ICE-2026-Haitian-Creole.pdf) | Claim-bearing Haitian Creole rights | 2026-07-27 |
| [MOIA Arabic Know Your Rights booklet](https://www.nyc.gov/assets/immigrants/downloads/pdf/KYR-with-ICE-2026-Arabic.pdf) | Claim-bearing Arabic rights | 2026-07-27 |
| [MOIA Haitian Response Initiative](https://www.nyc.gov/site/immigrants/legal-resources/haitian-response-initiative.page) | Haitian community support | 2026-07-27 |
| [New York State support notice](https://opwdd.ny.gov/news/important-resources-available-nyers-affected-recent-federal-temporary-status-revocation) | Current multilingual legal and service referral route | 2026-07-27 |
| [USCIS Haiti archive](https://www.uscis.gov/archive/temporary-protected-status-designated-country-haiti) | Historical agency posture only | 2026-07-27 |
| [USCIS Syria archive](https://www.uscis.gov/archive/temporary-protected-status-designated-country-syria) | Historical agency posture only | 2026-07-27 |
| [Executive Order 13](https://www.nyc.gov/mayors-office/news/2026/02/executive-order-13) | City-agency policy scope | 2026-07-27 |
| [Executive Order 13 audit](https://www.nyc.gov/mayors-office/news/2026/05/mayor-mamdani-releases-executive-order-13-report-of-audit-findin) | Implementation and limits | 2026-07-27 |
| [NY Attorney General enforcement announcement](https://ag.ny.gov/press-release/2026/attorney-general-james-and-governor-hochul-announce-first-enforcement-action-new) | Current state enforcement context | 2026-07-27 |
| [NY Attorney General reporting form](https://ag.ny.gov/federal-actions-form) | Resident-controlled reporting route and terms | 2026-07-27 |
| [Federal Register Syria notice](https://public-inspection.federalregister.gov/2025-18322.pdf) | Primary federal designation notice | 2026-07-27 |

## Not built

- No ICE tracker, sightings feed, map, geofence, or crowdsourcing
- No browser automation or automatic form submission
- No storage of resident media or identifying details
- No static answer asserting whether a particular resident still has TPS
