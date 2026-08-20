# Event retrieval and time normalization

_Verified 2026-08-20. Status: WOVEN._

## Retrieval verdict

A single broad query is not enough for event discovery. Tavily recommends splitting complex
retrieval into focused queries, running them concurrently, merging the results, and deduplicating
by URL. Its guidance also recommends keeping domain filters focused and using search followed by
page extraction when the result excerpt is insufficient. [Tavily search best
practices](https://docs.tavily.com/documentation/best-practices/best-practices-search)
[Tavily multi-query example](https://docs.tavily.com/examples/quick-tutorials/search-api)

The broad and focused lanes return independent rankings whose provider scores are not necessarily
comparable. Reciprocal rank fusion combines their ranks without score calibration and is the
recommended hybrid-search approach in Elastic. [Elastic hybrid
search](https://www.elastic.co/docs/solutions/search/hybrid-search) [Elastic RRF
reference](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)

HeyNYC therefore runs one open-web lane and one manifest-derived event-source lane concurrently,
deduplicates canonical URLs, and rank-fuses the two lists before the existing evidence labels and
resident-facing shortlist apply. The focused lane is additive, so an unknown organizer or venue
page can still enter through the open web. Source tier remains evidence metadata rather than a
claim that an event is more interesting. A generic discovery answer must also use at least one
relevant live-web candidate when that lane returned candidates. Every web candidate reaching the
answer first becomes a source-validated event record; incomplete time evidence produces `unknown`
rather than model arithmetic. This prevents the structured city and marketplace catalogs from
silently becoming the whole answer after retrieval succeeded.

## Time-model verdict

One-off events and recurring service hours share interval math but not the same source shape.
Google's event guidance keeps `startDate`, `endDate`, `eventStatus`, location, organizer, and offer
fields distinct, and says an unknown time should remain absent rather than being invented.
[Google event structured-data guidance](https://developers.google.com/search/docs/appearance/structured-data/event)

Recurring places and services instead use weekday opening periods, validity dates, and exceptions.
Schema.org defines `OpeningHoursSpecification` with `dayOfWeek`, `opens`, `closes`, `validFrom`, and
`validThrough`, including an explicit overnight rule when closing time is earlier than opening
time. [Schema.org OpeningHoursSpecification](https://schema.org/OpeningHoursSpecification) Google
Places separately exposes regular hours, current hours with special days, `openNow`, and next-open
or next-close timestamps. [Google Places opening-hours
model](https://developers.google.com/maps/documentation/places/web-service/reference/rest/v1/places)

HeyNYC therefore shares source clock parsing, New York timezone handling, and overnight interval
evaluation. Event adapters keep absolute intervals. Food, cooling, restroom, library, and benefits
adapters keep their provider-specific recurring schedules and exceptions. Missing exception data
continues to produce an honest schedule limitation rather than a claim of confirmed availability.

## Not adopted

- No vector index for live event pages. HeyNYC does not own a stable current-event corpus that
  would justify indexing and refresh machinery.
- No learned reranker. Rank fusion and the existing model selection are sufficient until labeled
  event judgments show a measurable ranking gap.
- No exhaustive domain whitelist. Known domains supply one focused recall lane and evidence
  metadata; they do not block the rest of the web.
