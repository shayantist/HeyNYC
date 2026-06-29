<!-- Thanks for contributing to HeyNYC! For a new module, see CONTRIBUTING.md + docs/MODULES.md -->

## Summary
<!-- What does this PR add/change? If it's a new module, name the service. -->

## Type
- [ ] New service module
- [ ] Fix / improvement to an existing module
- [ ] Core / harness / docs

## For a new module
- [ ] One folder under `heynyc/modules/<name>/` with `manifest.yaml`
- [ ] `keywords` reflect how real people phrase it
- [ ] If using a dataset, `field_map` matches the dataset's actual columns
- [ ] `prompt` says which tool to use, to cite sources, and **when to abstain**
- [ ] `eval.yaml` has ≥1 grounded case and ≥1 `abstain: true` case

## Test plan
<!-- Show it works. -->
- [ ] `uv run python -m heynyc modules` lists the module
- [ ] `uv run python -m pytest -q` is green
- [ ] (if it indexes pages) `uv run python -m heynyc index-build` succeeds and a `chat` query is grounded + cited

## Grounding confirmation
- [ ] No fabricated facts: every location/distance/hours/eligibility comes from a tool or dataset; otherwise the agent abstains and links out.
