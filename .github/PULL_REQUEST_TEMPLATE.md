<!-- Thanks for contributing to HeyNYC! For a new module, see CONTRIBUTING.md -->

## Summary
<!-- What resident or contributor problem does this change solve? -->

## Type
- [ ] New service module
- [ ] Fix / improvement to an existing module
- [ ] Core / harness / docs

## For a new module
- [ ] One folder under `heynyc/modules/<name>/` with `manifest.yaml`
- [ ] `examples` reflect how residents actually phrase the need
- [ ] If using a dataset, `field_map` matches the dataset's actual columns
- [ ] `prompt` names the operation, source boundaries, and limitations
- [ ] Typed inputs validate resident and model supplied constraints
- [ ] `eval.yaml` covers the normal path, a relevant source gap or failure, and any deterministic inverse

## Test plan
<!-- Show it works. -->
- [ ] `uv run python -m heynyc modules` lists the module
- [ ] `uv run python -m pytest -q` is green
- [ ] The focused live eval passes and a fresh-context reviewer has inspected the saved trace
- [ ] (if it indexes pages) `uv run python -m heynyc index-build` succeeds and a `chat` query is grounded + cited

## Grounding confirmation
- [ ] Verified claims come from retrieved evidence; partial or unsupported material stays labeled and keeps useful source links.
