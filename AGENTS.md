# AGENTS.md — Twilog Analytics (HTMX + FastAPI)

This repository builds a local-first web app to analyze and visualize Twilog CSV exports.
Frontend uses server-side rendered HTML with HTMX partial updates (no SPA).

## 1) Non-negotiables
- Stack: FastAPI + Jinja2 templates + HTMX (SSR + partial updates). No React/Vue.
- NLP: SudachiPy + sudachidict-full is required for tokenization/morphological analysis.
- Data processing: Prefer Polars. Pandas allowed only when Polars is inconvenient.
- Visualization:
  - Plotly for interactive charts where useful
  - Matplotlib for static charts and image outputs
  - WordCloud for word cloud images
- Must work with Twilog-like CSV variations (header/no-header, 3-5 columns).

## 2) Local run commands (authoritative)
- Install / sync deps: `uv sync`
- Run dev server: `uv run uvicorn app.main:app --reload`
- Run tests (if present): `uv run pytest -q`

## 3) Repository structure (authoritative)
- Web layer:
  - `app/main.py` FastAPI app entry
  - `app/routes/` route modules
  - `app/templates/` Jinja2 templates (base + pages + partials)
  - `app/static/` css/js assets
- Core logic (pure Python, no FastAPI imports):
  - `src/twilog_analytics/data/` loading + preprocessing
  - `src/twilog_analytics/analysis/` computations
  - `src/twilog_analytics/visualization/` chart/image generation

## 4) Work style rules
- Before large changes: produce a short implementation plan (files to touch + steps).
- Keep changes incremental and runnable at each step.
- Prefer pure functions in `src/` and thin route handlers in `app/`.
- Add type hints to public functions; keep modules small.

## 5) CSV contract & parsing rules
- Try to infer columns:
  - tweet_id (string)
  - created_at (parse `YYYY-MM-DD HH:MM:SS` if available)
  - text (tweet body)
  - status_url (optional)
- If parsing fails, show a user-friendly error page with hints.

## 6) HTMX conventions
- Use `hx-get` / `hx-post` for partial updates.
- Partials live in `app/templates/partials/`.
- Prefer returning HTML fragments for HTMX requests.
- Avoid heavy client-side JS; keep it minimal.

## 7) Output conventions
- Charts:
  - Plotly: render via HTML snippet or embed JSON + template
  - Matplotlib/WordCloud: generate PNG under a temp/cache directory and serve via route
- Tables:
  - Render HTML tables + provide CSV download endpoints.

## 8) Security & hygiene
- Never trust uploaded filenames; sanitize and store in temp dir.
- Do not execute user-provided content.
- Limit file size and validate content type where possible.

## 9) Definition of Done (for each feature)
- Has a route or function-level tests when feasible.
- Works on the provided sample Twilog CSV.
- No full page reload required for switching analysis panels (HTMX).