# Deploying to Hugging Face Spaces

This folder is a scaffold for a future Gradio app that visualizes
pre-computed results from `data/results/`. The app reads results via
`src/api.py` — it does not run any analysis itself.

## Architecture

```
HF Space (Gradio)
        │
        │ calls
        ▼
   src/api.py     ◄── reads data/results/{stationarity,sadf,gsadf,svadf}/*
        │
        ▼
   matplotlib Figure / pandas DataFrame
        │
        ▼
   rendered in the browser
```

Because `src/api.py` is pure (no global state, no I/O side effects beyond
reading), it can be called from any framework — Gradio, Streamlit, Flask,
FastAPI — without changes.

## Steps to deploy

1. **Generate results locally** by running `scripts/01-03` end-to-end.
2. **Commit `data/results/*` and `data/mc_cache/*`** to a Hugging Face
   dataset repo, or commit them directly with the Space.
3. **Copy `gradio_app.py.example` → `app.py`** in your HF Space repo.
4. **Add a `requirements.txt`** with at least:

       gradio>=4.0
       pandas
       numpy
       matplotlib
       pyarrow              # for parquet
       statsmodels          # only if running new analyses live
       yfinance             # only if running new analyses live
       numba                # only if running new analyses live

   If the Space only renders pre-computed results, omit the last four.

5. **Set the Space SDK to Gradio** and push.

The app will load on first request, list tickers from
`data/results/*`, and let the user pick a ticker → method → window.

## What `gradio_app.py.example` shows

A minimal Gradio interface with:
- a dropdown for ticker
- a dropdown for SV-ADF window
- a button to render the comparison figure
- a table panel showing the cross-method summary

It is intentionally minimal — extend it to match your thesis layout.
