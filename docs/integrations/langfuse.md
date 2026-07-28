# Langfuse evaluation layer

QuickVoice's AI service (`apps/ai`) can send every finished call, plus RAG
retrievals, to [Langfuse](https://langfuse.com) for tracing and evaluation.
The integration is opt-in and fails safe: if Langfuse is not configured or
unreachable, calls run exactly as before.

## What gets sent

For every finished call (unless the agent has `zero_pii_retention` enabled),
`handlers/langfuse_handler.py` creates one Langfuse trace per call:

- **Trace metadata**: agent id, organization id, provider, call direction,
  to/from number, duration.
- **One observation per transcript turn** (user and agent), in order.
- **Heuristic evaluation scores**, computed locally with no extra LLM calls:
  - `has_transcript` — did the call produce any transcript at all
  - `agent_responded` — did the agent ever speak
  - `conversation_balance` — did both sides participate (flags monologues
    and likely dropped calls)
  - `call_duration_seconds` — raw duration, useful for filtering/sorting

RAG (Pinecone) lookups in `handlers/rag_handler.py` are logged as a separate
`quickvoice-rag-retrieval` trace per query, tagged with the agent id, so
knowledge-base hit/miss/error rates and latency are inspectable per agent.

## Adding more evaluators

`DEFAULT_EVALUATORS` in `handlers/langfuse_handler.py` is a plain dict of
`name -> callable(transcript) -> float`. Add a new function and register it
there to score every call on that dimension going forward — no other code
needs to change.

For evaluations that need an LLM (for example, "did the agent follow the
script?" or sentiment scoring), the recommended path is to run those against
ingested traces from the Langfuse UI/API or a scheduled job, rather than
adding another LLM call to the live call path in `apps/ai`.

## Local setup

1. Start Langfuse locally:

   ```
   docker compose -f docker-compose.langfuse.yml --env-file .env.dev up -d
   ```

   This runs a fully separate, namespaced copy of Langfuse's self-host stack
   (Postgres, ClickHouse, Redis, MinIO, web, worker) so it doesn't collide
   with QuickVoice's own `postgres`/`redis` services from
   `docker-compose.dev.yml`.

2. Open `http://localhost:3300`, sign up, create a project, and copy its
   API keys.

3. Add them to `apps/ai/.env.dev`:

   ```
   LANGFUSE_ENABLED=true
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=http://localhost:3300
   ```

4. Restart the AI service (`task ai:api` / `task ai:worker`, or
   `task up:dev`). Finished calls will start showing up as traces in
   Langfuse within a few seconds of call end.

Set `LANGFUSE_ENABLED=false`, or leave the keys blank, to disable the
integration entirely — the AI service falls back to its existing behavior
with no code changes required.

## Using Langfuse Cloud instead

Skip step 1 above and point `LANGFUSE_BASE_URL` at
`https://cloud.langfuse.com` (or your region's endpoint) using API keys from
a [Langfuse Cloud](https://cloud.langfuse.com) project.

## Production notes

- This integration currently instruments `apps/ai` only (the LiveKit voice
  worker and its RAG lookups). `apps/server`'s MCP tool calls and agent
  config resolution are not yet traced; the same `langfuse_handler.py`
  pattern (lazy client, best-effort try/except, opt-in via env vars) can be
  ported to a Node/TS equivalent using the `langfuse` npm package if that
  visibility is needed later.
- `docker-compose.langfuse.yml` uses the same dev-only placeholder secrets
  as `docker-compose.dev.yml` (marked `# CHANGEME`). Generate real values
  (`openssl rand -hex 32` for `LANGFUSE_ENCRYPTION_KEY`, etc.) before running
  this anywhere beyond a local machine.
