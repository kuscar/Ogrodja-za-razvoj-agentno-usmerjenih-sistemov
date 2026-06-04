# Sistem za generiranje CV-ja in motivacijskega pisma

To je agentno usmerjen sistem za generiranje CV-ja in motivacijskega pisma implementiran v ogrodju LangChain / LangGraph.

## Namestitev

### 1. Ustvarite Supabase projekt
Ustvarite projekt na [app.supabase.com](https://app.supabase.com) in si shranite URL ter service-role ključ.

### 2. Nastavite bazo podatkov
```bash
cd supabase
npx supabase login
npx supabase link --project-ref <vaš-project-ref>
npx supabase db push
```

### 3. Konfigurirajte okoljske spremenljivke
Kopirajte `.env.example` v `.env` in izpolnite vrednosti:
```bash
cp .env.example .env
```

### 4. Zaženite sistem
```bash
docker compose up --build -d
```

API bo dostopen na `http://localhost:8000`.

## Nastavitve okoljskih spremenljivk

Vse nastavitve se nahajajo v `.env` datoteki (ustvarjeni iz `.env.example`):

| Spremenljivka | Opis |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API ključ za LLM dostop |
| `GEMINI_MODEL` | Model (privzeto: `gemini-2.5-flash-lite`) |
| `EMBED_MODEL` | Model za vgrajevanje (privzeto: `gemini-embedding-001`) |
| `SUPABASE_URL` | URL vašega Supabase projekta |
| `SUPABASE_ANON_KEY` | Javni (anon) ključ Supabase |
| `SUPABASE_SERVICE_ROLE_KEY` | Service-role ključ Supabase |
| `SUPABASE_JWT_SECRET` | JWT skrivnost za preverjanje žetonov |
| `SUPABASE_ACCESS_TOKEN` | Supabase CLI dostopni žeton |
| `HF_TOKEN` | Hugging Face žeton za prompt-guard model |
| `LANGCHAIN_API_KEY` | LangSmith API ključ za sledenje |
| `LANGSMITH_ENDPOINT` | LangSmith endpoint URL |

Ostale vrednosti (`SUPABASE_MCP_URL`, `PROMPT_GUARD_URL`, `RATE_LIMIT_STORAGE_URI` itd.) so že nastavljene za lokalno delovanje z Dockerjem in jih ni potrebno spreminjati.

> [!NOTE]
> Podrobna razlaga delovanja agentnega sistema in njegovih komponent je na voljo v pripadajoči diplomski nalogi.
