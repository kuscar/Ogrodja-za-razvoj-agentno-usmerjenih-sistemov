# Agentni sistem za podporo strankam pri kratkoročnih najemih nepremičnin

Prototip agentno usmerjenega sistema za podporo strankam pri kratkoročnih najemih nepremičnin v New Yorku, Dubaju in Sydneyju, razvit z ogrodjem Google Agent Development Kit (ADK) in nameščen v infrastrukturi Google Cloud.

## Namestitev v infrastrukturo Google Cloud

```bash
gcloud auth login
gcloud config set project <GCP_PROJECT_ID>
gcloud services enable run.googleapis.com sqladmin.googleapis.com aiplatform.googleapis.com artifactregistry.googleapis.com
```

### Konfiguracija okoljskih spremenljivk v Google Cloud Secret Manager

| Spremenljivka | Opis |
|---|---|
| `SQL_APPUSER_PASSWORD` | Geslo PostgreSQL uporabnika za Cloud SQL |
| `RAG_CORPUS_NAME` | Polno ime Vertex AI RAG korpusa |
| `MAPS_MCP_URL` | URL strežnika Google Maps MCP |
| `JWT_SECRET_KEY` | Skrivni ključ za podpisovanje JWT žetonov |
| `GOOGLE_MAPS_API_KEY` | Google Maps API ključ za MCP orodje za zemljevide |
| `GOOGLE_GENAI_USE_VERTEXAI` | Nastavi na `1` za uporabo Vertex AI namesto Google AI Studio |
| `GCP_REGION` | GCP regija za namestitev storitve |
| `GCP_PROJECT_ID` | Identifikator projekta v Google Cloud |
| `DISCORD_BOT_TOKEN` | Žeton Discord bota za pošiljanje nujnih obvestil |
| `DISCORD_CHANNEL_ID` | Identifikator Discord kanala za obvestila upravniku |

### Namestitev storitve Cloud Run in RAG korpusa

Gradnja Docker slike, objava v Artifact Registry in namestitev na Cloud Run:

```bash
gcloud builds submit --tag <REGION>-docker.pkg.dev/<GCP_PROJECT_ID>/<REPO>/rental-support:latest .
gcloud run deploy rental-support --image <REGION>-docker.pkg.dev/<GCP_PROJECT_ID>/<REPO>/rental-support:latest --region <GCP_REGION> --platform managed --allow-unauthenticated
```

Ustvari RAG korpus v Vertex AI RAG.

### Gradnja in namestitev spletnega vmesnika
```bash
cd frontend
VITE_API_BASE_URL=https://link.do.cloud.run.servica npm run build
```
Vsebino izhodne mape `dist` je potrebno objaviti na storitvi Cloudflare Pages ali enakovredni platformi za gostovanje statičnih spletnih strani.

> [!NOTE]
> Podrobna razlaga arhitekture agentnega sistema in delovanja njegovih komponent je na voljo v pripadajoči diplomski nalogi.

