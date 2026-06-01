# Sistem za detekcijo lažnih e-trgovin

To je agentno usmerjen sistem za detekcijo lažnih e-trgovin implementiran v ogrodju n8n.

## Vsebina repozitorija
V repozitoriju se nahajajo 3 ključne datoteke:
- **CSV datoteka** (`Scam Websites.csv`): kratek evaluacijski dataset.
- **Dve JSON datoteki** (`Scam Detector.json` in `Contact-Form-Check.json`): n8n workflowa.

## Kako uvoziti workflow v n8n
1. Odprite n8n in ustvarite nov (prazen) workflow.
2. V meniju zgoraj desno kliknite na nastavitve in izberite **Import from File** ter izberite ustrezno JSON datoteko. Lahko pa tudi povsem preprosto kopirate celotno vsebino JSON datoteke in jo prilepite.
3. Shranite in ponovite za drug workflow.

## Kako uvoziti evaluacijski dataset
1. V osrednjem meniju aplikacije kliknite na **Data tables** (namesto na Workflows).
2. Izberite možnost **Create data table**.
3. Izberite uvoz iz datoteke (**Import from CSV**) in naložite priloženo datoteko `Scam Websites.csv`.

## Nastavitve API ključev
Vse spodnje nastavitve se lahko enostavno uredi direktno preko n8n uporabniškega vmesnika, ko uvozite workflow:
- **Contact-Form-Check**: V tem workflowu je potrebno dodati svoj **ScrapeOps API key** v ustreznih HTTP poizvedbah.
- **Scam Detector**: Opremiti ga je potrebno z **Google AI Studio API** ključem (za LLM dostop) ter povezati vaš **Redis račun**.

> [!NOTE]
> Podrobna razlaga delovanja agentnega sistema in njegovih komponent je na voljo v pripadajoči diplomski nalogi.
