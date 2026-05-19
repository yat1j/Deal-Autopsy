# Deal Autopsy Agent 🔍

An AI agent that learns from your lost sales deals and warns reps before they repeat the same mistake.

Built with **Hindsight** (vector memory) + **cascadeflow** (smart model routing) + **Groq** (fast LLMs).

> Full analysis of a $48k at-risk deal: **$0.039**. Same work without cascadeflow routing: **$4.80**.

---

## How It Works

```
Lost Deal (JSON)
      ↓
extract_failure_signals()   ← qwen3-32b ($0.001) — cheap classification
      ↓
hindsight_retain()          ← stores deal + signals in Hindsight vector DB
      ↓
[20 deals loaded]

Live Deal (entered by rep)
      ↓
risk_score = cheap model    ← qwen3-32b ($0.001)
      ↓
hindsight_recall()          ← finds 5 most similar lost deals
      ↓
warning_card = exp model    ← llama-3.3-70b ($0.038) — only if risk > 5
      ↓
Rep sees: "In 11/14 similar deals, no tech stakeholder = lost deal. Here's a draft email."
```

---

## Stack

| Layer | Tech |
|---|---|
| Backend API | FastAPI + Uvicorn |
| AI Routing | cascadeflow |
| LLMs | Groq (qwen3-32b + llama-3.3-70b) |
| Deal Memory | Hindsight by vectorize.io |
| Frontend | Vanilla HTML / CSS / JS |

---

## Project Structure

```
deal-autopsy/
├── backend/
│   ├── main.py           # FastAPI server — all 4 endpoints
│   ├── ai_engine.py      # AI brain: cascadeflow routing + Hindsight
│   ├── seed.py           # Loads 20 synthetic lost deals into Hindsight
│   ├── requirements.txt
│   └── .env              # ← NOT committed (see .gitignore)
└── frontend/
    └── index.html        # 3-tab UI: Feed | Live Alert | Audit Trail
```

---

## Setup

### 1. Get API Keys

| Service | Where | Free? |
|---|---|---|
| Groq | [groq.com](https://groq.com) | ✅ Free tier |
| Hindsight | [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io) | ✅ Use code `MEMHACK515` for $50 credit |

### 2. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/deal-autopsy.git
cd deal-autopsy/backend
pip install -r requirements.txt
```

### 3. Create `backend/.env`

```env
GROQ_API_KEY=gsk_your_groq_key_here
HINDSIGHT_API_KEY=your_hindsight_key_here
HINDSIGHT_PIPELINE_ID=your_pipeline_id_here
```

### 4. Start the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
# Interactive docs → http://localhost:8000/docs
```

### 5. Start the Frontend

```bash
cd frontend
python3 -m http.server 3000
# Open → http://localhost:3000
```

### 6. Seed 20 Lost Deals

```bash
cd backend
python seed.py
# Loads 20 synthetic deals into Hindsight (~2-3 min due to rate limit spacing)
```

---

## API Reference

| Method | Endpoint | Used By | Description |
|---|---|---|---|
| `GET` | `/` | Anyone | Health check |
| `GET` | `/api/deals` | Frontend Feed tab | Returns all ingested lost deals |
| `POST` | `/api/deals/ingest` | seed.py | Ingest a lost deal → extract signals → store in Hindsight |
| `POST` | `/api/deals/analyze` | Frontend Alert tab | Analyze a live deal → risk score + warning card |
| `GET` | `/api/audit` | Frontend Audit tab | cascadeflow model call log + total cost |

---

## Demo Walkthrough

1. **Feed tab** — 20 ingested deals, each with AI-extracted failure signal tags (e.g. "no_technical_stakeholder", "champion_went_dark")
2. **Live Alert tab** — Enter a real deal, hit Analyze. Risk score + personalized warning appear in ~3 seconds
3. **Audit Trail tab** — Every model call logged: model used, tokens, latency, cost. Total for 20 deals: ~$1.24

---

## License

MIT
