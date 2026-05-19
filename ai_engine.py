import os, json, uuid
from datetime import datetime
from typing import List
from groq import Groq
from dotenv import load_dotenv
import requests

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Model config (cascadeflow routing logic) ──────────────
CHEAP_MODEL = "qwen-qwen3-32b"           # ~$0.001 / query — classification
EXP_MODEL   = "llama-3.3-70b-versatile" # ~$0.04  / query — generation

# ── Audit trail (cascadeflow logs) ───────────────────────
_audit_log: List[dict] = []

def _log(task: str, model: str, tokens: int,
         cost: float, latency_ms: float, outcome: str):
    _audit_log.append({
        "id":         str(uuid.uuid4())[:8],
        "timestamp":  datetime.now().strftime("%H:%M:%S"),
        "task":       task,
        "model":      model,
        "tokens":     tokens,
        "cost":       cost,
        "latency_ms": round(latency_ms),
        "outcome":    outcome,
    })

def get_audit_log():
    return _audit_log


# ── Hindsight memory ──────────────────────────────────────
_local_memory: List[dict] = []   # fallback if Hindsight unavailable

HINDSIGHT_API_KEY     = os.getenv("HINDSIGHT_API_KEY", "")
HINDSIGHT_PIPELINE_ID = os.getenv("HINDSIGHT_PIPELINE_ID", "")
HINDSIGHT_BASE        = "https://api.hindsight.vectorize.io/v1"

def hindsight_retain(deal: dict, signals: List[str]):
    """Store lost deal memory in Hindsight."""
    content = (
        f"Lost deal: {deal['company']} ({deal['industry']}, "
        f"${deal['deal_size']:,}). Stage: {deal['stage_died']}. "
        f"Signals: {', '.join(signals)}. "
        f"Technical stakeholder: {deal.get('technical_stakeholder', False)}. "
        f"Champion dark: {deal.get('champion_went_dark', False)}. "
        f"Notes: {deal.get('rep_notes', '')}"
    )
    if HINDSIGHT_API_KEY:
        try:
            requests.post(
                f"{HINDSIGHT_BASE}/pipelines/{HINDSIGHT_PIPELINE_ID}/documents",
                headers={"Authorization": f"Bearer {HINDSIGHT_API_KEY}",
                         "Content-Type": "application/json"},
                json={"content": content,
                      "metadata": {"industry": deal["industry"],
                                   "deal_size": deal["deal_size"],
                                   "signals": signals}},
                timeout=5
            )
            return
        except Exception:
            pass
    # Fallback: local memory
    _local_memory.append({"content": content, "deal": deal, "signals": signals})


def hindsight_recall(query: str, industry: str = "") -> List[dict]:
    """Recall similar deals from Hindsight."""
    if HINDSIGHT_API_KEY:
        try:
            r = requests.post(
                f"{HINDSIGHT_BASE}/pipelines/{HINDSIGHT_PIPELINE_ID}/retrieve",
                headers={"Authorization": f"Bearer {HINDSIGHT_API_KEY}",
                         "Content-Type": "application/json"},
                json={"query": query, "top_k": 5},
                timeout=5
            )
            return r.json().get("documents", [])
        except Exception:
            pass
    # Fallback: search local memory by industry
    results = [m for m in _local_memory
               if industry.lower() in m["deal"].get("industry", "").lower()]
    return results[-5:] if results else _local_memory[-5:]


# ── cascadeflow: cheap model — extract failure signals ────
def extract_failure_signals(deal: dict) -> List[str]:
    """
    CASCADEFLOW ROUTE: cheap model (qwen3-32b)
    Task: classify failure signals from deal data.
    Cost: ~$0.001
    """
    t0 = datetime.now()
    prompt = f"""Extract failure signals from this lost sales deal.

Company: {deal['company']} ({deal['industry']}, ${deal['deal_size']:,})
Stage died: {deal['stage_died']}
Technical stakeholder involved: {deal.get('technical_stakeholder', False)}
Champion went dark: {deal.get('champion_went_dark', False)}
Objections: {', '.join(deal.get('objections', []))}
Competitors: {', '.join(deal.get('competitors', []))}
Pricing discussed week: {deal.get('pricing_discussed_week', 'Never')}
Notes: {deal.get('rep_notes', '')}

Return a JSON object with ONE key "signals": a list of 3-5 specific failure signals.
Example signals: "No technical stakeholder", "Champion went dark week 2",
"Competitor Salesforce mentioned", "Legal came in too late", "No pricing before week 3".
ONLY return valid JSON. No other text."""

    resp = groq_client.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=150,
        temperature=0.2,
    )
    latency = (datetime.now() - t0).total_seconds() * 1000
    tokens  = resp.usage.total_tokens
    cost    = tokens * 0.0000009  # qwen3-32b rate

    _log("Signal Extraction", CHEAP_MODEL, tokens, cost, latency,
         f"Extracted signals for {deal['company']}")

    try:
        data = json.loads(resp.choices[0].message.content)
        return data.get("signals", [])
    except:
        return ["Unknown failure pattern"]


# ── cascadeflow: cheap model — risk score ─────────────────
def _risk_score(deal: dict, similar_signals: List[str]) -> dict:
    """
    CASCADEFLOW ROUTE: cheap model (qwen3-32b)
    Task: pattern match live deal vs historical losses.
    Cost: ~$0.001
    """
    t0 = datetime.now()
    prompt = f"""Score this live sales deal's risk of being lost (1-9 scale).

Live deal: {deal['company']} ({deal['industry']}, ${deal['deal_size']:,})
Stakeholders: {', '.join(deal.get('stakeholders', []))}
Week in cycle: {deal.get('week_in_cycle', 1)}
Pricing discussed: {deal.get('pricing_discussed', False)}

Historical failure signals from similar lost deals:
{', '.join(similar_signals) if similar_signals else 'No history yet'}

Return JSON with:
- "risk_score": integer 1-9
- "matched_signals": list of specific risk factors present in THIS deal
- "signal_count": integer

ONLY return valid JSON."""

    resp = groq_client.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=150,
        temperature=0.1,
    )
    latency = (datetime.now() - t0).total_seconds() * 1000
    tokens  = resp.usage.total_tokens
    cost    = tokens * 0.0000009

    _log("Risk Scoring", CHEAP_MODEL, tokens, cost, latency,
         f"Scored {deal['company']}")

    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"risk_score": 5, "matched_signals": [], "signal_count": 0}


# ── cascadeflow: expensive model — warning card ───────────
def _generate_warning(deal: dict, similar_docs: List[dict], risk: dict) -> dict:
    """
    CASCADEFLOW ROUTE: expensive model (llama-3.3-70b)
    Task: generate personalized warning card with email draft.
    Only fires if risk_score > 4 (quality gate).
    Cost: ~$0.038
    """
    t0 = datetime.now()

    similar_text = "\n".join([
        f"• {doc.get('content', '')[:200]}"
        for doc in similar_docs[:3]
    ]) if similar_docs else "No similar deals in memory yet."

    prompt = f"""You are a sales intelligence agent with institutional memory.
Generate a warning card for a rep about to send a proposal.

Deal: {deal['company']} ({deal['industry']}, ${deal['deal_size']:,})
Risk Score: {risk.get('risk_score', 5)}/9
Risk Signals: {', '.join(risk.get('matched_signals', []))}

Similar lost deals recalled from memory:
{similar_text}

Return a JSON object with these exact keys:
- "headline": one sharp warning sentence (max 15 words)
- "key_risk": the single biggest risk factor (max 10 words)
- "historical_context": what happened in similar deals (1-2 sentences, with numbers)
- "recommended_action": one specific action to take this week (1 sentence)
- "urgency": "HIGH" | "MEDIUM" | "LOW"
- "email_subject": subject line for a follow-up email
- "email_body": 3-sentence email body to the prospect

ONLY return valid JSON. Be specific, not generic."""

    resp = groq_client.chat.completions.create(
        model=EXP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=500,
        temperature=0.4,
    )
    latency = (datetime.now() - t0).total_seconds() * 1000
    tokens  = resp.usage.total_tokens
    cost    = tokens * 0.000059  # llama-3.3-70b rate

    _log("Warning Card Generation", EXP_MODEL, tokens, cost, latency,
         f"Warning for {deal['company']} (risk {risk.get('risk_score',5)}/9)")

    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"headline": "High risk deal detected", "key_risk": "Unknown"}


# ── Main orchestration (called by main.py) ────────────────
def analyze_live_deal(deal: dict) -> dict:
    """
    Full cascadeflow pipeline:
    1. Recall from Hindsight
    2. Cheap model → risk score
    3. Expensive model → warning (only if risk > 4)
    Returns complete analysis with cost breakdown.
    """
    # Step 1: Recall similar deals from Hindsight
    query = f"lost deal {deal['industry']} {deal['deal_size']}"
    similar_docs = hindsight_recall(query, deal["industry"])

    # Extract signals from recalled docs for pattern matching
    known_signals = []
    for doc in similar_docs:
        text = doc.get("content", "")
        if "Signals:" in text:
            sig_part = text.split("Signals:")[1].split(".")[0]
            known_signals.extend([s.strip() for s in sig_part.split(",")])

    # Step 2: cascadeflow → cheap model risk scoring
    risk = _risk_score(deal, known_signals)
    risk_score = risk.get("risk_score", 5)

    # Step 3: cascadeflow quality gate → expensive model only if needed
    if risk_score > 4:
        warning = _generate_warning(deal, similar_docs, risk)
    else:
        warning = {
            "headline":           "Deal looks healthy — keep the momentum.",
            "key_risk":           "Low risk",
            "urgency":            "LOW",
            "historical_context": "Similar deals in this range close successfully.",
            "recommended_action": "Proceed with proposal. Schedule a follow-up.",
            "email_subject":      "Next steps for our partnership",
            "email_body":         "Following up on our conversation...",
        }

    # Calculate cost breakdown
    recent_logs = _audit_log[-3:]
    total = sum(l["cost"] for l in recent_logs)

    return {
        "risk_score":        risk_score,
        "matched_signals":   risk.get("matched_signals", []),
        "similar_deals_count": len(similar_docs),
        "warning":           warning,
        "routing": {
            "risk_model":    CHEAP_MODEL,
            "warning_model": EXP_MODEL if risk_score > 4 else "skipped (low risk)",
            "escalated":     risk_score > 4,
        },
        "cost_breakdown": {
            "risk_check":  f"${recent_logs[-2]['cost']:.4f}" if len(recent_logs) >= 2 else "$0.001",
            "warning_gen": f"${recent_logs[-1]['cost']:.4f}" if risk_score > 4 else "$0.000",
            "total":       f"${total:.4f}",
        }
    }
