"""
AI Compare - Groq vs Gemini  (MCA Dissertation Project)

Improvements over the first version:
  1. Dual cross-judging: BOTH Groq and Gemini judge, scores are averaged,
     so self-preference bias of a single judge cancels out.
  2. Blind + randomized order: judges see "Answer A / Answer B" in random order.
  3. Fair setup: same system prompt and same temperature for both models.
  4. Speed is reported separately (default weight 0 in final score).
  5. Failed / one-sided comparisons are flagged valid=0 and excluded from stats.
  6. Human evaluation page (/human-eval) + /stats + /export.csv for analysis.
  7. Model names and all judge scores stored in DB for reproducibility.
  8. init_db runs on import, so it works with gunicorn on Render.
"""

from flask import (
    Flask, render_template, request, jsonify,
    Response, render_template_string
)
from groq import Groq
from google import genai
from google.genai import types
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import os
import io
import csv
import json
import time
import random
import sqlite3
import statistics

load_dotenv()

app = Flask(__name__)

# =========================
# CONFIGURATION
# =========================

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# Same settings for both answering models (fair comparison)
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Give accurate, clear and relevant answers."
)
ANSWER_TEMPERATURE = 0.3

# Scoring weights
QUALITY_WEIGHTS = {"accuracy": 0.45, "relevance": 0.35, "clarity": 0.20}
METRICS = list(QUALITY_WEIGHTS.keys())

# Speed weight in the final score. 0.0 = speed is reported but not scored.
SPEED_WEIGHT = float(os.environ.get("SPEED_WEIGHT", 0.0))

# If final scores differ by less than this, result is a Tie.
TIE_MARGIN = float(os.environ.get("TIE_MARGIN", 2.0))

DB_PATH = os.environ.get("DB_PATH", "database.db")

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


# =========================
# DATABASE
# =========================

# Columns added on top of the original schema (auto-migrated).
EXTRA_COLUMNS = {
    "groq_model": "TEXT",
    "gemini_model": "TEXT",
    "valid": "INTEGER DEFAULT 1",
    "groq_quality": "REAL",
    "gemini_quality": "REAL",
    "groq_speed_score": "REAL",
    "gemini_speed_score": "REAL",
    "judge_details": "TEXT",
    "human_winner": "TEXT",
}


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            groq_answer TEXT,
            gemini_answer TEXT,
            groq_speed REAL,
            gemini_speed REAL,
            groq_accuracy REAL,
            groq_relevance REAL,
            groq_clarity REAL,
            groq_score REAL,
            gemini_accuracy REAL,
            gemini_relevance REAL,
            gemini_clarity REAL,
            gemini_score REAL,
            winner TEXT,
            reason TEXT,
            created_at TEXT
        )
    """)

    # Migrate old databases: add any missing columns
    cur.execute("PRAGMA table_info(comparisons)")
    existing = {row["name"] for row in cur.fetchall()}
    for col, col_type in EXTRA_COLUMNS.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE comparisons ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()


def save_comparison(question, groq_answer, gemini_answer,
                    groq_speed, gemini_speed, result, valid):
    """Save one comparison. Returns the new row id (or None on error)."""
    try:
        g = result.get("groq", {})
        m = result.get("gemini", {})

        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO comparisons (
                question, groq_answer, gemini_answer,
                groq_speed, gemini_speed,
                groq_accuracy, groq_relevance, groq_clarity, groq_score,
                gemini_accuracy, gemini_relevance, gemini_clarity, gemini_score,
                winner, reason, created_at,
                groq_model, gemini_model, valid,
                groq_quality, gemini_quality,
                groq_speed_score, gemini_speed_score,
                judge_details
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            question, groq_answer, gemini_answer,
            groq_speed, gemini_speed,
            g.get("accuracy", 0), g.get("relevance", 0),
            g.get("clarity", 0), g.get("final_score", 0),
            m.get("accuracy", 0), m.get("relevance", 0),
            m.get("clarity", 0), m.get("final_score", 0),
            result.get("winner"), result.get("reason"),
            time.strftime("%Y-%m-%d %H:%M:%S"),
            GROQ_MODEL, GEMINI_MODEL, 1 if valid else 0,
            g.get("quality", 0), m.get("quality", 0),
            g.get("speed_score", 0), m.get("speed_score", 0),
            json.dumps(result.get("judge_details", {}))
        ))
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return row_id

    except Exception as e:
        print("Database Save Error:", e)
        return None


# Create tables at import time (needed for gunicorn on Render)
init_db()


# =========================
# ANSWERING MODELS
# =========================

def call_groq(text):
    start = time.perf_counter()
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=ANSWER_TEMPERATURE,
        )
        speed = round(time.perf_counter() - start, 2)
        return {
            "success": True,
            "answer": response.choices[0].message.content,
            "speed": speed,
        }, 200

    except Exception as e:
        speed = round(time.perf_counter() - start, 2)
        print("Groq Error:", e)
        return {
            "success": False,
            "answer": "Groq API Error: " + str(e),
            "speed": speed,
        }, 500


def call_gemini(text):
    """Call Gemini. Retries up to 2 times on temporary (503) errors.
    Only the time of the successful attempt is counted as speed."""
    error_text = ""
    speed = 0
    for attempt in range(3):
        start = time.perf_counter()
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=ANSWER_TEMPERATURE,
                ),
            )
            speed = round(time.perf_counter() - start, 2)
            return {"success": True,
                    "answer": response.text,
                    "speed": speed}, 200

        except Exception as e:
            speed = round(time.perf_counter() - start, 2)
            error_text = str(e)
            print("Gemini Error:", error_text)

            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                return {
                    "success": False,
                    "answer": ("Gemini is temporarily unavailable because "
                               "its API quota has been reached. "
                               "Please try again later."),
                    "speed": speed,
                    "error_type": "quota",
                    "quota": True,
                }, 429

            temporary = ("503" in error_text or "UNAVAILABLE" in error_text
                         or "overloaded" in error_text.lower())
            if temporary and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            break

    return {"success": False,
            "answer": "Gemini API Error: " + error_text,
            "speed": speed}, 500


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/groq", methods=["POST"])
def groq_route():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"success": False,
                        "answer": "Please enter a question.",
                        "speed": 0}), 400
    body, status = call_groq(text)
    return jsonify(body), status


@app.route("/gemini", methods=["POST"])
def gemini_route():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"success": False,
                        "answer": "Please enter a question.",
                        "speed": 0}), 400
    body, status = call_gemini(text)
    return jsonify(body), status


# =========================
# JUDGING
# =========================

JUDGE_SYSTEM = (
    "You are a strict, objective evaluator of AI answers. "
    "Return only valid JSON."
)


def build_judge_prompt(question, answer_a, answer_b):
    return f"""
Evaluate two anonymous answers to the same question.

QUESTION:
{question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Score each answer from 0 to 100 on:
- accuracy: factual correctness and absence of errors
- relevance: how directly it answers what was asked
- clarity: how clear, well organised and easy to understand it is

Rules:
- Do NOT prefer an answer because it is longer or has more formatting.
- Factual correctness matters most.
- Use the full 0-100 range and different scores where the answers differ.
- The reason must NOT guess which AI wrote which answer.

Return ONLY this JSON and nothing else:

{{
  "answer_a": {{"accuracy": 0, "relevance": 0, "clarity": 0}},
  "answer_b": {{"accuracy": 0, "relevance": 0, "clarity": 0}},
  "reason": "One or two sentences"
}}
"""


def extract_json(raw):
    """Pull the first JSON object out of a model reply."""
    raw = (raw or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in judge reply")
    return json.loads(raw[start:end + 1])


def ask_groq_judge(prompt):
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def ask_gemini_judge(prompt):
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM,
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    return response.text


def clamp(value):
    return max(0.0, min(100.0, float(value)))


def run_judge(judge_name, ask_fn, question, groq_answer, gemini_answer):
    """
    One judge evaluates both answers blind, in random order.
    Returns a dict, or None if the judge failed.
    """
    swap = random.random() < 0.5  # True => Gemini shown as Answer A

    if swap:
        answer_a, answer_b = gemini_answer, groq_answer
    else:
        answer_a, answer_b = groq_answer, gemini_answer

    prompt = build_judge_prompt(question, answer_a, answer_b)

    last_error = None
    for _ in range(2):  # one retry
        try:
            data = extract_json(ask_fn(prompt))
            a = {k: clamp(data["answer_a"][k]) for k in METRICS}
            b = {k: clamp(data["answer_b"][k]) for k in METRICS}

            groq_scores, gemini_scores = (b, a) if swap else (a, b)

            return {
                "judge": judge_name,
                "order": "Gemini first" if swap else "Groq first",
                "groq": groq_scores,
                "gemini": gemini_scores,
                "reason": str(data.get("reason", "")).strip(),
            }
        except Exception as e:
            last_error = e
            print(f"{judge_name} judge error:", e)

    print(f"{judge_name} judge failed:", last_error)
    return None


def quality_score(scores):
    return sum(scores[k] * QUALITY_WEIGHTS[k] for k in METRICS)


def ai_judge(question, groq_answer, gemini_answer,
             groq_speed=0, gemini_speed=0):

    # Both models judge, in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_judge, "groq", ask_groq_judge,
                        question, groq_answer, gemini_answer),
            pool.submit(run_judge, "gemini", ask_gemini_judge,
                        question, groq_answer, gemini_answer),
        ]
        judgements = [f.result() for f in futures]

    judgements = [j for j in judgements if j]

    if not judgements:
        return {
            "success": False,
            "winner": "Unable to determine",
            "reason": "AI comparison could not be completed.",
            "error": "All judges failed",
        }

    # Average each metric across the judges that worked
    def avg(model, metric):
        return sum(j[model][metric] for j in judgements) / len(judgements)

    groq_avg = {k: avg("groq", k) for k in METRICS}
    gemini_avg = {k: avg("gemini", k) for k in METRICS}

    groq_quality = quality_score(groq_avg)
    gemini_quality = quality_score(gemini_avg)

    # Speed score: fastest = 100
    if groq_speed > 0 and gemini_speed > 0:
        fastest = min(groq_speed, gemini_speed)
        groq_speed_score = fastest / groq_speed * 100
        gemini_speed_score = fastest / gemini_speed * 100
    else:
        groq_speed_score = 0
        gemini_speed_score = 0

    groq_final = (groq_quality * (1 - SPEED_WEIGHT)
                  + groq_speed_score * SPEED_WEIGHT)
    gemini_final = (gemini_quality * (1 - SPEED_WEIGHT)
                    + gemini_speed_score * SPEED_WEIGHT)

    if abs(groq_final - gemini_final) < TIE_MARGIN:
        winner = "Tie"
    elif groq_final > gemini_final:
        winner = "Groq"
    else:
        winner = "Gemini"

    reason = " | ".join(
        f"{j['judge'].title()} judge: {j['reason']}"
        for j in judgements if j["reason"]
    ) or "Comparison completed."

    if len(judgements) == 1:
        reason += " (Only one judge was available.)"

    def pack(avg_scores, quality, speed_score, final):
        return {
            "accuracy": round(avg_scores["accuracy"], 2),
            "relevance": round(avg_scores["relevance"], 2),
            "clarity": round(avg_scores["clarity"], 2),
            "quality": round(quality, 2),
            "speed_score": round(speed_score, 2),
            "final_score": round(final, 2),
        }

    return {
        "success": True,
        "valid": True,
        "winner": winner,
        "reason": reason,
        "judges_used": [j["judge"] for j in judgements],
        "groq": pack(groq_avg, groq_quality, groq_speed_score, groq_final),
        "gemini": pack(gemini_avg, gemini_quality,
                       gemini_speed_score, gemini_final),
        "judge_details": {j["judge"]: j for j in judgements},
    }


def run_judge_request(data):
    """Shared logic for /judge and /compare. Returns (result, status)."""
    question = data.get("question", "").strip()
    groq_answer = data.get("groq", "")
    gemini_answer = data.get("gemini", "")
    groq_speed = float(data.get("groqSpeed", 0) or 0)
    gemini_speed = float(data.get("geminiSpeed", 0) or 0)
    groq_ok = bool(data.get("groqSuccess", False))
    gemini_ok = bool(data.get("geminiSuccess", False))

    zero = {"accuracy": 0, "relevance": 0, "clarity": 0,
            "quality": 0, "speed_score": 0, "final_score": 0}

    # Only one model answered: no real comparison -> saved as invalid
    if groq_ok != gemini_ok:
        winner = "Groq" if groq_ok else "Gemini"
        other = "Gemini" if groq_ok else "Groq"
        result = {
            "success": True,
            "valid": False,
            "winner": winner,
            "reason": (f"{other} was unavailable, so a complete "
                       "two-model comparison could not be performed."),
            "groq": dict(zero),
            "gemini": dict(zero),
        }
        row_id = save_comparison(question, groq_answer, gemini_answer,
                                 groq_speed, gemini_speed, result, False)
        result["id"] = row_id
        return result, 200

    if groq_ok and gemini_ok:
        result = ai_judge(question, groq_answer, gemini_answer,
                          groq_speed, gemini_speed)
        if result.get("success"):
            row_id = save_comparison(question, groq_answer, gemini_answer,
                                     groq_speed, gemini_speed, result, True)
            result["id"] = row_id
            result.pop("judge_details", None)  # keep response small
            return result, 200
        return result, 200

    return {
        "success": False,
        "winner": "None",
        "reason": "Both AI services are currently unavailable.",
    }, 503


@app.route("/judge", methods=["POST"])
def judge_route():
    result, status = run_judge_request(request.get_json() or {})
    return jsonify(result), status


@app.route("/compare", methods=["POST"])
def compare():
    """Old compatibility route: answers + judging in one call."""
    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"success": False,
                        "error": "Please enter a question."}), 400

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_groq = pool.submit(call_groq, text)
        f_gemini = pool.submit(call_gemini, text)
        groq_json, _ = f_groq.result()
        gemini_json, _ = f_gemini.result()

    judge_json, _ = run_judge_request({
        "question": text,
        "groq": groq_json.get("answer", ""),
        "gemini": gemini_json.get("answer", ""),
        "groqSpeed": groq_json.get("speed", 0),
        "geminiSpeed": gemini_json.get("speed", 0),
        "groqSuccess": groq_json.get("success", False),
        "geminiSuccess": gemini_json.get("success", False),
    })

    return jsonify({
        "success": True,
        "groq": groq_json,
        "gemini": gemini_json,
        "judge": judge_json,
    })


# =========================
# HISTORY
# =========================

@app.route("/history", methods=["GET"])
def history():
    try:
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM comparisons ORDER BY id DESC").fetchall()
        conn.close()
        return jsonify({"success": True,
                        "history": [dict(r) for r in rows]})
    except Exception as e:
        print("History Error:", e)
        return jsonify({"success": False, "history": [],
                        "error": str(e)}), 500


@app.route("/history/clear", methods=["DELETE"])
def clear_history():
    try:
        conn = get_conn()
        conn.execute("DELETE FROM comparisons")
        conn.commit()
        conn.close()
        return jsonify({"success": True,
                        "message": "Comparison history cleared successfully."})
    except Exception as e:
        print("Clear History Error:", e)
        return jsonify({"success": False,
                        "message": "Could not clear history.",
                        "error": str(e)}), 500


# =========================
# HUMAN EVALUATION (ground truth)
# =========================

HUMAN_EVAL_PAGE = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human Evaluation</title>
<style>
  body{font-family:system-ui,sans-serif;margin:0;padding:12px;background:#f4f4f5}
  .card{background:#fff;border-radius:10px;padding:12px;margin-bottom:16px}
  pre{white-space:pre-wrap;word-wrap:break-word;font-family:inherit;
      background:#f9f9fa;padding:8px;border-radius:6px;max-height:260px;overflow:auto}
  button{padding:10px 14px;margin:4px 4px 0 0;border:0;border-radius:8px;
         background:#4f46e5;color:#fff;font-size:15px}
  button.tie{background:#6b7280}
  .done{opacity:.4}
</style>
</head>
<body>
<h3>Human Evaluation (blind): {{ rows|length }} left</h3>
<p>Read both answers and pick the better one. Model names are hidden.</p>
{% for r in rows %}
  {% set swap = (r.id % 2 == 1) %}
  <div class="card" id="c{{ r.id }}">
    <b>Q{{ r.id }}: {{ r.question }}</b>
    <p><b>Answer A</b></p>
    <pre>{{ r.gemini_answer if swap else r.groq_answer }}</pre>
    <p><b>Answer B</b></p>
    <pre>{{ r.groq_answer if swap else r.gemini_answer }}</pre>
    <button onclick="rate({{ r.id }}, '{{ 'Gemini' if swap else 'Groq' }}')">A is better</button>
    <button onclick="rate({{ r.id }}, '{{ 'Groq' if swap else 'Gemini' }}')">B is better</button>
    <button class="tie" onclick="rate({{ r.id }}, 'Tie')">Tie</button>
  </div>
{% endfor %}
<script>
async function rate(id, winner){
  const res = await fetch('/rate', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id:id, winner:winner})});
  if(res.ok){ document.getElementById('c'+id).classList.add('done'); }
  else { alert('Could not save'); }
}
</script>
</body>
</html>
"""


@app.route("/human-eval", methods=["GET"])
def human_eval():
    conn = get_conn()
    rows = conn.execute("""
        SELECT id, question, groq_answer, gemini_answer
        FROM comparisons
        WHERE valid = 1 AND human_winner IS NULL
        ORDER BY id
        LIMIT 50
    """).fetchall()
    conn.close()
    return render_template_string(HUMAN_EVAL_PAGE, rows=rows)


@app.route("/rate", methods=["POST"])
def rate():
    data = request.get_json() or {}
    row_id = data.get("id")
    winner = data.get("winner")

    if winner not in ("Groq", "Gemini", "Tie") or not row_id:
        return jsonify({"success": False, "error": "Invalid input"}), 400

    conn = get_conn()
    conn.execute("UPDATE comparisons SET human_winner = ? WHERE id = ?",
                 (winner, row_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


# =========================
# STATS + EXPORT (for the paper)
# =========================

@app.route("/stats", methods=["GET"])
def stats():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM comparisons WHERE valid = 1").fetchall()
    conn.close()

    n = len(rows)
    if n == 0:
        return jsonify({"success": True, "valid_comparisons": 0})

    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 2) if values else None

    def median(values):
        values = [v for v in values if v is not None]
        return round(statistics.median(values), 2) if values else None

    wins = {"Groq": 0, "Gemini": 0, "Tie": 0}
    for r in rows:
        if r["winner"] in wins:
            wins[r["winner"]] += 1

    # Judge bias check: does each judge favour its own model?
    # gap = (Groq answer quality - Gemini answer quality) as seen by that judge
    gaps = {"groq": [], "gemini": []}
    for r in rows:
        try:
            details = json.loads(r["judge_details"] or "{}")
        except Exception:
            continue
        for judge_name in ("groq", "gemini"):
            d = details.get(judge_name)
            if d:
                gaps[judge_name].append(
                    quality_score(d["groq"]) - quality_score(d["gemini"]))

    # Agreement with human raters
    rated = [r for r in rows if r["human_winner"]]
    agree = sum(1 for r in rated if r["human_winner"] == r["winner"])

    human_wins = {"Groq": 0, "Gemini": 0, "Tie": 0}
    for r in rated:
        human_wins[r["human_winner"]] += 1

    return jsonify({
        "success": True,
        "valid_comparisons": n,
        "ai_judge_wins": wins,
        "avg_quality": {
            "groq": mean([r["groq_quality"] for r in rows]),
            "gemini": mean([r["gemini_quality"] for r in rows]),
        },
        "avg_final_score": {
            "groq": mean([r["groq_score"] for r in rows]),
            "gemini": mean([r["gemini_score"] for r in rows]),
        },
        "response_time_sec": {
            "groq_mean": mean([r["groq_speed"] for r in rows]),
            "gemini_mean": mean([r["gemini_speed"] for r in rows]),
            "groq_median": median([r["groq_speed"] for r in rows]),
            "gemini_median": median([r["gemini_speed"] for r in rows]),
        },
        "judge_bias_check_mean_gap_groq_minus_gemini": {
            "judged_by_groq": mean(gaps["groq"]),
            "judged_by_gemini": mean(gaps["gemini"]),
            "note": ("If the Groq judge gap is clearly higher than the "
                     "Gemini judge gap, there is self-preference bias."),
        },
        "human_evaluation": {
            "rated": len(rated),
            "human_wins": human_wins,
            "agreement_with_ai_judge_pct":
                round(agree / len(rated) * 100, 1) if rated else None,
        },
        "settings": {
            "groq_model": GROQ_MODEL,
            "gemini_model": GEMINI_MODEL,
            "speed_weight": SPEED_WEIGHT,
            "tie_margin": TIE_MARGIN,
            "temperature": ANSWER_TEMPERATURE,
        },
    })


@app.route("/export.csv", methods=["GET"])
def export_csv():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM comparisons ORDER BY id").fetchall()
    conn.close()

    output = io.StringIO()
    if rows:
        writer = csv.writer(output)
        writer.writerow(rows[0].keys())
        for r in rows:
            writer.writerow(list(r))

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=comparisons.csv"},
    )


# =========================
# START
# =========================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
