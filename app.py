
from flask import Flask, render_template, request, jsonify
from groq import Groq
from google import genai
from dotenv import load_dotenv
import os
import time
import sqlite3
import json

load_dotenv()

app = Flask(__name__)

# =========================
# API CLIENTS
# =========================

groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

gemini_client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)


# =========================
# DATABASE
# =========================

def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
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

    conn.commit()
    conn.close()


def save_comparison(
    question,
    groq_answer,
    gemini_answer,
    groq_speed,
    gemini_speed,
    groq_data,
    gemini_data,
    winner,
    reason
):
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO comparisons (
                question,
                groq_answer,
                gemini_answer,
                groq_speed,
                gemini_speed,
                groq_accuracy,
                groq_relevance,
                groq_clarity,
                groq_score,
                gemini_accuracy,
                gemini_relevance,
                gemini_clarity,
                gemini_score,
                winner,
                reason,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            question,
            groq_answer,
            gemini_answer,
            groq_speed,
            gemini_speed,

            groq_data.get("accuracy", 0),
            groq_data.get("relevance", 0),
            groq_data.get("clarity", 0),
            groq_data.get("final_score", 0),

            gemini_data.get("accuracy", 0),
            gemini_data.get("relevance", 0),
            gemini_data.get("clarity", 0),
            gemini_data.get("final_score", 0),

            winner,
            reason,
            time.strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        return True

    except Exception as e:
        print("Database Save Error:", e)
        return False


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# GROQ API
# =========================

@app.route("/groq", methods=["POST"])
def groq():

    data = request.get_json() or {}

    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "success": False,
            "answer": "Please enter a question.",
            "speed": 0
        }), 400

    start_time = time.time()

    try:

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. "
                        "Give accurate, clear and relevant answers."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content

        speed = round(time.time() - start_time, 2)

        return jsonify({
            "success": True,
            "answer": answer,
            "speed": speed
        })

    except Exception as e:

        speed = round(time.time() - start_time, 2)

        print("Groq Error:", e)

        return jsonify({
            "success": False,
            "answer": "Groq API Error: " + str(e),
            "speed": speed
        }), 500


# =========================
# GEMINI API
# =========================

@app.route("/gemini", methods=["POST"])
def gemini():

    data = request.get_json() or {}

    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "success": False,
            "answer": "Please enter a question.",
            "speed": 0
        }), 400

    start_time = time.time()

    try:

        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=text
        )

        answer = response.text

        speed = round(time.time() - start_time, 2)

        return jsonify({
            "success": True,
            "answer": answer,
            "speed": speed
        })

    except Exception as e:

        speed = round(time.time() - start_time, 2)

        error_text = str(e)

        print("Gemini Error:", error_text)

        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:

            return jsonify({
                "success": False,
                "answer": (
                    "Gemini is temporarily unavailable because "
                    "its API quota has been reached. Please try again later."
                ),
                "speed": speed,
                "error_type": "quota",
                "quota": True
            }), 429

        return jsonify({
            "success": False,
            "answer": "Gemini API Error: " + error_text,
            "speed": speed
        }), 500


# =========================
# AI JUDGE
# =========================

def ai_judge(
    question,
    groq_answer,
    gemini_answer,
    groq_speed=0,
    gemini_speed=0
):

    judge_prompt = f"""
You are an expert AI response evaluator.

Compare the two AI answers for the given question.

QUESTION:
{question}

GROQ ANSWER:
{groq_answer}

GEMINI ANSWER:
{gemini_answer}

Evaluate both answers using:

Accuracy = 45%
Relevance = 35%
Clarity = 20%

Give each AI a score from 0 to 100.

Also determine the winner.

Return ONLY valid JSON in this exact format:

{{
    "groq_accuracy": 0,
    "groq_relevance": 0,
    "groq_clarity": 0,
    "gemini_accuracy": 0,
    "gemini_relevance": 0,
    "gemini_clarity": 0,
    "winner": "Groq",
    "reason": "Short explanation"
}}
"""

    try:

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an objective AI response evaluator. "
                        "Return only valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": judge_prompt
                }
            ],
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        if result.startswith("```"):
            result = result.replace("```json", "")
            result = result.replace("```", "")
            result = result.strip()

        scores = json.loads(result)

        groq_accuracy = float(scores["groq_accuracy"])
        groq_relevance = float(scores["groq_relevance"])
        groq_clarity = float(scores["groq_clarity"])

        gemini_accuracy = float(scores["gemini_accuracy"])
        gemini_relevance = float(scores["gemini_relevance"])
        gemini_clarity = float(scores["gemini_clarity"])

        # =========================
        # QUALITY SCORE
        # =========================

        groq_quality = (
            groq_accuracy * 0.45
            + groq_relevance * 0.35
            + groq_clarity * 0.20
        )

        gemini_quality = (
            gemini_accuracy * 0.45
            + gemini_relevance * 0.35
            + gemini_clarity * 0.20
        )

        # =========================
        # SPEED SCORE
        # =========================

        if groq_speed > 0 and gemini_speed > 0:

            fastest = min(groq_speed, gemini_speed)

            groq_speed_score = (
                fastest / groq_speed
            ) * 100

            gemini_speed_score = (
                fastest / gemini_speed
            ) * 100

        else:

            groq_speed_score = 0
            gemini_speed_score = 0

        # =========================
        # FINAL SCORE
        # =========================

        groq_final = (
            groq_quality * 0.90
            + groq_speed_score * 0.10
        )

        gemini_final = (
            gemini_quality * 0.90
            + gemini_speed_score * 0.10
        )

        if groq_final > gemini_final:
            winner = "Groq"

        elif gemini_final > groq_final:
            winner = "Gemini"

        else:
            winner = "Tie"

        return {
            "success": True,
            "winner": winner,
            "reason": scores.get(
                "reason",
                "Comparison completed."
            ),

            "groq": {
                "accuracy": groq_accuracy,
                "relevance": groq_relevance,
                "clarity": groq_clarity,
                "quality": round(groq_quality, 2),
                "speed_score": round(groq_speed_score, 2),
                "final_score": round(groq_final, 2)
            },

            "gemini": {
                "accuracy": gemini_accuracy,
                "relevance": gemini_relevance,
                "clarity": gemini_clarity,
                "quality": round(gemini_quality, 2),
                "speed_score": round(gemini_speed_score, 2),
                "final_score": round(gemini_final, 2)
            }
        }

    except Exception as e:

        print("Judge Error:", e)

        return {
            "success": False,
            "winner": "Unable to determine",
            "reason": "AI comparison could not be completed.",
            "error": str(e)
        }


# =========================
# JUDGE ENDPOINT
# =========================

@app.route("/judge", methods=["POST"])
def judge():

    data = request.get_json() or {}

    question = data.get("question", "").strip()

    groq_answer = data.get("groq", "")
    gemini_answer = data.get("gemini", "")

    groq_speed = float(data.get("groqSpeed", 0) or 0)
    gemini_speed = float(data.get("geminiSpeed", 0) or 0)

    groq_success = data.get("groqSuccess", False)
    gemini_success = data.get("geminiSuccess", False)

    # =========================
    # ONLY GROQ AVAILABLE
    # =========================

    if groq_success and not gemini_success:

        result = {
            "success": True,
            "winner": "Groq",
            "reason": (
                "Gemini was unavailable, so a complete "
                "two-model comparison could not be performed."
            ),

            "groq": {
                "accuracy": 0,
                "relevance": 0,
                "clarity": 0,
                "quality": 0,
                "speed_score": 100,
                "final_score": 0
            },

            "gemini": {
                "accuracy": 0,
                "relevance": 0,
                "clarity": 0,
                "quality": 0,
                "speed_score": 0,
                "final_score": 0
            }
        }

        save_comparison(
            question,
            groq_answer,
            gemini_answer,
            groq_speed,
            gemini_speed,
            result["groq"],
            result["gemini"],
            result["winner"],
            result["reason"]
        )

        return jsonify(result)

    # =========================
    # ONLY GEMINI AVAILABLE
    # =========================

    if gemini_success and not groq_success:

        result = {
            "success": True,
            "winner": "Gemini",
            "reason": (
                "Groq was unavailable, so a complete "
                "two-model comparison could not be performed."
            ),

            "groq": {
                "accuracy": 0,
                "relevance": 0,
                "clarity": 0,
                "quality": 0,
                "speed_score": 0,
                "final_score": 0
            },

            "gemini": {
                "accuracy": 0,
                "relevance": 0,
                "clarity": 0,
                "quality": 0,
                "speed_score": 100,
                "final_score": 0
            }
        }

        save_comparison(
            question,
            groq_answer,
            gemini_answer,
            groq_speed,
            gemini_speed,
            result["groq"],
            result["gemini"],
            result["winner"],
            result["reason"]
        )

        return jsonify(result)

    # =========================
    # BOTH AVAILABLE
    # =========================

    if groq_success and gemini_success:

        result = ai_judge(
            question,
            groq_answer,
            gemini_answer,
            groq_speed,
            gemini_speed
        )

        if result.get("success"):

            save_comparison(
                question,
                groq_answer,
                gemini_answer,
                groq_speed,
                gemini_speed,
                result["groq"],
                result["gemini"],
                result["winner"],
                result["reason"]
            )

        return jsonify(result)

    # =========================
    # BOTH FAILED
    # =========================

    return jsonify({
        "success": False,
        "winner": "None",
        "reason": "Both AI services are currently unavailable."
    }), 503


# =========================
# HISTORY API
# =========================

@app.route("/history", methods=["GET"])
def history():

    try:

        conn = sqlite3.connect("database.db")
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM comparisons
            ORDER BY id DESC
        """)

        rows = cursor.fetchall()

        conn.close()

        history_data = []

        for row in rows:
            history_data.append(dict(row))

        return jsonify({
            "success": True,
            "history": history_data
        })

    except Exception as e:

        print("History Error:", e)

        return jsonify({
            "success": False,
            "history": [],
            "error": str(e)
        }), 500


# =========================
# DELETE HISTORY
# =========================

@app.route("/history/clear", methods=["DELETE"])
def clear_history():

    try:

        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        cursor.execute("DELETE FROM comparisons")

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Comparison history cleared successfully."
        })

    except Exception as e:

        print("Clear History Error:", e)

        return jsonify({
            "success": False,
            "message": "Could not clear history.",
            "error": str(e)
        }), 500


# =========================
# OLD COMPATIBILITY ROUTE
# =========================

@app.route("/compare", methods=["POST"])
def compare():

    data = request.get_json() or {}

    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "success": False,
            "error": "Please enter a question."
        }), 400

    # Call Groq
    groq_response = groq()

    if isinstance(groq_response, tuple):
        groq_json = groq_response[0].get_json()
    else:
        groq_json = groq_response.get_json()

    # Call Gemini
    gemini_response = gemini()

    if isinstance(gemini_response, tuple):
        gemini_json = gemini_response[0].get_json()
    else:
        gemini_json = gemini_response.get_json()

    groq_success = groq_json.get(
        "success",
        False
    )

    gemini_success = gemini_json.get(
        "success",
        False
    )

    judge_json = {
        "success": False,
        "winner": "None",
        "reason": "Both AI services are unavailable."
    }

    if groq_success or gemini_success:

        judge_data = {
            "question": text,
            "groq": groq_json.get("answer", ""),
            "gemini": gemini_json.get("answer", ""),
            "groqSpeed": groq_json.get("speed", 0),
            "geminiSpeed": gemini_json.get("speed", 0),
            "groqSuccess": groq_success,
            "geminiSuccess": gemini_success
        }

        with app.test_request_context(
            "/judge",
            method="POST",
            json=judge_data
        ):
            judge_response = judge()

        if isinstance(judge_response, tuple):
            judge_json = judge_response[0].get_json()
        else:
            judge_json = judge_response.get_json()

    return jsonify({
        "success": True,
        "groq": groq_json,
        "gemini": gemini_json,
        "judge": judge_json
    })


# =========================
# START APPLICATION
# =========================

if __name__ == "__main__":

    init_db()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
