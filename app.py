from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq
from google import genai
import os
import time
import json

load_dotenv()

app = Flask(__name__)

# -------------------------
# API CLIENTS
# -------------------------

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# -------------------------
# HOME
# -------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -------------------------
# COMPARISON
# -------------------------

@app.route("/compare", methods=["POST"])
def compare():

    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({
            "groq": "Please enter some text.",
            "gemini": "Please enter some text.",
            "groqSpeed": 0,
            "geminiSpeed": 0,
            "groqScore": 0,
            "geminiScore": 0,
            "winner": "No winner"
        })

    # -------------------------
    # GROQ ANSWER
    # -------------------------

    groq_start = time.time()

    try:
        groq_response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": text
                }
            ]
        )

        groq_result = groq_response.choices[0].message.content

    except Exception as e:
        groq_result = f"Groq API Error: {str(e)}"

    groq_speed = round(time.time() - groq_start, 2)


    # -------------------------
    # GEMINI ANSWER
    # -------------------------

    gemini_start = time.time()

    try:
        gemini_response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=text
        )

        gemini_result = gemini_response.text

    except Exception as e:
        gemini_result = f"Gemini API Error: {str(e)}"

    gemini_speed = round(time.time() - gemini_start, 2)


    # -------------------------
    # AI JUDGE
    # -------------------------

    judge_result = ai_judge(
        text,
        groq_result,
        gemini_result
    )


    # -------------------------
    # RETURN RESULT
    # -------------------------

    return jsonify({
        "groq": groq_result,
        "gemini": gemini_result,

        "groqSpeed": groq_speed,
        "geminiSpeed": gemini_speed,

        "groqRelevance": judge_result["groqRelevance"],
        "groqClarity": judge_result["groqClarity"],
        "groqAccuracy": judge_result["groqAccuracy"],
        "groqScore": judge_result["groqOverall"],

        "geminiRelevance": judge_result["geminiRelevance"],
        "geminiClarity": judge_result["geminiClarity"],
        "geminiAccuracy": judge_result["geminiAccuracy"],
        "geminiScore": judge_result["geminiOverall"],

        "winner": judge_result["winner"],
        "reason": judge_result["reason"]
    })


# -------------------------
# AI JUDGE FUNCTION
# -------------------------

def ai_judge(question, groq_answer, gemini_answer):

    prompt = f"""
You are an impartial AI answer evaluator.

Compare the two AI answers below for the given question.

QUESTION:
{question}

GROQ ANSWER:
{groq_answer}

GEMINI ANSWER:
{gemini_answer}

Evaluate both answers using these criteria:

1. Relevance - Does the answer directly answer the question?
2. Accuracy - Is the information correct?
3. Clarity - Is the answer easy to understand?
4. Overall - Overall quality of the answer.

Give each score from 0 to 10.

Return ONLY valid JSON in exactly this format:

{{
    "groqRelevance": 0,
    "groqAccuracy": 0,
    "groqClarity": 0,
    "groqOverall": 0,

    "geminiRelevance": 0,
    "geminiAccuracy": 0,
    "geminiClarity": 0,
    "geminiOverall": 0,

    "winner": "Groq",
    "reason": "Short explanation of why the winner is better."
}}

Do not use markdown.
Do not add anything outside the JSON.
"""

    try:

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a fair and unbiased AI judge."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        # Remove markdown JSON if AI accidentally adds it
        if result.startswith("```"):
            result = result.replace("```json", "")
            result = result.replace("```", "")
            result = result.strip()

        judge = json.loads(result)

        return judge

    except Exception as e:

        print("AI Judge Error:", e)

        return {
            "groqRelevance": 0,
            "groqAccuracy": 0,
            "groqClarity": 0,
            "groqOverall": 0,

            "geminiRelevance": 0,
            "geminiAccuracy": 0,
            "geminiClarity": 0,
            "geminiOverall": 0,

            "winner": "Unable to Judge",
            "reason": "AI Judge could not evaluate the responses."
        }


# -------------------------
# RUN SERVER
# -------------------------

if __name__ == "__main__":
    app.run(debug=True)