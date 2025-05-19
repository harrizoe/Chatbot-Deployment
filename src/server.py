from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from app import query_rag
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    incoming_msg = request.form.get("Body")
    print(f"📩 Received: {incoming_msg}")
    answer = query_rag(incoming_msg)
    print(f"🤖 Answer: {answer}")
    resp = MessagingResponse()
    resp.message(answer)
    return str(resp)

if __name__ == "__main__":
    app.run(port=5000)
