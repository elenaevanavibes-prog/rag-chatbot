from flask import Flask, request

from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)

from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.exceptions import InvalidSignatureError

from dotenv import load_dotenv

import os
import requests

from bs4 import BeautifulSoup

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain.docstore.document import Document

from langchain_groq import ChatGroq


# LOAD ENVIRONMENT VARIABLES
load_dotenv()

app = Flask(__name__)


# LINE SETTINGS
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")


configuration = Configuration(
    access_token=CHANNEL_ACCESS_TOKEN
)

handler = WebhookHandler(CHANNEL_SECRET)


# LOAD GROQ MODEL
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)


# LOAD LINKS
with open("knowledge_base/links.txt", "r") as file:
    links = file.readlines()


all_text = ""


# SCRAPE WEBSITES
for link in links:

    link = link.strip()

    try:
        response = requests.get(link)

        soup = BeautifulSoup(response.text, "html.parser")

        text = soup.get_text()

        all_text += text

        print(f"Loaded: {link}")

    except:
        print(f"Failed: {link}")


# SPLIT TEXT
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_text(all_text)


# CREATE DOCUMENTS
documents = [Document(page_content=chunk) for chunk in chunks]


# EMBEDDINGS
embeddings = FakeEmbeddings(size=1352)


# VECTOR DATABASE
vectorstore = FAISS.from_documents(documents, embeddings)


# CALLBACK ROUTE
@app.route("/callback", methods=["POST"])
def callback():

    signature = request.headers["X-Line-Signature"]

    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)

    except InvalidSignatureError:
        return "Invalid signature", 400

    return "OK"


# HANDLE LINE MESSAGE
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):

    user_message = event.message.text


    # SEARCH RELEVANT DOCUMENTS
    docs = vectorstore.similarity_search(user_message)


    # CREATE CONTEXT
    context = "\n".join([doc.page_content for doc in docs])


    # PROMPT
    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {user_message}
    """


    # GET AI RESPONSE
    response = llm.invoke(prompt)


    # SEND REPLY TO LINE
    with ApiClient(configuration) as api_client:

        line_bot_api = MessagingApi(api_client)

        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    TextMessage(
                        text=response.content
                    )
                ]
            )
        )


# RUN FLASK APP
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)