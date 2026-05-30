from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain.docstore.document import Document

from langchain_groq import ChatGroq

load_dotenv()


# READ LINKS
with open("knowledge_base/links.txt", "r") as file:
    links = file.readlines()


all_text = ""


# SCRAPE EACH WEBSITE
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


# LOAD GROQ MODEL
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)


print("TTU RAG Chatbot Ready!")
print("Type exit to stop.\n")


while True:

    query = input("Ask Question: ")

    if query.lower() == "exit":
        break

    docs = vectorstore.similarity_search(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)
    print("\n")