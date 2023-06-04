import pinecone
import openai
import streamlit as st
import os

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX_NAME")

model = SentenceTransformer("all-MiniLM-L6-v2")

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

index = pinecone.Index(index_name)


def find_match(user_input):
    input_em = model.encode(user_input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return (
        result["matches"][0]["metadata"]["text"]
        + "\n"
        + result["matches"][1]["metadata"]["text"]
    )


def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate"
        f" a question that would be the most relevant to provide the user with "
        f"an answer from a knowledge base.\n\nCONVERSATION LOG: "
        f"\n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"]


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state["responses"]) - 1):
        conversation_string += (
            "Human: " + st.session_state["requests"][i] + "\n"
        )
        conversation_string += (
            "Bot: " + st.session_state["responses"][i + 1] + "\n"
        )
    return conversation_string
