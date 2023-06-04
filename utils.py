import pinecone
import openai
import streamlit as st
import os

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX_NAME")

model = SentenceTransformer("all-MiniLM-L6-v2")

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

index = pinecone.Index(index_name)


# Finding Matches in Pinecone Index
def find_match(user_input: str) -> str:
    input_em = model.encode(user_input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return (
        result["matches"][0]["metadata"]["text"]
        + "\n"
        + result["matches"][1]["metadata"]["text"]
    )


# Refining Queries with OpenAI
def query_refiner(conversation: str, query: str) -> str:
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, "
        f"formulate a question that would be the most relevant to "
        f"provide the user with an answer from a knowledge "
        f"base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: "
        f"{query}\n\nRefined Query:",
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"]


# Tracking the Conversation
def get_conversation_string() -> str:
    conversation_string = ""
    for i in range(len(st.session_state["responses"]) - 1):
        conversation_string += (
            "Human: " + st.session_state["requests"][i] + "\n"
        )
        conversation_string += (
            "Bot: " + st.session_state["responses"][i + 1] + "\n"
        )
    return conversation_string
