import datetime
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from streamlit_chat import message
from utils import *


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="ChatBot", page_icon=":question:")
st.subheader("Чат-бот підтримки")

if "responses" not in st.session_state:
    st.session_state["responses"] = ["Вітаю! Чим я Вам можу допомогти?"]

if "requests" not in st.session_state:
    st.session_state["requests"] = []

llm = ChatOpenAI(
    temperature=0.2,
    model_name="gpt-3.5-turbo",
    openai_api_key=openai.api_key,
)

if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(
        k=3, return_messages=True
    )

actual_datetime = datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y")

system_msg_template = SystemMessagePromptTemplate.from_template(
    template=f""" You are a helpful assistant answer the question as 
    truthfully as possible using the provided context. If you don't have 
    enough information to answer the question or you don't find necessary 
    information, say only 'Питаю менеджера' and don't give any other 
    information in your response. Use {actual_datetime} as today time and 
    day for answering questions connected with schedule, date or time."""
)

human_msg_template = HumanMessagePromptTemplate.from_template(
    template="{input}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        system_msg_template,
        MessagesPlaceholder(variable_name="history"),
        human_msg_template,
    ],
)

conversation = ConversationChain(
    memory=st.session_state.buffer_memory,
    prompt=prompt_template,
    llm=llm,
    verbose=True,
)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Запит: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            context = find_match(refined_query)
            response = conversation.predict(
                input=f"Context:\n {context} \n\n Query:\n{query}"
            )
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
with response_container:
    if st.session_state["responses"]:
        for i in range(len(st.session_state["responses"])):
            message(st.session_state["responses"][i], key=str(i))
            if i < len(st.session_state["requests"]):
                message(
                    st.session_state["requests"][i],
                    is_user=True,
                    key=str(i) + "_user",
                )
