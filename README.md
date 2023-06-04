# Interactive chatbot


This is Python based chatbot that can answer queries from its own documents or knowledge base. This chatbot, leveraging the power of advanced language models, can also respond to follow-up questions from the users, ensuring a seamless and interactive user experience.

## How it works

The application follows these steps to provide responses:

1. The documents in working folder are read and divided into smaller chunks for processing.
2. OpenAI embeddings are used to create vector representations of the text chunks and indexing in [Pinecone](https://www.pinecone.io/).
3. The application finds chunks that are semantically similar to the user's question.
4. The similar chunks are passed to the LLM to generate a response.

The application utilizes [Streamlit](https://streamlit.io/) for the graphical user interface (GUI) and [Langchain](https://python.langchain.com/en/latest/index.html) for LLM integration.


## Installation

1. Clone this repository:

```bash
git clone https://github.com/SergiiMachulin/interactive-chatbot.git
```

2. Install the requirements:

```bash
pip install -r requirements.txt
```

You will also need to add to the `.env` file next environment variables(see `.env.sample`):
#### OpenAI
- OPENAI_API_KEY

#### Pinecone
- PINECONE_API_KEY
- PINECONE_ENVIRONMENT
- PINECONE_INDEX_NAME

## Usage

To use the application:
1. Load your documents to `content/data` directory.

2. Run the `indexing.py` module to load documents from the directory, splitting them into chunks, creating embeddings, and indexing them in Pinecone:

```bash
python indexing.py
```

3. Run chatbot: 

```bash
streamlit run app.py
```

*Please make sure to update the file paths or other details according to your specific*.