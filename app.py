import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
import os

st.set_page_config(page_title="Job-profile_matcher", page_icon=":briefcase:")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Set up the models and configurations
model = genai.GenerativeModel(model_name="models/gemini-pro")
os.environ["GOOGLE_API_KEY"] = "AIzaSyCX3h6oMLzRtYronmp07eZX8xIYhVJFy5A"
gemini_embedding_model = GeminiEmbedding(model_name="models/embedding-001")
llm = Gemini(model_name="models/gemini-pro")

Settings.llm = llm
Settings.embed_model = gemini_embedding_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 2080
Settings.context_window = 3900

Persist_dir = "./storage"

if not os.path.exists(Persist_dir):
    documents = SimpleDirectoryReader("doc").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=Persist_dir)
else:
    storage_context = StorageContext.from_defaults(persist_dir=Persist_dir)
    index = load_index_from_storage(storage_context=storage_context)

query_engine = index.as_query_engine()

memory = ChatMemoryBuffer.from_defaults(token_limit=10000)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    llm=llm,
    context_prompt=(
        "You are a bot that matches job descriptions with candidate profiles. Only provide profiles that are relevant. "
        "Here are the relevant profiles:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=False,
)


def extract_and_rewrite_query(query):
    # Use the model to generate content and extract skills
    gemini_response = model.generate_content(
        f"Extract and list all technical skills from the following job description: {query}. Only include specific technical skills like programming languages, frameworks, or tools."
    )  
    skills = gemini_response.text.strip().split(', ')    
    job_title_response = model.generate_content(
        f"Extract the job title and list key responsibilities from the following job description: {query}. "
    ) 
    job_title = job_title_response.text.split('\n')[0].strip()
    responsibilities = '\n'.join(job_title_response.text.split('\n')[1:]).strip() 
    rewritten_query = (
        f"Job Title: {job_title}\n"
        f"Key Responsibilities:\n"
        f"{responsibilities}\n\n"
        f"Required Skills:\n"
        f"{', '.join(skills)}\n\n"
        f"Objective: Retrieve top 10 profiles with proficiency in the above skills and relevant experience.and the profile should contain atleast two skills in common with the above skills."
    )

    return rewritten_query
    
def main():
    st.title("Job profile matcher")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Let's explore the world of Jobs Together!"}]

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        rewritten_query = extract_and_rewrite_query(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":  
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):     
                response = chat_engine.chat(rewritten_query).response
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
   

if __name__ == "__main__":
    main()
