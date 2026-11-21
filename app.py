import streamlit as st

from rag_app import load_documents, generate_answer, DOCS_FOLDER
st.sidebar.title("About")

st.sidebar.write("This is a simple RAG application.")

st.sidebar.write("It answers questions using uploaded knowledge documents.")
st.sidebar.subheader("Available Documents")

for doc_file in DOCS_FOLDER.glob("*.txt"):
    st.sidebar.write("📄", doc_file.name)
    
st.title("My RAG Assistant")

st.write("Welcome! Ask questions about Python, Machine Learning, and Artificial Intelligence.")
st.divider()
docs = load_documents(DOCS_FOLDER)

st.write(f"Loaded {len(docs)} document chunks.")
st.subheader("Ask Your Question")
if "question" not in st.session_state:
    st.session_state.question = ""


def clear_question():
    st.session_state.question = ""

if "question" not in st.session_state:
    st.session_state.question = ""

question = st.text_input(
    "Enter your question:",
    key="question"
)

st.button("Clear", on_click=clear_question)

if st.button("Ask"):
    if question.strip():
        answer = generate_answer(question, docs, "")

        st.subheader("Answer")

        if answer.startswith("Source:"):
            source, answer_text = answer.split("\nAnswer:\n", 1)

            st.caption(source)
            st.write(answer_text)
        else:
            st.info(answer)

    else:
        st.warning("Please enter a question.")