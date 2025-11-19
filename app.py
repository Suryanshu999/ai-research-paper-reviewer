import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.schema import Document
import traceback
from PyPDF2 import PdfReader
import os

# Embedding Model
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# LLM Setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
llm = ChatGroq(
    model_name="qwen/qwen3-32b",
    temperature=0.3,
    max_tokens=4098,
    groq_api_key=GROQ_API_KEY
)

# Review Function
def review_paper(file):
    try:
        if file is None:
            return "Please upload a Markdown or PDF file."

        content = ""
        if file.name.endswith(".md"):
            with open(file.name, "r", encoding="utf-8") as f:
                content = f.read()
        elif file.name.endswith(".pdf"):
            reader = PdfReader(file.name)
            content = "\n".join([page.extract_text() or "" for page in reader.pages])
            if not content.strip():
                return "This PDF seems to contain scanned images. Text extraction failed."
        else:
            return "Unsupported file format. Please upload .md or .pdf"

        content = content.strip()
        if not content:
            return "The uploaded file is empty or contains no readable text."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(content)

        if not chunks:
            return "The file could not be split into chunks."

        documents = [Document(page_content=chunk) for chunk in chunks]

        vector_store = FAISS.from_documents(
            documents,
            embedding_model,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5, "score_threshold": 0.7}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=retriever,
            chain_type_kwargs={"verbose": True},
            return_source_documents=True
        )

        static_question = "How is this research paper overall based on standard review parameters?"
        result = qa_chain({"query": static_question})
        return result["result"]

    except Exception as e:
        print(traceback.format_exc())
        return f"An error occurred:\n{str(e)}"

# Gradio UI
with gr.Blocks(title="Research Paper Reviewer") as demo:
    gr.Markdown("# AI Research Paper Reviewer")

    with gr.Row():
        file_input = gr.File(
            label="Upload your research paper (.md or .pdf)",
            file_types=[".md", ".pdf"]
        )

    submit_btn = gr.Button("Generate Review")
    output = gr.Textbox(label="Review Output", lines=15)

    submit_btn.click(fn=review_paper, inputs=[file_input], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
