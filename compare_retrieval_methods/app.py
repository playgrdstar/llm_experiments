import gradio as gr
# from dotenv import load_dotenv
# load_dotenv()

import warnings
warnings.filterwarnings("ignore")

import os, requests, shutil
from collections import defaultdict
from itertools import chain

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma  
from langchain.llms import HuggingFaceEndpoint
from langchain.storage import InMemoryStore
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever, BM25Retriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, EmbeddingsFilter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate


HF_READ_API_KEY = os.environ["HF_READ_API_KEY"]

def get_text(docs):
    return ['Result ' + str(i+1) + '\n' + d.page_content + '\n' for i, d in enumerate(docs)]

def load_pdf(path):
    loader = PyMuPDFLoader(path)
    docs = loader.load()

    return docs, 'PDF loaded successfully'


def multi_query_retrieval(query, llm, retriever):
    DEFAULT_QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant. Generate 3 different versions of the given question to retrieve relevant docs. 
        Provide these alternative questions separated by newlines. 
        Original question: {question}""",
    )
    mq_llm_chain = LLMChain(llm=llm, prompt=DEFAULT_QUERY_PROMPT)
    
    generated_queries = mq_llm_chain.invoke(query)['text'].split("\n")
    all_queries = [query] + generated_queries
    
    all_retrieved_docs = []
    for q in all_queries:
        retrieved_docs = retriever.get_relevant_documents(q)
        all_retrieved_docs.extend(retrieved_docs)
    
    unique_retrieved_docs = [doc for i, doc in enumerate(all_retrieved_docs) if doc not in all_retrieved_docs[:i]]
    
    return get_text(unique_retrieved_docs)

def compressed_retrieval(query, llm, retriever, extractor_type='chain', embedding_model=None):
    retrieved_docs = retriever.get_relevant_documents(query)
    if extractor_type == 'chain':
        extractor = LLMChainExtractor.from_llm(llm)
    elif extractor_type == 'filter':
        extractor = LLMChainFilter.from_llm(llm)
    elif extractor_type == 'embeddings':
        if embedding_model is None:
            raise ValueError("Embeddings model must be provided for embeddings extractor.")
        extractor = EmbeddingsFilter(embeddings=embedding_model, similarity_threshold=0.5)
    else:
        raise ValueError("Invalid extractor_type. Options are 'chain', 'filter', or 'embeddings'.")
    compressed_docs = extractor.compress_documents(retrieved_docs, query)
    return get_text(compressed_docs)

def unique_by_key(iterable, key_func):
    seen = set()
    for element in iterable:
        key = key_func(element)
        if key not in seen:
            seen.add(key)
            yield element

def ensemble_retrieval(query, retrievers_list, c=60):
    retrieved_docs_by_retriever = [retriever.get_relevant_documents(query) for retriever in retrievers_list]
    weights = [1 / len(retrievers_list)] * len(retrievers_list)
    rrf_score = defaultdict(float)
    for doc_list, weight in zip(retrieved_docs_by_retriever, weights):
        for rank, doc in enumerate(doc_list, start=1):
            rrf_score[doc.page_content] += weight / (rank + c)
            
    all_docs = chain.from_iterable(retrieved_docs_by_retriever)
    sorted_docs = sorted(
        unique_by_key(all_docs, lambda doc: doc.page_content),
        key=lambda doc: rrf_score[doc.page_content],
        reverse=True
    )
    return get_text(sorted_docs)

def long_context_reorder_retrieval(query, retriever):
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_docs.reverse() 
    reordered_results = []
    for i, doc in enumerate(retrieved_docs):
        if i % 2 == 1:
            reordered_results.append(doc) 
        else:
            reordered_results.insert(0, doc)
    return get_text(reordered_results)

def process_query(docs, query, embedding_model, inference_model, retrieval_method, chunk_size, chunk_overlap, max_new_tokens, temperature, top_p):

    
    chunking_parameters = {'chunk_size': chunk_size, 'chunk_overlap': chunk_overlap}
    inference_model_params = {'max_new_tokens': max_new_tokens, 'temperature': temperature, 'top_p': top_p}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunking_parameters['chunk_size'], chunk_overlap=chunking_parameters['chunk_overlap'])

    texts = text_splitter.split_documents(docs)

    hf = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_db_from_docs = Chroma.from_documents(texts, hf)
    simple_retriever = vector_db_from_docs.as_retriever(search_kwargs={"k": 5})

    llm_model = HuggingFaceEndpoint(repo_id=inference_model,                     
                            max_new_tokens=inference_model_params['max_new_tokens'], 
                            temperature=inference_model_params['temperature'], 
                            top_p=inference_model_params['top_p'],
                            huggingfacehub_api_token=HF_READ_API_KEY)

    if retrieval_method == "Simple":
        retrieved_docs = simple_retriever.get_relevant_documents(query)
        result = get_text(retrieved_docs)
    elif retrieval_method == "Parent & Child":
        parent_text_splitter = child_text_splitter = text_splitter
        vector_db = Chroma(collection_name="parent_child", embedding_function=hf)
        store = InMemoryStore()
        pr_retriever = ParentDocumentRetriever(
            vectorstore=vector_db,
            docstore=store,
            child_splitter=child_text_splitter,
            parent_splitter=parent_text_splitter,
        )
        pr_retriever.add_documents(docs)
        retrieved_docs = pr_retriever.get_relevant_documents(query)
        result = get_text(retrieved_docs)
    elif retrieval_method == "Multi Query":
        result = multi_query_retrieval(query, llm_model, simple_retriever)
    elif retrieval_method == "Contextual Compression (chain extraction)":
        result = compressed_retrieval(query, llm_model, simple_retriever, extractor_type='chain')
    elif retrieval_method == "Contextual Compression (query filter)":
        result = compressed_retrieval(query, llm_model, simple_retriever, extractor_type='filter')
    elif retrieval_method == "Contextual Compression (embeddings filter)":
        result = compressed_retrieval(query, llm_model, simple_retriever, extractor_type='embeddings', embedding_model=hf)
    elif retrieval_method == "Ensemble":
        bm25_retriever = BM25Retriever.from_documents(docs)
        all_retrievers = [simple_retriever, bm25_retriever]
        result = ensemble_retrieval(query, all_retrievers)
    elif retrieval_method == "Long Context Reorder":
        result = long_context_reorder_retrieval(query, simple_retriever)
    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}")
    
    
    prompt_template = PromptTemplate.from_template(
        "Answer the query {query} with the following context:\n {context}. If you cannot use the context to answer the query, say 'I cannot answer the query with the provided context.'"
    )
    
    answer = llm_model.invoke(prompt_template.format(query=query, context=result))

    return "\n".join(result), answer.strip()

embedding_model_list = ['sentence-transformers/all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5', 'BAAI/bge-large-en-v1.5'] 
inference_model_list = ['google/gemma-2b-it', 'google/gemma-7b-it', 'microsoft/phi-2', 'mistralai/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mistral-7B-Instruct-v0.2']
retrieval_method_list = ["Simple", "Parent & Child", "Multi Query", 
                         "Contextual Compression (chain extraction)", "Contextual Compression (query filter)",
                         "Contextual Compression (embeddings filter)", "Ensemble", "Long Context Reorder"]


with gr.Blocks() as demo:
    gr.Markdown("## Compare Retrieval Methods for PDFs")
    with gr.Row():
        with gr.Column():
            pdf_url = gr.Textbox(label="Enter URL to PDF", value="https://www.berkshirehathaway.com/letters/2023ltr.pdf")
            load_button = gr.Button("Load and process PDF")
            status = gr.Textbox(label="Status")
            docs = gr.State()
            load_button.click(load_pdf, inputs=[pdf_url], outputs=[docs, status])
            
            query = gr.Textbox(label="Enter your query", value="What does Warren Buffet think about Coca Cola?")
            with gr.Row():
                embedding_model = gr.Dropdown(embedding_model_list, label="Select Embedding Model", value=embedding_model_list[0])
                inference_model = gr.Dropdown(inference_model_list, label="Select Inference Model", value=inference_model_list[0])
            retrieval_method = gr.Dropdown(retrieval_method_list, label="Select Retrieval Method", value=retrieval_method_list[0])
            
            with gr.Row():
                chunk_size = gr.Number(label="Chunk Size", value=1000)
                chunk_overlap = gr.Number(label="Chunk Overlap", value=200)
            
            with gr.Row():
                max_new_tokens = gr.Number(label="Max New Tokens", value=100)
                temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Temperature", value=0.7)
                top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Top P", value=0.9)
            
            search_button = gr.Button("Retrieval")
        with gr.Column():
            answer = gr.Textbox(label="Answer")
            retrieval_output = gr.Textbox(label="Retrieval Results")
            
    search_button.click(process_query, inputs=[docs, query, embedding_model, inference_model, retrieval_method, chunk_size, chunk_overlap, max_new_tokens, temperature, top_p], outputs=[retrieval_output, answer])

if __name__ == "__main__":
    demo.launch()