# main.py
import os
import sys
from huggingface_qa_inference import ExtractiveQAPipeline
from pdf_processing import pdf_to_text_chunks
from indexing import PDFVectorStore
from llama_inference import LlamaQA
##from huggingface_qa_inference import ExtractiveQAPipeline  # optional alternative

def main():
    # 1. Build or load your vector store
    vector_store = PDFVectorStore(embedding_model="all-MiniLM-L6-v2")

    # 2. Process and index PDF files
    pdf_files = ["1-G0.5ArchSpecs.pdf", "2-CaliberExhibits.pdf"]
    for pdf_file in pdf_files:
        chunks = pdf_to_text_chunks(pdf_file, chunk_size=5000, overlap=500)        
        vector_store.add_documents(chunks)
        print("\nextracted chunk count: ", chunks.__len__(), "\n")

    # 3. Initialize your chosen QA model
    # For LLaMA:
    ##llama_path = "tiiuae/Falcon3-7B-Instruct"  # local path or Hugging Face repo
    ##llama_qa = LlamaQA(model_path=llama_path, low_precision=False)

    # (Alternatively, for a smaller QA model)
    qa_model = ExtractiveQAPipeline(model_name="deepset/roberta-base-squad2")

    # 4. Ask the user a question
    while True:
        user_question = input("Enter your question: ")
        if not user_question:
            print("Exiting...")
            break

        # 5. Retrieve top chunks from the vector store
        top_chunks = vector_store.search(user_question, top_k=6)
        #print("found chunks\n", top_chunks)
        # 6. Generate an answer using LLaMA (or the smaller QA model)
        ##answer = llama_qa.answer_question(user_question, top_chunks)
        answer = qa_model.answer_question(user_question, top_chunks)  # if using extractive QA

        # 7. Print the result
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()