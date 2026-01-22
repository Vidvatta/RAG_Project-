import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from ingestion.embedder import create_faiss_index
from rag.qa_chain import answer_question
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

index = None
chunks = None

def upload_pdf():
    global index, chunks
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if not file_path:
        return

    text = load_pdf(file_path)
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    index, chunks = create_faiss_index(chunks, EMBEDDING_MODEL)

    messagebox.showinfo("Success", "PDF processed successfully!")

def ask_question():
    if index is None:
        messagebox.showerror("Error", "Upload a PDF first!")
        return

    question = question_entry.get()
    answer = answer_question(question, index, chunks)

    answer_box.delete("1.0", tk.END)
    answer_box.insert(tk.END, answer)

# UI
root = tk.Tk()
root.title("📘 Story PDF – RAG QA System")
root.geometry("800x600")

tk.Button(root, text="Upload Story PDF", command=upload_pdf).pack(pady=10)

question_entry = tk.Entry(root, width=80)
question_entry.pack(pady=10)

tk.Button(root, text="Ask Question", command=ask_question).pack()

answer_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=20)
answer_box.pack(pady=10)

root.mainloop()
