# Document-RAG-Chatbot
A Document-based RAG Chatbot built using Python and Streamlit that allows users to upload PDF or TXT files and ask questions about their content. The system uses text chunking, TF-IDF embeddings, and FAISS for similarity search to retrieve relevant information and generate answers without relying on external APIs.
Step-by-Step Procedure for Working of the program
1. File Upload
uploaded_file = st.file_uploader(...)

👉 User uploads a PDF or text file

👉 Streamlit handles the UI

2. Read Document
text = read_file(file_path, uploaded_file.name)

👉 If PDF → uses PdfReader

👉 If TXT → reads normally

# Converts document → plain text

3. Text Chunking
chunks = split_text(text)

# Large text is split into small pieces (chunks)

4. Convert Text to Vectors (Embeddings)
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(chunks)

👉 Each chunk → converted into numbers (vectors)
By Using:
TF-IDF (Term Frequency - Inverse Document Frequency)

👉 Because it gives importance to important words

5. Store in FAISS (Vector Database)
index = faiss.IndexFlatL2(...)
index.add(embeddings)

👉 Stores all vectors

👉 Enables fast similarity search

6. User Question
query = st.text_input(...)

👉 User asks a question

7. Convert Question to Vector
query_vec = vectorizer.transform([query])

👉 Question → vector format

8. Retrieve Relevant Chunks
distances, indices = index.search(query_vec, k=3)

👉 Finds top 3 similar chunks

💡 This is the retrieval step

9. Generate Answer
context = " ".join([chunks[i] for i in indices[0]])

👉 Combines relevant chunks

👉 Shows them as answer

💡 This is:
Retrieval-based response (no AI generation)
