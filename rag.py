print("Starting program...")

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    print("Imports successful")

    # Load PDF
    loader = PyPDFLoader("customer_support_kb.pdf")
    docs = loader.load()
    print("PDF Loaded")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print("Chunking done")

    # Embeddings
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embedding)
    print("Database ready")

    retriever = db.as_retriever(search_kwargs={"k": 2})

    print("System ready!")

    while True:
        query = input("Ask: ")
        if query.lower() == "exit":
            break

        # Retrieve context
        docs = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in docs])

        
        query_words = query.lower().split()
        sentences = context.split(".")

        best_sentence = ""
        max_score = 0

        for sentence in sentences:
            score = sum(word in sentence.lower() for word in query_words)
            if score > max_score:
                max_score = score
                best_sentence = sentence

        answer = best_sentence.strip()

        # HITL logic
        if len(answer) < 10:
            print("Escalating to human...")
        else:
            print("Answer:", answer)

except Exception as e:
    print("ERROR OCCURRED:")
    print(e)
