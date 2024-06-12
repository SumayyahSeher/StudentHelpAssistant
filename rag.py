from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata


try:

#1 reading pdf
    def readPdf(pdf_path):

        docs = PyPDFLoader(file_path=pdf_path).load()

        return docs

    #function for generating chunks
    def generateChunks(document):
        text_splitter= RecursiveCharacterTextSplitter(
            chunk_size= 1000,
            chunk_overlap= 20,
            length_function= len,
        )

        chunks= text_splitter.split_documents(document)

        chunks = filter_complex_metadata(chunks)
        return chunks
    
     #function to create and persist db
    def createVector(documents, embeddings, persist_directory):

        vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)
        #persiste the db to disk
        vector_store.persist()
        vector_store = None

    #function to create promt template
    def promptTemplate():
        template = """Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        {context}

        Question: {question}
        """
        return template
    

    #function to create retrieval chain
    def getRetrievalChain():
        prompt = ChatPromptTemplate.from_template(promptTemplate())
        model = ChatOllama(model="llama3")

        retrieval_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        return retrieval_chain
    

    #chunks=generateChunks(readPdf('student-handbook-2023-2024(2).pdf'))
    
    #print(len(chunks))

    persist_dir='VectorStore'#dir to store vectordb

    embeddings= OllamaEmbeddings(model="nomic-embed-text")
    #print("embedding model choosen")

   

    #calling function to generate db
    #createVector(chunks,embeddings, persist_dir)


    # Now we can load the persisted database from disk, and use it as normal. 
    vector_store = Chroma(persist_directory=persist_dir, 
                   embedding_function=embeddings)
    
    #print( vector_store)
    

    retriever= vector_store.as_retriever()
    
    user_input=input("Question: ")

    while user_input!='exit':
        result = getRetrievalChain().invoke(user_input)
        print("Answer:", result)
        if user_input=='exit':
            break
        else: user_input=input("Question: ")



except Exception as e:
    print(f"An error occurred: {e}")

print("hello")