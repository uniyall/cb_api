from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from openai import ChatCompletion
from langchain.document_loaders import UnstructuredURLLoader
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    try:
            data = request.get_json()
            query = data['query']
            url=['https://brainlox.com/courses/category/technical']
            loaders=UnstructuredURLLoader(urls=url)
            data=loaders.load()
            document_text=""
            for i in data:
                if isinstance(i.page_content, bytes):
                    document_text= document_text+i.page_content.decode('utf-8')
                else:
                    document_text+=i.page_content

            text_splitter=CharacterTextSplitter(separator="\n",
                                                chunk_size=1000,
                                                chunk_overlap=200,
                                                length_function=len)
            docs=text_splitter.split_text(document_text)
            embeddings = OpenAIEmbeddings()
            
            db = FAISS.from_texts(docs, embeddings)
            docs_and_scores = db.similarity_search_with_score(query)
            docs= db.similarity_search(query)

            text_resp=""
            for i in range (len(docs_and_scores)):
                text_resp=text_resp+docs_and_scores[i][0].page_content
            xtest=""
            for i in range(len(docs)):
                xtest=xtest+docs_and_scores[i][0].page_content

            llm = ChatOpenAI(temperature=1.0, model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)      
            return jsonify(response)
    except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == '__main__':
     app.run(debug=True)