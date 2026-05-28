from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import ChatHistory

import os
import json
import fitz
import chromadb
import cohere

from dotenv import load_dotenv

# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# =========================
# COHERE CLIENT
# =========================

co = cohere.Client(COHERE_API_KEY)

# =========================
# CHROMADB
# =========================

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="pdf_collection"
)

# =========================
# HOME PAGE
# =========================

def index(request):

    return render(request, 'index.html')

# =========================
# PDF UPLOAD FUNCTION
# =========================

@csrf_exempt
def upload_pdf(request):

    if request.method == 'POST':

        pdf_file = request.FILES['pdf']

        # CREATE MEDIA FOLDER

        media_path = "media/documents"

        os.makedirs(media_path, exist_ok=True)

        # SAVE PDF

        file_path = os.path.join(media_path, pdf_file.name)

        with open(file_path, 'wb+') as destination:

            for chunk in pdf_file.chunks():

                destination.write(chunk)

        # READ PDF

        doc = fitz.open(file_path)

        full_text = ""

        for page in doc:

            full_text += page.get_text()

        # SPLIT TEXT INTO CHUNKS

        chunks = []

        chunk_size = 500

        for i in range(0, len(full_text), chunk_size):

            chunk = full_text[i:i + chunk_size]

            chunks.append(chunk)

        # CLEAR OLD DATA

        try:
            chroma_client.delete_collection("pdf_collection")
        except:
            pass

        global collection

        collection = chroma_client.get_or_create_collection(
            name="pdf_collection"
        )

        # CREATE EMBEDDINGS

        for index, chunk in enumerate(chunks):

            response = co.embed(
                texts=[chunk],
                model="embed-english-v3.0",
                input_type="search_document"
            )

            embedding = response.embeddings[0]

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[str(index)]
            )

        return JsonResponse({
            'message': 'PDF uploaded successfully!'
        })

    return JsonResponse({
        'error': 'Invalid request'
    })

# =========================
# CHAT FUNCTION
# =========================

@csrf_exempt
def chat(request):

    if request.method == 'POST':

        body = json.loads(request.body)

        question = body.get('question')

        # =========================
        # QUESTION EMBEDDING
        # =========================

        response = co.embed(
            texts=[question],
            model="embed-english-v3.0",
            input_type="search_query"
        )

        question_embedding = response.embeddings[0]

        # =========================
        # SEARCH CHROMADB
        # =========================

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )

        documents = results['documents'][0]

        context = "\n".join(documents)

        # =========================
        # PROMPT
        # =========================

        prompt = f"""
You are a helpful AI PDF assistant.

Answer the user's question ONLY using the PDF content below.

If the answer is not available in the PDF,
reply exactly:
"Answer not found in uploaded PDF."

PDF Content:
{context}

User Question:
{question}

Answer:
"""

        # =========================
        # AI RESPONSE
        # =========================

        response = co.chat(
            model="command-a-03-2025",
            message=prompt
        )

        # =========================
        # SAFE RESPONSE EXTRACTION
        # =========================

        answer = ""

        try:

            if hasattr(response, "text"):

                answer = response.text

            elif hasattr(response, "message"):

                answer = response.message.content[0].text

            else:

                answer = str(response)

        except Exception as e:

            answer = f"Error generating response: {str(e)}"

        # =========================
        # SAVE CHAT HISTORY
        # =========================

        ChatHistory.objects.create(
            question=question,
            answer=answer
        )

        return JsonResponse({
            'answer': answer
        })

    return JsonResponse({
        'error': 'Invalid request'
    })
