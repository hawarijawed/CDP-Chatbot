import os
import requests
from bs4 import BeautifulSoup
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy

# Step 1: Define Schema for Indexing
def create_search_index(index_dir="index_dir"):
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True), url=ID(stored=True))
    ix = create_in(index_dir, schema)
    return ix

# Step 2: Scrape Documentation
def scrape_documentation(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    data = []
    title = "Untitled Section"  # Default title
    subtitle = ""  # Default subtitle

    for section in soup.find_all(['h1', 'h2', 'p']):
        if section.name == 'h1':
            title = section.text.strip()
            subtitle = ""  # Reset subtitle when a new title is encountered
        elif section.name == 'h2':
            subtitle = section.text.strip()
        elif section.name == 'p':
            # Only append if there's valid content in the paragraph
            if section.text.strip():
                data.append((title, subtitle, section.text.strip()))
    
    return data


# Step 3: Index Documentation
def index_documentation(ix, data):
    writer = ix.writer()
    for title, subtitle, content in data:
        writer.add_document(title=title, content=content, url=subtitle)
    writer.commit()

# Step 4: Query Documentation
def search_documentation(ix, query):
    with ix.searcher() as searcher:
        qp = QueryParser("content", schema=ix.schema)
        q = qp.parse(query)
        results = searcher.search(q, limit=5)
        return [(result['title'], result['content'], result['url']) for result in results]

# Step 5: NLP Preprocessing
def preprocess_query(nlp_model, query):
    doc = nlp_model(query)
    return {"entities": [(ent.text, ent.label_) for ent in doc.ents]}

# Step 6: Flask API for Chatbot
app = Flask(__name__)
CORS(app)
nlp = spacy.load("en_core_web_sm")
ix = create_search_index()

def get_documentation_url(query):
    query = query.lower()
    if "segment" in query:
        return "https://segment.com/docs/?ref=nav"
    elif "mparticle" in query:
        return "https://docs.mparticle.com/"
    elif "lytics" in query:
        return "https://docs.lytics.com/"
    elif "zeotap" in query:
        return "https://docs.zeotap.com/home/en-us/"
    return None

@app.route('/query', methods=['POST'])
def query_bot():
    user_query = request.json.get("query", "")
    print(f"Received query: {user_query}")

    # Preprocess query
    nlp_data = preprocess_query(nlp, user_query)
    print(f"NLP Data: {nlp_data}")

    # Determine the documentation URL to scrape
    doc_url = get_documentation_url(user_query)
    if not doc_url:
        return jsonify({"message": "No relevant documentation found for the query.", "nlp_data": nlp_data})

    # Scrape and index documentation
    print(f"Scraping documentation from: {doc_url}")
    data = scrape_documentation(doc_url)
    index_documentation(ix, data)

    # Perform search
    search_results = search_documentation(ix, user_query)
    if not search_results:
        return jsonify({"message": "No relevant content found for your query.", "nlp_data": nlp_data})

    # Format response
    print(search_results)
    formatted_results = [
        {"title": result[0], "content": result[1], "url": result[2]} for result in search_results
    ]
    return jsonify({
        "nlp_data": nlp_data,
        "results": formatted_results,
    })

if __name__ == '__main__':
    app.run(debug=True)
