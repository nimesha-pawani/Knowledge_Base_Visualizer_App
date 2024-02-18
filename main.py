import streamlit as st
from neo4j import GraphDatabase
import spacy
import openai
import PyPDF2
from io import BytesIO
import networkx as nx
from pyvis.network import Network
import time
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up Neo4j connection
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(username, password))


nlp = spacy.load("en_core_web_sm")

# Function to process a batch of PDFs
def process_batch_pdf(uploaded_files, document_ids, knowledge_graphs, extracted_data_list):
    G = nx.Graph()

    with driver.session() as session:
        for uploaded_file in uploaded_files:
            # Extract text from each PDF in the batch
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name
            with BytesIO(file_bytes) as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()

            # Split text into chunks of a certain size (adjust size based on token limits)
            chunk_size = 4000
            text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

            # Extract keywords using OpenAI for each text chunk
            keywords = []
            for chunk in text_chunks:
                # Introduce a delay to stay within rate limits
                # time.sleep(5)  # Adjust the sleep duration based on your rate limit
                while True:
                    try:
                        keywords_response = openai.Completion.create(
                            engine="gpt-3.5-turbo-instruct",
                            prompt="Identify the keywords in the following text: " + chunk,
                        )
                        keywords += keywords_response["choices"][0]["text"].split(", ")
                        break  # Break the loop if the request is successful
                    except openai.error.OpenAIError as e:
                        # Handle rate limit errors
                        if 'Rate limit reached' in str(e):
                            time.sleep(60)  # Wait for 1 minute before retrying
                        else:
                            raise  # Re-raise other errors

            # Save data to Neo4j
            document_id = filename.split(".pdf")[0]
            document_ids.add(document_id)

            session.run(
                "MERGE (doc:Document {id: $document_id}) "
                "SET doc.text = $text",
                document_id=document_id,
                text=text,
            )

            for keyword in keywords:
                session.run(
                    "MERGE (kw:Keyword {name: $keyword}) "
                    "MERGE (doc:Document {id: $document_id})-[:CONTAINS]->(kw)",
                    keyword=keyword,
                    document_id=document_id,
                )

                # Create relationships between documents based on shared keywords
                session.run(
                    """
                    MATCH (doc:Document)-[:CONTAINS]->(kw:Keyword)
                    WHERE kw.name = $keyword AND doc.id <> $document_id
                    MERGE (doc)-[:SHARES_KEYWORD]->(otherDoc:Document {id: $document_id})
                    """,
                    keyword=keyword,
                    document_id=document_id,
                )

            # Append extracted data to the list
            extracted_data_list.append({"filename": document_id, "text": text, "keywords": keywords})


# Function to retrieve answer from Neo4j based on the question
def get_answer(question):
    with driver.session() as session:
        # Define additional query patterns or keywords for improved handling
        if "summary" in question.lower() or "overview" in question.lower():
            return "Provide a summary of the document here."
        elif "details" in question.lower() or "specific information" in question.lower():
            return "Provide specific information about the document here."

        # Handle relationship-related questions
        if "contains" in question.lower() or "shares" in question.lower():
            keyword = extract_entity(question, ["NOUN"])
            if keyword is not None:
                result = session.run(
                    f"MATCH (d:Document)-[:CONTAINS|SHARES_KEYWORD]->(k:Keyword) WHERE k.name = $keyword RETURN d.id, k.name",
                    keyword=keyword)
                if result.peek() is None:
                    return f"No information found for the keyword '{keyword}'."
                answer = ""
                for record in result:
                    answer += f"The document '{record['d.id']}' contains/shares the keyword '{record['k.name']}'.\n"
                return answer
            else:
                return "Sorry, I couldn't extract a relevant keyword from your question."

        # Handle document-related questions
        elif "document" in question.lower():
            document_name = extract_entity(question, ["NOUN"])
            if document_name is not None:
                result = session.run(f"MATCH (d:Document {{id: $document_name}}) RETURN d.text",
                                     document_name=document_name)
                record = result.single()
                if record is not None:
                    return record["d.text"]
                else:
                    return f"No information found for the document '{document_name}'."
            else:
                return "Sorry, I couldn't extract a relevant document name from your question."

        # Handle keyword-related questions
        elif "keyword" in question.lower():
            keyword_name = extract_entity(question, ["NOUN"])
            if keyword_name is not None:
                result = session.run(f"MATCH (k:Keyword {{name: $keyword_name}}) RETURN k", keyword_name=keyword_name)
                record = result.single()
                if record is not None:
                    return f"Details of keyword '{keyword_name}': {record['k']}"
                else:
                    return f"No information found for the keyword '{keyword_name}'."
            else:
                return "Sorry, I couldn't extract a relevant keyword name from your question."

        # Handle other types of questions
        else:
            return "Sorry, I don't understand what you're asking."


# Function to extract entities from a question
def extract_entity(question, entity_types):
    # Tokenize the question
    doc = nlp(question)

    # Extract named entities
    named_entities = [ent.text for ent in doc.ents if ent.label_ in entity_types]
    if named_entities:
        return " ".join(named_entities)

    # Extract nouns and proper nouns as keywords
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
    if keywords:
        return " ".join(keywords)

    return None  # Return None if no relevant entities or keywords are found


# Function to handle chatbot interactions
def chatbot(input, history=[]):
    output = get_answer(input)
    history.append((input, output))
    return history, output

# Streamlit interface for uploading PDFs
def upload_section():
    st.sidebar.title("Upload Files")
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        st.sidebar.write("Files uploaded successfully!")
    return uploaded_files

# Streamlit interface for displaying knowledge graph
# Streamlit interface for displaying knowledge graph
def display_graph_section(uploaded_files):
    st.title("Knowledge Base Visualiser App")

    # Define color mappings
    node_colors = {
        "Document": "#33FF57",  # Orange color for documents
        "Keyword": "#ADD8E6",   # Green color for keywords
        "SharesKeyword": "#337AFF"  # Blue color for shares keywords
    }

    # Sidebar widgets for filtering nodes and relationships
    st.sidebar.title("Filter Nodes and Relationships")
    node_types = st.sidebar.multiselect("Select Node Types", ["Document", "Keyword", "SharesKeyword"], default=["Document", "Keyword", "SharesKeyword"])
    relationship_types = st.sidebar.multiselect("Select Relationship Types", ["CONTAINS", "SHARES_KEYWORD"], default=["CONTAINS", "SHARES_KEYWORD"])

    # Streamlit button
    if st.sidebar.button("Process"):  # Moved the button to the sidebar
        extracted_data_list = []  # To store extracted data from multiple files
        document_ids = set()
        knowledge_graphs = []

        # Retrieve data from Neo4j database
        with driver.session() as session:
            # Query Neo4j to retrieve document data
            neo4j_results = session.run(
                """
                MATCH (doc:Document)-[:CONTAINS]->(kw:Keyword)
                RETURN doc.id AS document_id, doc.text AS document_text, COLLECT(kw.name) AS keywords
                """
            )
            for neo4j_record in neo4j_results:
                document_id = neo4j_record["document_id"]
                document_text = neo4j_record["document_text"]
                keywords = neo4j_record["keywords"]
                extracted_data_list.append({"filename": document_id, "text": document_text, "keywords": keywords})

                # Construct knowledge graph for each document
                G = nx.Graph()
                G.add_node(document_id, text=document_text, type="Document")
                for keyword in keywords:
                    G.add_node(keyword, type="Keyword")
                    G.add_edge(document_id, keyword, relationship="CONTAINS")
                knowledge_graphs.append(G)

        # Process PDFs in batches (e.g., batches of 2)
        if uploaded_files is not None:
            with ThreadPoolExecutor() as executor:
                batch_size = 3
                for i in range(0, len(uploaded_files), batch_size):
                    batch_files = uploaded_files[i:i + batch_size]
                    process_batch_pdf(batch_files, document_ids, knowledge_graphs, extracted_data_list)

        # Visualize filtered knowledge graphs in a single image using pyvis
        net = Network(notebook=True, width="100%", height="800px")
        for G in knowledge_graphs:
            for node in G.nodes:
                node_type = G.nodes[node].get("type", None)
                if node_type in node_types:
                    net.add_node(node, label=node, title=node, color=node_colors.get(node_type, "#000000"))  # Default color black if not found
            for edge in G.edges:
                if "relationship" in G[edge[0]][edge[1]]:
                    if G[edge[0]][edge[1]]["relationship"] in relationship_types:
                        net.add_edge(edge[0], edge[1], title=f"{edge[0]} to {edge[1]}",
                                     label=G[edge[0]][edge[1]]["relationship"])
                else:
                    if "CONTAINS" in relationship_types:
                        net.add_edge(edge[0], edge[1], title=f"{edge[0]} to {edge[1]}", label="CONTAINS")

        net.show("knowledge_graphs.html")

        # Display the interactive HTML file
        st.components.v1.html(open("knowledge_graphs.html").read(), height=800)



# Streamlit interface for chat interface
def chat_section():
    st.subheader("Chat")

    # Chat input
    user_query = st.text_input("Ask a question:")

    # Process user queries and generate responses
    if st.button("Submit"):
        if user_query:
            history, response = chatbot(user_query)
            st.text_area("Chatbot:", value=response, height=100)
            # st.write("History:")
            # st.write(history)
        else:
            st.warning("Please enter a question.")

# Main function to run the Streamlit app
def main():
    # Divide the screen into three sections: upload files, processing, and chat interface
    col1, col2, col3 = st.columns([0.3, 4, 2])

    with col1:
        uploaded_files = upload_section()

    with col2:
        display_graph_section(uploaded_files)

    with col3:
        chat_section()

if __name__ == "__main__":
    main()

# Close Neo4j driver connection
driver.close()
