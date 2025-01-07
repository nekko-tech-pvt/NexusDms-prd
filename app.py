,import streamlit as st
import os
import pandas as pd
import json
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF

# Azure Document Intelligence Client Setup
endpoint = "https://nexusdms-prod.cognitiveservices.azure.com/"
key = "ESfia82tujFgMjU0ijYWqfQ73YqEOVZPB4voGO6uqiSPyiBvcjlmJQQJ99BAACYeBjFXJ3w3AAALACOGW9T6"
document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# Function for querying documents
def query_documents_page():
    st.title("Query Documents")

    # Initialize session state for storing query history
    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    # Sidebar to display query history
    st.sidebar.subheader("Session Query History")
    for i, query in enumerate(st.session_state.query_history):
        st.sidebar.write(f"{i + 1}. {query}")

    # Input for the user to ask a question
    query = st.text_input("Ask a question about the documents (e.g., 'Compare amounts for employee benefits and management')")

    # Process the query and display the results
    if query:
        result = query_documents(query)
        st.session_state.query_history.append(query)  # Append query to the session history
        
        # Display results or generated Python code
        if "```python" in result:
            st.write(result.split("```python")[0])
            code = result.split("```python")[1].split("```")[0]
            print(f"<<<{code}<<<")
            try:
                # Execute and display Plotly chart if generated
                exec_globals = {"px": px, "go": go}
                exec(code, exec_globals)
                if "fig" in exec_globals:
                    st.plotly_chart(exec_globals["fig"])  # Render the Plotly chart in Streamlit
                else:
                    st.error("No valid Plotly figure was generated in the code.")
            except Exception as e:
                st.error(f"Error executing code: {e}")
        else:
            st.write(result)

# Function for processing uploaded files (PDFs and images)
def process_uploaded_file(uploaded_file):
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    temp_file_path = os.path.join("tmp", file_name)
    os.makedirs("tmp", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    if file_type in ["application/pdf"]:
        return process_pdf(temp_file_path)
    elif file_type in ["image/jpeg", "image/png"]:
        return process_image(temp_file_path)
    else:
        st.error("Unsupported file type.")
        return None, None, None, None


# Azure OCR and bounding box extraction for PDFs
def analyze_pdf(pdf_file):
    with open(pdf_file, "rb") as file:
        poller = document_analysis_client.begin_analyze_document(model_id="prebuilt-read", document=file)
        result = poller.result()
    content = ""
    bboxes = []
    for page in result.pages:
        width, height = page.width, page.height
        for line in page.lines:
            content += line.content + "\n"
            bboxes.append([(p.x / width, p.y / height) for p in line.polygon])
    return content, bboxes


# Azure OCR for images
def analyze_image(image_path):
    with open(image_path, "rb") as image_file:
        poller = document_analysis_client.begin_analyze_document(model_id="prebuilt-read", document=image_file)
        result = poller.result()
    content = ""
    bboxes = []
    for page in result.pages:
        width, height = page.width, page.height
        for line in page.lines:
            content += line.content + "\n"
            bboxes.append([(p.x / width, p.y / height) for p in line.polygon])
    return content, bboxes


# Process PDF files
def process_pdf(pdf_file_path):
    inv_content, bboxes = analyze_pdf(pdf_file_path)
    return classify_and_save(inv_content, bboxes, pdf_file_path)


# Process image files
def process_image(image_file_path):
    inv_content, bboxes = analyze_image(image_file_path)
    return classify_and_save(inv_content, bboxes, image_file_path)


# Common classification and saving logic
def classify_and_save(content, bboxes, file_path):
    response = call_gpt4_api(content)
    response = response[7:-3].replace("\n", "").replace("\\n", "")

    try:
        response = json.loads(response)
    except json.JSONDecodeError:
        st.error("Failed to decode JSON response.")

    classification = response.get("category", "Uncategorized")
    destination_folder = os.path.join("Documents", classification)
    os.makedirs(destination_folder, exist_ok=True)
    classified_file_path = os.path.join(destination_folder, os.path.basename(file_path))

    if os.path.exists(classified_file_path):
        os.remove(classified_file_path)

    os.rename(file_path, classified_file_path)

    new_row = {"Classified_File_Path": classified_file_path, "File_Content": content}
    file_path = os.path.join("tmp", "nexus_dms.xlsx")
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        df = df.drop_duplicates(subset=["File_Content"], keep='last')
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_excel(file_path, index=False)
    return classification, response, bboxes, classified_file_path


# Annotate images with bounding boxes
def annotate_image_with_plotly(file_path, bboxes):
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    else:
        img = np.array(Image.open(file_path))

    fig = px.imshow(img)
    for bbox in bboxes:
        shape = bbox_to_shape(bbox, img.shape[1], img.shape[0])
        fig.add_shape(shape)
    return fig


# Convert bounding box to Plotly shape
def bbox_to_shape(bbox, width, height):
    x_min = int(bbox[0][0] * width)
    y_min = int(bbox[0][1] * height)
    x_max = int(bbox[2][0] * width)
    y_max = int(bbox[2][1] * height)
    return {
        "type": "rect",
        "x0": x_min,
        "y0": y_min,
        "x1": x_max,
        "y1": y_max,
        "line": {"color": "rgba(255, 0, 0, 0.8)", "width": 2},
    }


# Upload and annotate documents
def document_upload_page():
    st.title("Nexus DMS: Document Upload and Annotation")

    uploaded_file = st.file_uploader("Upload a file (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])
    if uploaded_file:
        classification, entities, bboxes, classified_file_path = process_uploaded_file(uploaded_file)
        st.sidebar.subheader("Document Classification")
        st.sidebar.write(classification)
        st.sidebar.subheader("Extracted Entities")
        st.sidebar.json(entities)

        st.subheader("Annotated Image")
        if bboxes:
            fig = annotate_image_with_plotly(classified_file_path, bboxes)
            st.plotly_chart(fig)


# Main app function
def main():
    logo_path = "nekko logo black bg.png"  # Update this to the correct path
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Document Upload", "Query Documents"])

    if page == "Document Upload":
        document_upload_page()
    elif page == "Query Documents":
        query_documents_page()

# GPT-4 API call function for classification  
def call_gpt4_api(prompt):  
    url = "https://oainekko.openai.azure.com/openai/deployments/gpt-4o-nekko/chat/completions?api-version=2024-08-01-preview"  
    # url = "https://oainekko.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"
    headers = {  
        "Content-Type": "application/json",  
        "api-key": "cacf6dcb95134cdeab048a36fb6232eb"  
    }  
    messages = [  
        {"role": "system", "content": """You are Nexus DMS, The world most advanced Document Management System. Your task is to go through the document contents and do the following:
        1. Assign a `category` to the document. The `category` should be assigned based on the contents of the document and should be as descriptive as possible. Example an invoice charging the company for background verification of new joinees could be categorized as `Employee Management/Invoice/Background Verification`.
        2. Extract relevant entities/ necessary fields from the document.  (depending upon the type of document)
        3. Return the result in structured JSON format. (Note: All Values are to be saved as String)"""},  
        {"role": "user", "content": f"Please find the document contents extracted as text: ```{prompt}```"}  
    ]  
    payload = {  
        "messages": messages,  
        "temperature": 0.7,  
        "max_tokens": 4096  
    }  
    response = requests.post(url, headers=headers, data=json.dumps(payload))  
    response.raise_for_status()  
    print("***************************")
    print(response.json()["choices"][0]["message"]["content"])
    print("***************************")
    return response.json()["choices"][0]["message"]["content"]  

# Querying function to handle document-based queries  
def query_documents(query):  
    # Simulate a response based on the query content.  
    file_path = os.path.join("tmp", 'nexus_dms.xlsx')
    df = pd.read_excel(file_path)
    # df = df.drop_duplicates()
    relevant_data = df.to_json(orient="records")
    
    # Existing prompt modification
    # Existing prompt modification
    query_prompt = f"""  
    Given the extracted data from the uploaded documents, please respond to the user queries. 
    # Important: Remember these answers are for the leadership team so any valuable additional insights are always appreciated.
    # Always provide all necessary details and quote sources when getting yur answers.
    # Note : If the customer query requires a bar chart or graph, generate the equivalent Python code with all necessary imports 
    # and ensure the code uses Plotly for interactive graph creation (not Matplotlib).
    
    Extracted data (In JSON Formatting): 
        ```
        {json.dumps(relevant_data)}
        ```
    """

    url = "https://oainekko.openai.azure.com/openai/deployments/gpt-4o-nekko/chat/completions?api-version=2024-08-01-preview"  
    # url = "https://oainekko.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"
    headers = {  
        "Content-Type": "application/json",  
        "api-key": "cacf6dcb95134cdeab048a36fb6232eb"  
    }  
    messages = [  
        {"role": "system", "content": query_prompt},  
        {"role": "user", "content": query}  
    ]  
    payload = {  
        "messages": messages,  
        "temperature": 0.7,  
        "max_tokens": 4096  
    }  
    response = requests.post(url, headers=headers, data=json.dumps(payload))  
    response.raise_for_status()  
    print("***************************")
    print(response.json()["choices"][0]["message"]["content"])
    print("***************************")
    return response.json()["choices"][0]["message"]["content"]  

# Function for Document Analytics and Insights
# def atom_analytics():

#     # Mock data for visualization
#     df1 = pd.DataFrame({"Category": ["A", "B", "C"], "Value": [10, 20, 30]})
#     df2 = pd.DataFrame({"Date": pd.date_range(start="2023-01-01", periods=10), "Metric": range(10)})
#     df3 = pd.DataFrame({"Label": ["X", "Y", "Z"], "Count": [100, 200, 300]})

#     fig1 = px.bar(df1, x="Category", y="Value", title="Figure 1: Category Analysis")
#     fig2 = px.line(df2, x="Date", y="Metric", title="Figure 2: Trend Over Time")
#     fig3 = px.pie(df3, names="Label", values="Count", title="Figure 3: Distribution")

#     insights = [
#         "Insight 1: Category B shows the highest value.",
#         "Insight 2: The metric trend indicates steady growth.",
#         "Insight 3: Label Y accounts for the largest share of the distribution."
#     ]

#     return fig1, fig2, fig3, insights


# Function for Nexus Atom page
def nexus_atom_page():
    st.title("Nexus Atom: Intelligent Analytics")

    # # Fetch analytics data
    # fig1, fig2, fig3, insights = atom_analytics()

    # # Display insights in the sidebar
    st.sidebar.subheader("Key Insights")
    # for insight in insights:
    #     st.sidebar.write(f"- {insight}")

    # # Display the three Plotly figures on the main page
    # st.plotly_chart(fig1, use_container_width=True)
    # st.plotly_chart(fig2, use_container_width=True)
    # st.plotly_chart(fig3, use_container_width=True)
        # Input for the user to ask a question
    #1
    query = "You are Nexus Atom, the world's best data analyst known for their Capability to understand data and provide Insights. Your task is to through my documents in detail and provide me with Key Actionable Insights, Action Items, etc and more as you see fit"

    # Process the query and display the results
    if query:
        result = query_documents(query)
        
        # Display results or generated Python code
        if "```python" in result:
            st.write(result.split("```python")[0])
            code = result.split("```python")[1].split("```")[0]
            print(f"<<<{code}<<<")
            try:
                # Execute and display Plotly chart if generated
                exec_globals = {"px": px, "go": go}
                exec(code, exec_globals)
                if "fig" in exec_globals:
                    st.plotly_chart(exec_globals["fig"])  # Render the Plotly chart in Streamlit
                else:
                    st.error("No valid Plotly figure was generated in the code.")
            except Exception as e:
                st.error(f"Error executing code: {e}")
        else:
            st.sidebar.write(result)
    #2
    query = "You are Nexus Atom, the world's best data analyst known for their Capability to understand data and provide Insights. Your task is to through my documents in detail and provide me with a Bar Chart Representation of all my expenses. Provide Necessary legends to the generated Chart for easy understanding and Readability"

    # Process the query and display the results
    if query:
        result = query_documents(query)
        
        # Display results or generated Python code
        if "```python" in result:
            st.write(result.split("```python")[0])
            code = result.split("```python")[1].split("```")[0]
            print(f"<<<{code}<<<")
            try:
                # Execute and display Plotly chart if generated
                exec_globals = {"px": px, "go": go}
                exec(code, exec_globals)
                if "fig" in exec_globals:
                    st.plotly_chart(exec_globals["fig"])  # Render the Plotly chart in Streamlit
                else:
                    st.error("No valid Plotly figure was generated in the code.")
            except Exception as e:
                st.error(f"Error executing code: {e}")
        else:
            st.write(result)
    #3
    query = "You are Nexus Atom, the world's best data analyst known for their Capability to understand data and provide Insights. Your task is to through my documents in detail and provide me with a Sunburst Chart of my Entire Backend Document Structure showing a hierarchial Representation. Provide Necessary legends to the generated Chart for easy understanding and Readability"

    # Process the query and display the results
    if query:
        result = query_documents(query)
        
        # Display results or generated Python code
        if "```python" in result:
            st.write(result.split("```python")[0])
            code = result.split("```python")[1].split("```")[0]
            print(f"<<<{code}<<<")
            try:
                # Execute and display Plotly chart if generated
                exec_globals = {"px": px, "go": go}
                exec(code, exec_globals)
                if "fig" in exec_globals:
                    st.plotly_chart(exec_globals["fig"])  # Render the Plotly chart in Streamlit
                else:
                    st.error("No valid Plotly figure was generated in the code.")
            except Exception as e:
                st.error(f"Error executing code: {e}")
        else:
            st.write(result)
    #4
    query = "You are Nexus Atom, the world's best data analyst known for their Capability to understand data and provide Insights. Your task is to through my documents in detail and based on your understanding Generate a Chart/ Graph that provides some insight about some data or all of my data in my document. You are to intelligently decide which data you shall choose to represent and intelligently choose the graph type for it as well. Provide Necessary legends to the generated Chart for easy understanding and Readability"

    # Process the query and display the results
    if query:
        result = query_documents(query)
        
        # Display results or generated Python code
        if "```python" in result:
            st.write(result.split("```python")[0])
            code = result.split("```python")[1].split("```")[0]
            print(f"<<<{code}<<<")
            try:
                # Execute and display Plotly chart if generated
                exec_globals = {"px": px, "go": go}
                exec(code, exec_globals)
                if "fig" in exec_globals:
                    st.plotly_chart(exec_globals["fig"])  # Render the Plotly chart in Streamlit
                else:
                    st.error("No valid Plotly figure was generated in the code.")
            except Exception as e:
                st.error(f"Error executing code: {e}")
        else:
            st.write(result)


# Main App Function
def main():
    # Display the company logo at the top
    logo_path = "nekko logo black bg.png"  # Update this to the correct path to your logo
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
        
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Document Upload", "Query Documents", "Nexus Atom"])

    if page == "Document Upload":
        document_upload_page()
    elif page == "Query Documents":
        query_documents_page()
    elif page == "Nexus Atom":
        nexus_atom_page()

if __name__ == "__main__":
    main()