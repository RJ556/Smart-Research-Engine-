import streamlit as st
from rag_system_pipeline_codes import url
from rag_system_pipeline_codes import chunker
from rag_system_pipeline_codes import dbstoragemanager
from rag_system_pipeline_codes import processing_manager
from rag_system_pipeline_codes import structuring_response_manager
from rag_system_pipeline_codes import get_the_answer
from rag_system_pipeline_codes import pdf_processing_manager
import tempfile
st.title("⚡Smart Research Engine")
st.sidebar.title("Try These Things Using This Tool")
st.sidebar.write("""
📝 Research on blogs \n
🛍️ Summarize & evaluate product pages \n
📄 Research on PDF \n
🌐 Research on multiple websites \n
""")

uploaded_url = st.text_input("Enter A Valid Url")
uploaded_pdf = st.file_uploader('Upload The Pdf', type=["pdf"])
## using session state to preserve the varibale created_vecdb_1 so that when the streamlit script reruns the varibale get defined again

##if ['created_vecdb_1'] not in st.session_state:
    ##st.session_state['created_vecdb_1'] = None


if uploaded_url.strip():  ## this will prevent the running of the function even when the url box is empty
    ## this session code will ensure :
    # that after the running of preprocessing_manager function once for an URL it stores the variable in the session state
    # and also checks for if user has given an url which is new
    # so this combination will help to solve the problem which is calling the preprocessing_manager function everytime user enters new query and clicks get_the_answer button
    if ("created_vecdb_1" not in st.session_state) or ("last_uploaded_url" not in st.session_state) or (st.session_state.get("last_uploaded_url") != uploaded_url):
        #these 3 conditions ensure that the both parts runs independently and stores their vectors in the same vectordatabase and also donot reruns again and again on the clicking of get_the_answer button by the user
        #if st.button('Process The url'):
        st.session_state.created_vecdb_1 = processing_manager(uploaded_url)
        st.session_state.last_uploaded_url = uploaded_url
        st.success("URL Processed Successfully")
    #created_vecdb_1 = processing_manager(uploaded_url)
    #st.success('URL Processed Successfully')
user_query = st.text_input("Enter A Query")
if uploaded_pdf is not None:
    if ("created_vecdb_1" not in st.session_state) or ("last_uploaded_pdf" not in st.session_state) or (st.session_state.get("last_uploaded_pdf") != uploaded_pdf):
    #if "created_vecdb_1" not in st.session_state or st.session_state.uploaded_pdf != uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file: ## this ensures pdf path is extracted by the pdf given from user and that path is given to the function
            tmp_file.write(uploaded_pdf.read())
            pdf_path = tmp_file.name
        #pdf_bytes = uploaded_pdf.read()
        st.session_state.created_vecdb_1 = pdf_processing_manager(pdf_path)
        st.session_state.last_uploaded_pdf = uploaded_pdf
        st.success("PDF Processed Successfully")
        #if user_query is not None:
if st.button('Get The Answer'):
    with st.spinner("Getting Your Answer"):
        answer = get_the_answer(st.session_state.created_vecdb_1, user_query)  ## here the created_vecdb_1 is showing not defined because the rerun of streamlit script and that time the created_vecdb is not defined
    st.success(f"Answer Of Your Question Is: \n\n  {answer}")

with st.expander("👨‍💻 About the Developer"):
    st.write("""
**Name:** Ranveer Raj    
**About:**  
- Pursuing B.Tech In NIT Raipur
- Builds AI-powered tools  
- Intrested In Building Complete Pipeline Of AI Products  
- Passionate about automation & AI  
""")
#else :
    #st.error('Please Give The Query')


    

