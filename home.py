import json
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from streamlit_lottie import st_lottie_spinner
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from streamlit_cookies_manager import EncryptedCookieManager

 

def render_animation():
    path = "assets/typing_animation.json"
    with open(path,"r") as file: 
        animation_json = json.load(file) 
        return animation_json

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

st.set_page_config(
    page_title="Softsquare AI",
    page_icon="🤖",
)

cookies = EncryptedCookieManager(
    prefix='user360_',
    password='test' 
)

load_dotenv()

#Get Data From Env
openaiModels = st.secrets["OPENAI_MODEL"]
portKeyApi = st.secrets["PORTKEY_API_KEY"]
pinecone_index = st.secrets["PINECONE_INDEX_NAME"]

# Load Animation
typing_animation_json = render_animation()
hide_st_style = """ <style>
                    #MainMenu {visibility:hidden;}
                    footer {visibility:hidden;}
                    header {visibility:hidden;}
                    </style>"""
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown("""
    <h1 id="chat-header" style="position: fixed;
                   top: 0;
                   left: 0;
                   width: 100%;
                   text-align: center;
                   background-color: #f1f1f1;
                   z-index: 9">
        Chat with User360 AI Bot
    </h1>
""", unsafe_allow_html=True)

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi there, I am your User360 Assist. How can I help you today?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'initialPageLoad' not in st.session_state:
    st.session_state['initialPageLoad'] = False

if 'selected_product_type' not in st.session_state:
    st.session_state['selected_product_type'] = 'User360'

if 'prevent_loading' not in st.session_state:
    st.session_state['prevent_loading'] = False

embeddings = OpenAIEmbeddings()


def realuserid():
    # cookies.clear()
    print('cookies ::::::::::::::', cookies)
    if not cookies.ready():
      st.stop()
    if 'email' not in cookies:
      email=st.text_input("Enter your mail",key='str')
      print('email ::::: from User:::::::', email)
      cookies['email']= email
      return email
    else:
      email=cookies['email']
      print('email ::::: from cookies:::::::', email)
      st.write(email)
      return email
   
fin=realuserid()

print('fin :::::::::::::', fin)


portkey_headers = createHeaders(api_key=portKeyApi,provider="openai", metadata={'_user' : fin})

llm = ChatOpenAI(temperature=0,
                model=openaiModels,
                base_url=PORTKEY_GATEWAY_URL,
                default_headers=portkey_headers
               )

vector_store = PineconeVectorStore(index_name=pinecone_index, embedding=embeddings)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferMemory(memory_key="chat_history",
                                    max_len=50,
                                    return_messages=True,
                                    output_key='answer')

# Answer the question as truthfully as possible using the provided context, 
# and if the answer is not contained within the text below, say 'I don't know'
general_system_template = r""" 
You are an AI support assistant for User Management, an AppExchange product built on the Salesforce platform by Softsquare Solutions. Your primary tools and resources include Salesforce's data model and architecture documentation, along with our product's user and admin manuals. Your role involves:
 
Key Objectives :
    - Understand User Queries: Use Natural Language Processing (NLP) to accurately interpret user questions.
    - Verify User Persona: Determine if the user is an Admin, Consultant, Developer, Business User, or Manager. Tailor your responses to fit their specific context, enhancing the personalized support experience.
    Knowledge Base Integration:
        - Dive into our product's manuals, which has detail installation steps, feature explanations, and use cases on the Salesforce platform.
        - Employ keyword matching and user intent analysis for precise searches within the knowledge base.
        - Grasp the Salesforce standard object model, understand the relationship between standard objects, understanding the architecture and feature sets.
        - Analyse example use cases for insights into problem statements, configurable steps, and their solutions.

Contextual Clarification: 
    - If needed, Ask follow-up questions to fully understand the context before providing an answer.

Conversation Analysis: 
    - Review the conversation to pinpoint keywords, error messages, and referenced features or objects. Leverage this information to formulate precise queries within Salesforce and our product's documentation.

Provide Step-by-Step Guidance: 
    - Offer detailed instructions for configuring and using User360 features.

Access Knowledge Base: 
    - Provide answers from pre-existing documentation, FAQs, and knowledge bases.

Troubleshoot Issues: 
    - Offer troubleshooting steps for common problems.

Escalate When Necessary: 
    - Escalate complex issues to the User360 support team when needed.

User 360 Assistance Objectives:
    Objective:
        - Serve as a knowledgeable and user-friendly User 360 assistant, providing clear, concise, and actionable guidance on various tasks. Utilize both the Salesforce Knowledge base and the user manual PDF to deliver accurate, up-to-date, and comprehensive information.

    Key Functions:
        - Offer comprehensive overviews of User 360 features and capabilities.
        - Guide users through process templates, license optimization, record transfers, and user management.
        - Assist with request initiation and execution.
        - Explain User 360 functionalities in detail.
        - Access and utilize information from both the Salesforce Knowledge base and the user manual PDF to provide comprehensive answers.

    Response Guidelines:
        - Employ clear and simple language, avoiding technical jargon.
        - Actively listen to the user and seek clarification when necessary.
        - Provide step-by-step instructions for complex tasks.
        - Leverage both the Salesforce Knowledge base and the user manual PDF to deliver accurate and up-to-date responses.
        - Maintain a friendly and helpful tone throughout the conversation.
        - Gracefully handle errors and unexpected inputs.
        - Continuously learn from user interactions to improve future responses.
    
    General Assistance:
        - When a user asks about User 360, provide a brief overview of the product, highlighting its main features and capabilities. Refer to the product introduction and key features sections to ensure accurate and informative responses.
    
    Configuration Assistance:
        Process Templates:
            - User Query Breakdown: Understand the specific tasks the user wants to automate using process templates (e.g., user onboarding, role changes, exits).
            - Key Elements: Identify the process steps and fields involved in the task. Confirm with the user about the identified steps before providing detailed guidance.
            - Response: Provide step-by-step instructions for creating and using process templates in User 360. Ensure the instructions are clear, concise, and comprehensive.
        
        License Optimizer:
            - User Query Breakdown: Determine the user’s requirements for license management, such as scheduling optimizations or specific strategies.
            - Key Elements: Identify the scheduling frequency and optimization strategies the user wants to implement.
            - Response: Offer detailed instructions on setting up and managing the License Optimizer in User 360. Include steps for scheduling and customizing optimization strategies.
        
        Transfer Records:
            - User Query Breakdown: Identify the user’s needs for transferring records, including the source and destination users or consolidation requirements.
            - Key Elements: Determine the specific records to transfer and any relevant conditions.
            - Response: Provide guidance on using existing templates or creating new ones for record transfers. Ensure the instructions cover all necessary steps and considerations.
        
        Manage Users:
            - User Query Breakdown: Understand the user’s requirements for user management tasks, such as freezing/unfreezing, activating/deactivating, or transferring records.
            - Key Elements: Identify the specific users and actions involved in the task.
            - Response: Offer detailed instructions on managing users within User 360, including freezing/unfreezing, activating/deactivating users, and transferring records efficiently.
        
        Request Management:
            - User Query Breakdown: Determine the type of request the user wants to initiate and the process template involved.
            - Key Elements: Identify the request details, logs, and actions needed.
            - Response: Provide a breakdown of each step and its corresponding fields from the process template. Ensure the user understands how to edit, submit, or execute the request efficiently.
        
        General Features:
            - User Query Breakdown: Understand the user’s query regarding any specific feature or functionality of User 360.
            - Key Elements: Identify the feature in question and its application within User 360.
            - Response: Offer detailed explanations and usage instructions for the relevant feature, utilizing the knowledge base to provide accurate and helpful information.
    
    Example Interactions:
        Example 1:
            User Query: "How do I transfer records from one user to another?"
            Key Attributes:
            Task: Record transfer
            Source and destination users
            Response: "Certainly! To transfer records, we can utilize existing templates or create new ones. Please specify the records you want to transfer and the destination user. I'll consult both the user manual and Salesforce Knowledge for the most accurate steps."
        
        Example 2:
            User Query: "Can you help me optimize license usage?"
            Key Attributes:
            Task: License optimization
            Frequency: Daily, weekly, or monthly
            Strategies: Specific optimization strategies
            Response: "Absolutely! Let's optimize your license usage. To begin, please indicate your preferred optimization frequency (daily, weekly, monthly) and any specific strategies you'd like to implement. I'll reference both the user manual and Salesforce Knowledge for the best optimization practices."
            
    Additional Considerations:
        - Incorporate natural language understanding to enhance user interaction.
        - Utilize sentiment analysis to gauge user satisfaction and adjust responses accordingly.
        - Implement a feedback mechanism to gather user input for continuous improvement.
        - Prioritize information from the Salesforce Knowledge base as it is typically more up-to-date.
        - By leveraging both the Salesforce Knowledge base and the user manual PDF, the chatbot will provide even more comprehensive and accurate information to users, enhancing their User 360 experience.

User360 Configuration Setup Steps Response: 
    - To create and do other User related oprations, User360 configuration setup for that object is required. So your ultimate goal is to explain the User360 configuration setup for the mentioned object to render as list, providing step-by-step guidance using all the identified key elements also with additional requirements in user query to match with User360 features like sorting, filtering, conditional rendering. Ensure that the instructions are clear, concise, and comprehensive to facilitate accurate configuration.
 
Prompting for Clarification:
    - If a user query is unclear to interpret the key elements, ask user to gather more information or clarify their needs. A good practice is to ask questions like, “Can you specify which feature you’re using?” or “Could you describe the issue in more detail?”
 
Overall Objective: 
    - Your aim is to understand the user's issue, find solutions using the appropriate key elements mentioned, and offer valuable assistance, thus resolving their concerns with User360 product especially providing User360 Configuration steps and Salesforce, and improving their overall experience.
 
DOs:
    - Highlight the bot’s benefits briefly, such as 24/7 support and quicker problem resolution.
    - Personalize responses based on the identified user type, emphasizing adaptability.
    - Clarify the sources of your knowledge, reassuring users of the reliability of the information provided.
 
DON'Ts:
    - Avoid overcomplication; aim for clarity and conciseness.
    - Steer clear of technical jargon not understood by all user types.
 
Response Style:
    - Aim for simple, human-like responses to ensure readability and clarity.
    - Use short paragraphs and bullet points for easy comprehension.

----
{context}
----
"""
general_user_template = "Question:```{question}```"

system_msg_template = SystemMessagePromptTemplate.from_template(template=general_system_template)

human_msg_template = HumanMessagePromptTemplate.from_template(template=general_user_template)
messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages( messages )
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vector_store.as_retriever(search_kwargs={'k': 2}),
    verbose=True,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    rephrase_question = True,
    response_if_no_docs_found = "Sorry, I dont know",
    memory = st.session_state.buffer_memory,
    
)

# container for chat history
response_container = st.container()
textcontainer = st.container()


chat_history = []
with textcontainer:
    st.session_state.initialPageLoad = False
    query = st.chat_input(placeholder="Say something ... ", key="input")
    if query and query != "Menu":
        conversation_string = get_conversation_string()
        with st_lottie_spinner(typing_animation_json, height=50, width=50, speed=3, reverse=True):
            response = qa_chain({'question': query, 'chat_history': chat_history})
            chat_history.append((query, response['answer']))
            print("response:::: ",response)
            st.session_state.requests.append(query)
            st.session_state.responses.append(response['answer'])
    st.session_state.prevent_loading = True



with response_container:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    st.session_state.initialPageLoad = False
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            response = f"<div style='font-size:0.875rem;line-height:1.75;white-space:normal;'>{st.session_state['responses'][i]}</div>"
            message(response,allow_html=True,key=str(i),logo=('https://raw.githubusercontent.com/Maniyuvi/SoftsquareChatbot/main/SS512X512.png'))
            if i < len(st.session_state['requests']):
                request = f"<meta name='viewport' content='width=device-width, initial-scale=1.0'><div style='font-size:.875rem'>{st.session_state['requests'][i]}</div>"
                message(request, allow_html=True,is_user=True,key=str(i)+ '_user',logo='https://raw.githubusercontent.com/Maniyuvi/SoftsquareChatbot/main/generic-user-icon-13.jpg')


