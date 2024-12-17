import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# Load API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM (Groq)
llm = ChatGroq(groq_api_key=api_key, model="llama-3.3-70b-versatile")

# Set up memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt Template for Extracting Categories and Products
category_and_product_template = PromptTemplate(
    input_variables=["input"],
    template="""You will be provided with customer service queries.

Output a python list of objects, where each object has the following format:
    'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
OR
    'products': <a list of products that must be found in the allowed products below>

Allowed products:
Computers and Laptops category:
TechPro Ultrabook
BlueWave Gaming Laptop
PowerLite Convertible
TechPro Desktop
BlueWave Chromebook

Smartphones and Accessories category:
SmartX ProPhone
MobiTech PowerCase
SmartX MiniPhone
MobiTech Wireless Charger
SmartX EarBuds

Televisions and Home Theater Systems category:
CineView 4K TV
SoundMax Home Theater
CineView 8K TV
SoundMax Soundbar
CineView OLED TV

Gaming Consoles and Accessories category:
GameSphere X
ProGamer Controller
GameSphere Y
ProGamer Racing Wheel
GameSphere VR Headset

Audio Equipment category:
AudioPhonic Noise-Canceling Headphones
WaveSound Bluetooth Speaker
AudioPhonic True Wireless Earbuds
WaveSound Soundbar
AudioPhonic Turntable

Cameras and Camcorders category:
FotoSnap DSLR Camera
ActionCam 4K
FotoSnap Mirrorless Camera
ZoomMaster Camcorder
FotoSnap Instant Camera

Only output the list of objects, with nothing else:
{input}"""
)

# Prompt Template for Customer Service Response
customer_service_template = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="""You are a helpful and friendly customer service assistant for a large electronics store. 

Use the chat history to maintain context. Respond to the user's current question in a concise and friendly tone. If needed, ask relevant follow-up questions.

Chat History:
{chat_history}

Current Query:
{input}

Response:"""
)

# Initialize Chains
extract_chain = LLMChain(llm=llm, prompt=category_and_product_template, output_key="products")
response_chain = LLMChain(llm=llm, prompt=customer_service_template, memory=memory, output_key="response")

# Streamlit UI
st.title("🛒 Customer Service Assistant - Electronics Store")

# Initialize Session State for Memory
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = ""

# User Query Input
user_query = st.text_input("Enter your question about our products:")

if user_query:
    # Step 1: Extract Categories and Products
    extraction_result = extract_chain.run({"input": user_query})

    # Step 2: Generate Customer Service Response
    final_response = response_chain.run({
        "input": user_query,
        "chat_history": st.session_state["chat_history"],
    })

    # Display the Response
    st.write("**Customer Service Response:**")
    st.write(final_response)

    # Update chat history
    st.session_state["chat_history"] += f"User: {user_query}\nAssistant: {final_response}\n"

# Display Chat History
st.subheader("Chat History")
st.text(st.session_state["chat_history"])
