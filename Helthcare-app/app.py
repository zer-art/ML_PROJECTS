import streamlit as st
from src.functions import load_rag_chain

rag = load_rag_chain()

# --- Setup page ---
st.set_page_config(page_title="🩺 Medical Assistant", layout="wide")

# --- Icons (you can customize) ---
user_avatar = "👤"
bot_avatar = "🤖"

# --- Chatbot logic placeholder ---
def get_medical_response(message):
   response = rag.invoke(
    {
        "input": message,
    }
   )
   output = f"""
**Medical Assistant**   

👤 : *{message}?*  

🤖 : {response['answer']} Let me know more symptoms for better context.
"""
   return output
 

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 About")
    st.markdown("""
    This is a **Medical Assistant Chatbot** built with **Langchain, Pinecone , Gemini** and **Streamlit**.

    Ask health-related questions like:
    - What does a sore throat mean?
    - Remedies for high fever
    - Symptoms of Vitamin D deficiency

    ⚠️ This is not a substitute for professional medical advice.
    """)
    if st.button("🧹 Clear Chat"):
        st.session_state.messages.clear()
        

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")

# --- Main Chat Interface ---
st.markdown("## 🩺 Medical Chat Assistant")
chat_container = st.container()

# Display chat messages
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=user_avatar if msg["role"] == "user" else bot_avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)

# Input box at bottom
user_input = st.chat_input("Ask your medical question...")

if user_input:
    # Append user's message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message immediately
    with chat_container:
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(user_input)

    # Get bot response
    response = get_medical_response(user_input)

    # Append bot response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display bot response
    with chat_container:
        with st.chat_message("assistant", avatar=bot_avatar):
            st.markdown(response, unsafe_allow_html=True)

# --- Disclaimer at bottom ---
with st.expander("⚠️ Medical Disclaimer"):
    st.markdown("""
    This assistant provides general medical information based on your inputs.
    **It does not replace a consultation with a qualified medical professional.**

    Always consult your doctor for medical advice, diagnosis, or treatment.
    """)
