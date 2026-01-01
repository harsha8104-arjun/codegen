import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(
    page_title="AI Code Generator",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Code Generator")
st.write("Generate high-quality Python code using open-source code models.")

# Sidebar settings
st.sidebar.header("⚙️ Model & Generation Settings")

model_name = st.sidebar.selectbox(
    "Choose Model",
    [
        "Salesforce/codegen-350M-mono",
        "Salesforce/codegen-2B-mono",
        "Salesforce/codegen-6B-mono"
    ]
)

temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.5, 0.3, 0.05)
max_tokens = st.sidebar.slider("Max New Tokens", 50, 500, 200, 10)

st.sidebar.info("Higher temperature = more creative but less precise")

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return tokenizer, model

with st.spinner("Loading model… This may take some time the first run ⏳"):
    tokenizer, model = load_model(model_name)

# User query input box
user_query = st.text_area(
    "📝 Enter your coding task",
    placeholder="Example: Write a Python program that finds the factorial of a number using recursion",
    height=140
)

# Generate button
if st.button("🚀 Generate Code"):
    if user_query.strip() == "":
        st.warning("Please enter a task description.")
    else:
        with st.spinner("Generating code…"):
            prompt = user_query + "\n\n### Python Code:\n"

            inputs = tokenizer(prompt, return_tensors="pt")

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )

            code = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("✅ Generated Code")
        st.code(code, language="python")

        st.download_button(
            label="📥 Download code as .py file",
            data=code,
            file_name="generated_code.py"
        )

st.caption("💡 Tip: For best quality, use 2B or 6B models with GPU.")
