import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Code Generator", page_icon="💻", layout="centered")

st.title("💻 AI Code Generator")
st.write("Enter your programming task below. The model will output **only code** — no text, no explanations.")

@st.cache_resource
def load_model():
    model_name = "Salesforce/codegen-350M-mono"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

prompt = st.text_area(
    "📝 Enter your coding task",
    placeholder="Example: Write a Python program to check if a number is prime"
)

temperature = st.slider("Creativity", 0.0, 1.5, 0.2, 0.1)

if st.button("🚀 Generate Code"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Generating code..."):
            system_prompt = (
                "You are an AI code assistant. "
                "Output ONLY executable code. "
                "Do NOT output explanations, comments, markdown formatting, or text. "
                "Respond only with code."
            )

            full_prompt = system_prompt + "\n\nUser task:\n" + prompt + "\n\nAnswer:\n"

            inputs = tokenizer(full_prompt, return_tensors="pt")

            # 🚫 No max_length or max_new_tokens specified
            output_tokens = model.generate(
                **inputs,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature
            )

            code = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

            # remove system prompt portion if echoed
            if "Answer:" in code:
                code = code.split("Answer:")[-1].strip()

        st.subheader("✅ Generated Code")
        st.code(code, language="python")
