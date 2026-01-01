import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Code Generator", page_icon="💻", layout="centered")

st.title("💻 AI Code Generator")
st.caption("Outputs ONLY executable Python code. No comments, docstrings, or explanations.")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model_name = "Salesforce/codegen-350M-mono"  # small model for cloud deployment
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- User Input ----------------
prompt = st.text_area(
    "📝 Enter your coding task",
    placeholder="Example: Write a Python function to check if a number is prime",
    height=140
)

st.sidebar.header("⚙️ Generation Settings")
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.0, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 200, 10)

# ---------------- Generate Code ----------------
if st.button("🚀 Generate Code"):
    if not prompt.strip():
        st.warning("Please enter a coding task first.")
    else:
        with st.spinner("Generating code..."):

            # System prompt to enforce code-only output
            system_prompt = """
You STRICTLY output ONLY valid Python code.
Do NOT use comments.
Do NOT use docstrings.
Do NOT use triple quotes.
Do NOT explain.
Output code only.
"""

            full_prompt = f"{system_prompt}\nTask: {prompt}\nAnswer:\n"

            # Tokenize
            inputs = tokenizer(full_prompt, return_tensors="pt")

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode
            raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean output
            if "Answer:" in raw:
                raw = raw.split("Answer:")[-1]

            raw = raw.replace('"""', "").replace("'''", "")
            raw = raw.replace("```python", "").replace("```", "")

            # Remove all comment lines
            lines = raw.split("\n")
            clean_lines = [line for line in lines if not line.strip().startswith("#")]
            code = "\n".join(clean_lines).strip()

        # ---------------- Display & Download ----------------
        st.subheader("✅ Generated Code")
        st.code(code, language="python")

        st.download_button(
            label="📥 Download code as .py file",
            data=code,
            file_name="generated_code.py"
        )
