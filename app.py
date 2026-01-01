import streamlit as st
from llama_cpp import Llama
import re

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="StarCoder2 Code Generator", page_icon="💻", layout="centered")

st.title("💻 StarCoder2 Code Generator")
st.caption("Outputs ONLY executable Python code. No comments, docstrings, or explanations.")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return Llama.from_pretrained(
        repo_id="bigcode/starcoder2",
        filename="starcoder2-q4_k_m.gguf",  # your downloaded GGUF file
        n_ctx=8192
    )

with st.spinner("Loading StarCoder2 model..."):
    llm = load_model()

# ---------------- User Input ----------------
prompt = st.text_area(
    "📝 Enter your coding task",
    placeholder="Example: Write a Python function to check if a number is prime",
    height=140
)

temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.0, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 50, 2000, 512, 50)

# ---------------- Generate Code ----------------
if st.button("🚀 Generate Code"):
    if not prompt.strip():
        st.warning("Please enter a coding task first.")
    else:
        with st.spinner("Generating code..."):

            system_prompt = """
You STRICTLY output ONLY valid Python code.
Do NOT use comments.
Do NOT use docstrings.
Do NOT use triple quotes.
Do NOT explain.
Output code only.
"""

            full_prompt = f"{system_prompt}\nTask: {prompt}\nAnswer:\n"

            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )

            raw = response["choices"][0]["message"]["content"]

            # --- Clean output ---
            if "Answer:" in raw:
                raw = raw.split("Answer:")[-1]

            raw = raw.replace('"""', "").replace("'''", "")
            raw = raw.replace("```python", "").replace("```", "")

            # Remove lines that start with #
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
