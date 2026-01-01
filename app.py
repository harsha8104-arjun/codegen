import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

st.set_page_config(page_title="AI Code Generator", page_icon="💻")

st.title("💻 AI Code Generator")
st.caption("Outputs ONLY executable code. No comments, docstrings, or explanations.")

@st.cache_resource
def load_model():
    model_name = "Salesforce/codegen-350M-mono"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

prompt = st.text_area(
    "📝 Enter your coding task",
    placeholder="Example: Write a Python function to check if a number is prime"
)

if st.button("🚀 Generate Code"):
    if not prompt.strip():
        st.warning("Please enter a task first.")
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

            inputs = tokenizer(full_prompt, return_tensors="pt")

            output = model.generate(
                **inputs,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,         # deterministic -> fewer broken snippets
                temperature=0.0,
                repetition_penalty=1.05
            )

            raw = tokenizer.decode(output[0], skip_special_tokens=True)

            # --- Clean output ---
            # Remove everything before "Answer:"
            if "Answer:" in raw:
                raw = raw.split("Answer:")[-1]

            # Remove triple quotes if any left
            raw = raw.replace('"""', "").replace("'''", "")

            # Remove markdown fences if present
            raw = raw.replace("```python", "").replace("```", "")

            # Trim leading/trailing whitespace
            code = raw.strip()

        st.subheader("✅ Generated Code")
        st.code(code, language="python")
