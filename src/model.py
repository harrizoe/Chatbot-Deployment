from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
import torch


def load_embedding_model(model_name="intfloat/multilingual-e5-base"):
    """
    Load a HuggingFace embedding model (multilingual-e5-base).
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def load_reranker_model(model_name="BAAI/bge-reranker-base", device=0):
    """
    Load a CrossEncoder reranker model.
    """
    return CrossEncoder(model_name, device=device)


def load_phi_pipeline(model_name="microsoft/Phi-3.5-mini-instruct"):
    """
    Load the Phi-3.5 Mini model as a text generation pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "30GB"},
        use_cache=True
    )
    phi_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=phi_pipeline)

