from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm =  HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result=model.invoke("""
Explain in detail:
What is the capital of India?
Give historical background, geography, political importance, and tourist attractions.
""")
print(result.content)
