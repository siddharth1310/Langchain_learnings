from langchain_openai import ChatOpenAI

try:
    from langchain_community.memory import ConversationSummaryMemory
except ImportError:
    from langchain.memory import ConversationSummaryMemory

llm = ChatOpenAI(model="gpt-4o-mini")
memory = ConversationSummaryMemory(llm=llm)
print("✅ Memory class loaded successfully:", type(memory))
