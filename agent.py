# agent_factory.py

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI

from tools.pdf_tools import PDFProcessorTool

def create_pdf_agent():
    """Builds and returns the PDF processing agent."""
    llm = ChatVertexAI(model='gemini-2.5-pro', temperature=0)

    tools = [
        PDFProcessorTool(llm=llm),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the available tool to process the user's PDF request."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True,
    )
    return executor