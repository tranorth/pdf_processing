#agent.py

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI

# Import your tool classes
from tools.pdf_tools import TableFinderTool, FormatAndValidateTool

def create_pdf_agent():
    """Builds and returns the PDF processing agent."""
    llm = ChatVertexAI(
    model='gemini-2.5-pro',
    temperature=0.7,
    max_output_tokens=2048,
)

    # Update the tool list
    tools = [
        TableFinderTool(llm=llm),
        FormatAndValidateTool(llm=llm), # Use the new combined tool
    ]

    # Update the system prompt to a simpler 2-step process
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a powerful PDF processing assistant. Your goal is to extract and validate industrial real estate data."
            "You must follow these steps in order:"
            "1. Use the `find_pdf_submarket_tables` tool to get the raw text of the relevant table from the PDF."
            "2. Take the raw text from step 1 and use the `format_and_validate_table` tool to get the final, validated result."
        )),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # ... the rest of the file is the same
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
    )
    return executor