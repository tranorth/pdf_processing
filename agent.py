# agent.py

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI

from tools.pdf_tools import PDFTableExtractorTool, WebSearchTool, CSVFinalizerTool

def create_pdf_agent():
    """Builds and returns the PDF processing agent."""
    llm = ChatVertexAI(model='gemini-2.5-pro', temperature=0)

    tools = [
        PDFTableExtractorTool(),
        WebSearchTool(), # <-- The new, simpler search tool
        CSVFinalizerTool(llm=llm),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a powerful PDF processing assistant. Your goal is to find the correct table and extract its data."
            "You must follow this thought process:"
            "1. First, use the `extract_all_tables_from_pdf` tool to get a list of all tables from the PDF."
            "2. Identify the main geographic area of the report (e.g., 'Austin' from the filename)."
            "3. For each table, look at the first few values in its first column. Use the `web_search` tool to check if these values are actual submarkets, counties, or neighborhoods within the main geographic area."
            "4. **IMPORTANT**: Do NOT use the web_search tool for generic directional terms like 'CBD', 'Central Business District', 'North', 'South', 'Overall', 'Southwest', or 'Total', etc.. Only search for specific, named places."
            "5. After your research, you will be confident which table is the correct one. Once you are sure, take that single correct table and use the `format_data_to_final_csv` tool to get the final result."
        )),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=15,
        handle_parsing_errors=True,
    )
    return executor