import os
import streamlit as st
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_core.tools import BaseTool, Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import YFinanceAPIWrapper
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory

class WebSearchTool(BaseTool):
    """Custom web search tool using DuckDuckGo Search API Wrapper"""
    name = "web_search_tool"
    description = "Performs web searches to gather current information"

    def __init__(self):
        super().__init__()
        self.search = DuckDuckGoSearchAPIWrapper(max_results=5)

    def _run(self, query: str) -> str:
        try:
            results = self.search.run(query)
            return results
        except Exception as e:
            return f"Search error: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)

class FinancialAnalysisTool(BaseTool):
    """Comprehensive financial analysis tool"""
    name = "financial_analysis_tool"
    description = "Provides in-depth financial analysis for stocks and companies"

    def __init__(self):
        super().__init__()
        self.yf = YFinanceAPIWrapper()

    def _run(self, ticker: str) -> str:
        try:
            # Comprehensive financial data gathering
            stock_info = self.yf.get_company_overview(ticker)
            stock_price = self.yf.get_price(ticker)
            
            financial_summary = f"""
📊 Financial Analysis for {ticker}:
-----------------------------
🔹 Company Name: {stock_info.get('longName', 'N/A')}
🔹 Current Price: ${stock_price:.2f}
🔹 Sector: {stock_info.get('sector', 'N/A')}
🔹 Market Cap: ${stock_info.get('marketCap', 'N/A'):,}
🔹 P/E Ratio: {stock_info.get('trailingPE', 'N/A')}
🔹 Dividend Yield: {stock_info.get('dividendYield', 'N/A') * 100:.2f}%
🔹 52-Week High: ${stock_info.get('fiftyTwoWeekHigh', 'N/A'):.2f}
🔹 52-Week Low: ${stock_info.get('fiftyTwoWeekLow', 'N/A'):.2f}
"""
            return financial_summary
        except Exception as e:
            return f"Error retrieving financial data: {str(e)}"

    async def _arun(self, ticker: str) -> str:
        return self._run(ticker)

def create_financial_agent(groq_api_key: str):
    """Create a comprehensive financial analysis agent"""
    # Initialize tools
    web_search_tool = Tool(
        name="web_search",
        func=WebSearchTool()._run,
        description="Useful for searching the web for current information"
    )
    financial_tool = Tool(
        name="financial_analysis",
        func=FinancialAnalysisTool()._run,
        description="Provides detailed financial analysis for stocks"
    )

    # Initialize Groq Language Model
    llm = ChatGroq(
        temperature=0.2, 
        model_name="llama3-8b-8192", 
        groq_api_key=groq_api_key
    )

    # Create memory for conversation context
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    # Define system prompt
    system_prompt = """You are an advanced AI financial analyst assistant. 
    Your goal is to provide comprehensive, data-driven financial insights.
    
    Key Guidelines:
    - Use web search and financial tools to gather accurate information
    - Provide clear, concise financial summaries
    - Offer balanced, objective perspectives
    - Explain complex financial concepts simply
    - Recommend further research
    - Never provide direct investment advice"""

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Create agent with tools
    agent = create_tool_calling_agent(
        llm, 
        [web_search_tool, financial_tool], 
        prompt
    )

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=[web_search_tool, financial_tool], 
        verbose=True,
        memory=memory,
        max_iterations=5
    )

    return agent_executor

def streamlit_financial_assistant():
    """Streamlit app for financial analysis"""
    st.set_page_config(
        page_title="AI Financial Analyst", 
        page_icon="💹", 
        layout="wide"
    )

    st.title("💹 AI Financial Analyst")
    st.sidebar.header("Configuration")
    
    # API Key Input
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    
    if not groq_api_key:
        st.warning("Please enter your Groq API Key")
        return

    # Initialize agent
    try:
        financial_agent = create_financial_agent(groq_api_key)
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI Financial Analyst. What would you like to know about stocks or financial markets?"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask about stocks, financial analysis, or market trends"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            try:
                # Stream the response
                full_response = ""
                for chunk in financial_agent.stream({"input": prompt}):
                    full_response += chunk.get('output', '')
                    response_placeholder.markdown(full_response)

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response
                })

            except Exception as e:
                st.error(f"Error processing request: {e}")

def main():
    streamlit_financial_assistant()

if __name__ == "__main__":
    main()