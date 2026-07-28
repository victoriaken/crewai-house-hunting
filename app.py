import os
import streamlit as st
from dotenv import load_dotenv
from crewai import LLM, Agent, Crew, Process, Task

# Load environment variables
load_dotenv()
os.environ["OTEL_SDK_DISABLED"] = "true"

st.set_page_config(page_title="House Hunting AI Agent", page_icon="🏠", layout="wide")

st.title("🏠 AI House Hunting Assistant")
st.markdown("Enter your preferences below to let the multi-agent crew search, analyze, and compile a report.")

# Sidebar for API Key check / Settings
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

# User Input Form
with st.form("house_search_form"):
    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("Target Location", value="Nairobi, Kenya (Westlands, Kilimani, or Lavington)")
        budget = st.text_input("Maximum Budget", value="$1,200/month (or equivalent KES)")
    with col2:
        preferred_features = st.text_area("Preferred Features", value="2 bedrooms, good security, reliable high-speed internet capability, balcony, proximity to grocery stores.")
    
    submit_button = st.form_submit_button("Start Property Search 🚀")

if submit_button:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please provide a valid Groq API Key in the sidebar or `.env` file.")
    else:
        with st.spinner("Agents are analyzing properties... This will take just a few seconds on Groq!"):
            # 1. Initialize Groq via OpenAI-compatible endpoint (No LiteLLM required)
            llm = LLM(
                model="openai/llama-3.3-70b-versatile",
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY")
            )

            # 2. Define Agents
            property_researcher = Agent(
                role="Real Estate Sourcing Specialist",
                goal="Discover property listings matching user criteria.",
                backstory="Expert real estate researcher with an eye for hidden gems.",
                verbose=True,
                allow_delegation=False,
                llm=llm
            )

            market_analyst = Agent(
                role="Real Estate Analyst & Valuer",
                goal="Analyze properties for fair market value, neighborhood quality, and trade-offs.",
                backstory="Urban planning specialist evaluating true market values.",
                verbose=True,
                allow_delegation=False,
                llm=llm
            )

            property_advisor = Agent(
                role="Lead House Hunting Consultant",
                goal="Synthesize findings into an actionable report.",
                backstory="Trusted consultant helping clients make confident decisions.",
                verbose=True,
                allow_delegation=False,
                llm=llm
            )

            # 3. Define Tasks
            research_task = Task(
                description="Search for properties in {location} within budget {budget} with features: {preferred_features}.",
                expected_output="A structured list of 3 candidate properties.",
                agent=property_researcher
            )

            analysis_task = Task(
                description="Review candidate properties, evaluate pricing, and highlight pros/cons for {location}.",
                expected_output="A comparative market analysis of candidate properties.",
                agent=market_analyst
            )

            recommendation_task = Task(
                description="Compile all findings into a polished House Hunting Summary Report with actionable viewing advice.",
                expected_output="A comprehensive Markdown report.",
                agent=property_advisor
            )

            # 4. Execute Crew
            crew = Crew(
                agents=[property_researcher, market_analyst, property_advisor],
                tasks=[research_task, analysis_task, recommendation_task],
                process=Process.sequential
            )

            inputs = {
                "location": location,
                "budget": budget,
                "preferred_features": preferred_features
            }

            result = crew.kickoff(inputs=inputs)

        st.success("Analysis Complete!")
        st.markdown("### 📊 Final House Hunting Report")
        st.markdown(result.raw)