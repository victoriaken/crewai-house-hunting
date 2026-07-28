import os
from dotenv import load_dotenv
from crewai import LLM, Agent, Crew, Process, Task

# ------------------------------------------------------------------
# ENVIRONMENT & MODEL CONFIGURATION
# ------------------------------------------------------------------

# Disable telemetry noise in the terminal
os.environ["OTEL_SDK_DISABLED"] = "true"

# Load environment variables from the .env file
load_dotenv()

# Initialize Groq LLM via CrewAI's native LLM interface
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ------------------------------------------------------------------
# AGENTS DEFINITION
# ------------------------------------------------------------------

property_researcher = Agent(
    role="Real Estate Sourcing Specialist",
    goal="Discover current property listings that strictly match user criteria such as location, budget, and amenities.",
    backstory=(
        "You are an expert real estate researcher with an eagle eye for hidden market gems. "
        "You excel at sifting through listings to find options that align precisely with specific preferences."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

market_analyst = Agent(
    role="Real Estate Analyst & Valuer",
    goal="Analyze identified properties for price fairness, neighborhood quality, potential red flags, and long-term value.",
    backstory=(
        "With a background in urban planning and financial analysis, you evaluate the true value "
        "of a home beyond its pictures. You look at price-per-square-foot, neighborhood pros/cons, and commuting factors."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

property_advisor = Agent(
    role="Lead House Hunting Consultant",
    goal="Synthesize research and analysis into a polished, actionable recommendation report for the client.",
    backstory=(
        "You are a trusted advisor known for helping clients make confident real estate decisions. "
        "You weigh trade-offs clear-headedly and present key findings in an easy-to-read summary."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# ------------------------------------------------------------------
# TASKS DEFINITION
# ------------------------------------------------------------------

research_task = Task(
    description=(
        "Search for properties based on these criteria:\n"
        "- Location: {location}\n"
        "- Maximum Budget: {budget}\n"
        "- Desired Features: {preferred_features}\n\n"
        "Identify at least 3 candidate properties with key details (price, specs, location, key amenities)."
    ),
    expected_output="A structured list of 3 candidate properties including price, key specs, address/location, and primary features.",
    agent=property_researcher,
)

analysis_task = Task(
    description=(
        "Review the candidate properties provided by the Property Researcher.\n"
        "For each property:\n"
        "1. Assess price fairness against general market norms in {location}.\n"
        "2. Evaluate neighborhood suitability, pros, and potential drawbacks.\n"
        "3. Identify key trade-offs between the properties."
    ),
    expected_output="A detailed comparative analysis highlighting market value, neighborhood pros/cons, and trade-offs for each listing.",
    agent=market_analyst,
)

recommendation_task = Task(
    description=(
        "Compile the findings from the research and analysis tasks into a final House Hunting Summary Report.\n"
        "Structure the report with:\n"
        "- Executive Summary\n"
        "- Ranked Property Recommendations (Top Pick #1, #2, #3)\n"
        "- Detailed Breakdown per property (Pros, Cons, Value Assessment)\n"
        "- Suggested Next Steps (questions to ask the landlord/seller during a viewing)."
    ),
    expected_output="A comprehensive Markdown report summarizing top property picks, trade-offs, and actionable viewing advice.",
    agent=property_advisor,
)

# ------------------------------------------------------------------
# CREW EXECUTION
# ------------------------------------------------------------------

def main():
    house_hunting_crew = Crew(
        agents=[property_researcher, market_analyst, property_advisor],
        tasks=[research_task, analysis_task, recommendation_task],
        process=Process.sequential,
        verbose=True
    )

    # Define input parameters
    search_inputs = {
        "location": "Nairobi, Kenya (focusing on Westlands, Kilimani, or Lavington)",
        "budget": "$1,200/month (or equivalent KES)",
        "preferred_features": "2 bedrooms, good security, reliable high-speed internet capability, balcony, and proximity to grocery stores.",
    }

    # Run the crew
    result = house_hunting_crew.kickoff(inputs=search_inputs)

    print("\n\n########################")
    print("## HOUSE HUNTING REPORT ##")
    print("########################\n")
    print(result.raw)

if __name__ == "__main__":
    main()