# main.py
import os
import sys
from pathlib import Path
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from autommm.config.process_configuration import process_config
from autommm.config import configuration

from autommm.src.eda_agent import eda_workflow

config = process_config(configuration)

eda_report_path = config['eda_report_path']

if __name__ == "__main__":

    print("Running agent.py")
    response = eda_workflow.workflow.invoke({"input" :"generate the report"})
    print(response)
    # Save the output to a Markdown file
    # output_path = Path(eda_report_path)
    # output_path.write_text(f"# LLM Response\n\n{response['formatted_report'].content}", encoding="utf-8")



# --- 4. Run the Graph ---

# if __name__ == "__main__":
#     # Your example text context (the AutoMMM thesis proposal)
#     sample_text_context = """
#     This thesis proposes AutoMMM, an AI-powered system to analyze marketing data, like ad spend and sales, automatically. Using Python, LangChain, and LangGraph, AutoMMM will use multiple AI agents to process data, run models, and suggest better marketing strategies. It will generate synthetic data to test the system and validate its performance. This report explains the problem, goals, methods, expected results, and a two-month plan.

#     Market Mix Modeling (MMM) helps businesses see how marketing efforts, like ads and promotions, drive sales. It guides smart budget decisions to maximize return on investment (ROI). Traditional MMM is slow and needs a lot of manual work, which struggles in today’s fast digital marketing world. Artificial Intelligence (AI) can improve MMM by automating tasks and finding data patterns. However, there are few systems that fully automate MMM using multiple AI agents. AutoMMM will solve this by using Python, LangChain, and LangGraph to analyze ad spend, sales, and external factors, delivering clear business advice. The problem is the lack of fully automated MMM systems with multi-agent collaboration. The goals are to: Build an automated MMM system with AI agents. Test it using synthetic data and validate performance. Add new ideas to AI in marketing analytics.

#     Understanding Market Mix Modeling (MMM) is a method that uses statistics to understand how different marketing actions—like advertising, promotions, and pricing—affect product sales. It often uses regression models and includes features like: Adstock: The effect of ads continues over time. Saturation: Spending more gives less extra benefit after a point. Seasonality: Certain times of year affect sales more than others. Modern MMM tools like Adobe Mix Modeler and NextBrain AI use machine learning to make this process faster. But they still need people to clean the data, choose features, and understand the results.

#     What Are Multi-Agent Systems (MAS)? A Multi-Agent System (MAS) is a group of AI agents that work together. Each agent can do its own task but also talk to other agents. MAS systems are good for handling complex problems where many things happen at once. MAS Structures: Hierarchical: One supervisor agent manages others. Decentralized: All agents work independently but still coordinate. These designs help divide work and manage big projects, which fits well with a system like AutoMMM. MAS has not been used directly in MMM yet, but it has been helpful in related areas: Marketing simulations: MAS is used to model how customers behave and respond to marketing. Advertising tools: Platforms like AdSim use MAS to test how ad spend affects ROI by simulating customer behavior. Social campaigns: MAS helps understand how ideas or products spread in networks. MAS has also been used in: Supply chains: Simulating suppliers, factories, and deliveries. City systems: Modeling how people move and interact in cities. These examples show how MAS can help in large, complex systems like MMM. Tools for Building MAS: langchain, openAI, CrewAI, Agno.

#     Key Insights: MMM still relies heavily on regression and needs human support. MAS is used in marketing and other fields but not yet in MMM. MAS features—like automation and teamwork—match well with AutoMMM’s design. This shows that AutoMMM is filling a real gap by combining MMM and MAS in a new, automated way.

#     Methodology: AutoMMM is designed as a multi-agent system. It uses LangChain and LangGraph to let agents talk to each other and work together. The system is managed by a special agent called the Supervisor Agent. This agent controls the overall workflow and makes sure each step is done in the correct order. Each agent has a clear job. They pass results to the next agent in a smart way, depending on the outcome. The Supervisor Agent watches the whole process and can tell an agent to repeat a task or fix a problem if needed.

#     Data Team: Data Processing Agent: Will run Python scripts for exploratory data analysis (EDA) on ad spend, sales, and external factors, creating Excel reports with charts and notes on data issues like missing values or outliers. Insight Agent: Will summarize key performance indicators (KPIs) and data stories using LangChain to guide modeling. Modeling Team: Model Running Agent: Will run a simple regression model (MCP) to predict sales, saving model summaries and graphs (PNG files). Model Tuning Agent: Will adjust hyperparameters using insights from the Data Team to improve accuracy. Contribution Agent: Will run run_contributions.py to calculate marketing contributions using the Shapley method, producing an Excel file. Final Suggestions: Will turn results into clear business advice, like optimizing ad spend for better ROI.

#     How Agents Interact: The system works in steps: 1. The Supervisor gives data to the Data Processing Agent. 2. After cleaning, results go to the Insight Agent. 3. Then the model is run. If it is not good, the Supervisor calls the Tuning Agent. 4. Once the model is good, the Contribution Agent finds which ads worked best. 5. Finally, the Recommendation Agent gives business advice. If any step fails or needs improvement, the Supervisor can go back and repeat it. This makes the system flexible and reliable. Some tasks can run at the same time, like testing different models. The system can also reuse previous results, making it fast and efficient. This setup follows a popular design called “orchestrator-worker,” where one main agent manages many workers.

#     Synthetic Data Generation: Synthetic data helps test our model in a safe and controlled way. It acts as a “gold standard” because we know the correct answers ahead of time. This is important in marketing mix modeling (MMM), where collecting real-world causal data (like from randomized experiments) is expensive and slow. Instead of running real tests, we simulate realistic sales data using known values. This lets us check if the model gives back the correct results—just like how tests work in science. To test AutoMMM, I will generate synthetic data that mimics real marketing data. The process will include: Data Structure: Create 104 weeks (2 years) of weekly data for ad spend (e.g., TV, digital), sales, and external factors (e.g., holidays, seasonality). Realistic Patterns: Add positive correlations between ad spend and sales, diminishing returns (saturation), and carryover effects (adstock) using decaying functions and logistic transformations. Noise and Outliers: Include Gaussian noise and occasional outliers to simulate real-world data challenges. Ground Truth: Set known coefficients (e.g., 0.3 for TV, 0.4 for digital) to compare with model outputs. Config File: Use a YAML config to define parameters like adstock decay, saturation, and noise levels for reproducibility. This synthetic setup helps in three ways: Model Testing: We can check if the model correctly recovers the true input values (coefficients). Fast Feedback: We avoid the delay and cost of real-world experiments. Safe and Scalable: No real user data means no privacy issues.

#     Model Validation: To check AutoMMM’s performance, I will: Compare Coefficients: Compare model-estimated coefficients to ground truth coefficients from the synthetic data to measure accuracy. Evaluate Metrics: Use metrics like Mean Absolute Error (MAE) and R-squared to assess prediction quality. Test Robustness: Validate the model on noisy data and outliers to ensure it handles real-world challenges. ROAS Analysis: Check if the Contribution Agent’s outputs align with expected return on ad spend (ROAS) based on ground truth. The Supervisor Agent will ensure all agents work together to validate results.

#     Implementation Details: Tools: Python for coding, LangChain for agent logic, LangGraph for workflows. Modeling: Use a simple regression model (MCP) for sales prediction. Contributions: Calculate using run_contributions.py with the Shapley method. Data Generation: Create synthetic data with realistic MMM patterns.

#     Expected Results: Academic Results: Develop a new way to automate MMM with AI agents. Create a framework for multi-agent systems in marketing analytics. Practical Results: Build a tool that automates MMM, saving time. Provide clear advice to improve marketing and boost profits.

#     Timeline: The project will take two months: Week 1: Study past work and design system. Weeks 2-3: Build Data Team and synthetic data generator. Weeks 4-5: Develop Modeling Team. Week 6: Create Contribution Agent. Week 7: Test system and validate performance. Week 8: Finalize testing and write report.
#     """

#     # Invoke the graph with the initial state
#     print("\n--- Invoking the InfographicGenius workflow ---")
#     final_state = infograph_bot.infograph_workflow.invoke({"raw_text": sample_text_context})

#     # Retrieve the generated HTML from the final state
#     generated_infographic_html = final_state.get("generated_html")
#     error = final_state.get("error_message")
#     print("html: " , generated_infographic_html)

#     # Retrieve the generated HTML from the final state
#     generated_infographic_html = final_state.get("generated_html")
#     error = final_state.get("error_message")

#     if generated_infographic_html and not error:
#         print("\n--- Infographic HTML Generated Successfully! ---")
#         # Save this HTML to a file and open it in a browser:
#         output_filename = "autommm_infographic_real_agent.html"
#         with open(output_filename, "w") as f:
#             f.write(generated_infographic_html)
#         print(f"\nInfographic saved to '{output_filename}'")
#         print(f"Open '{output_filename}' in your web browser to view the infographic.")
#     elif error:
#         print(f"\n--- Workflow Failed: {error} ---")
#     else:
#         print("\n--- Workflow completed, but no HTML was generated or an error occurred. ---")

#     print("\n--- Full Final State of the Graph ---")
#     # For debugging, you can print the entire final state
#     # print(final_state)
    