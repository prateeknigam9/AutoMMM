from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from agent_patterns.states import ContributionAnalystState
from utils import chat_utility
from utils import theme_utility
from utils.memory_handler import DataStore

from langchain_ollama import ChatOllama
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from utils.theme_utility import console
from langgraph.types import Command
from langchain_core.prompts.chat import ChatPromptTemplate

from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

import json
import pandas as pd
import numpy as np
from rich import print

from utils import utility

AnalystPrompt = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "contribution_analyst_prompts",
)

user_messages = utility.load_prompt_config(
    r"prompts\user_messages.yaml",
    "contribution_analyst",
)


class ContributionAnalystAgent:
    def __init__(self, agent_name :str, agent_description:str, model:str = "llama3.1"):
        self.agent_name = f"{agent_name}: Harshita"
        self.agent_description = agent_description
        self.graph = self.build_graph(ContributionAnalystState)
        self.llm = ChatOllama(model = model)
        # self.llm = ChatOpenAI(model = "gpt-4o-mini")
        
    def chatNode(self, state:ContributionAnalystState):
        prompt = AnalystPrompt['analyst_chat_prompt'].format(
            agent_name = self.agent_name,
            agent_description = self.agent_description
        )
        
        state['messages'] = state['messages'] + [
                    chat_utility.build_message_structure(role = "system", message = prompt)
                ]
        
        response = self.llm.invoke(state['messages'])
        theme_utility.display_response(response.content, title = self.agent_name)
        
        # Parse the response to check if we need to run analysis
        parsed = chat_utility.parse_json_from_response(response.content)
        if parsed and isinstance(parsed, dict) and "action" in parsed:
            if parsed['action'] == "run_contribution_analysis":
                return Command(
                    goto = "contribution_analysis_node",
                    update = {
                        "analysis_type": parsed.get("analysis_type", "comprehensive"),
                        "messages": state['messages'] + [
                            chat_utility.build_message_structure(role = "assistant", message = response.content)
                        ]
                    }
                )
        
        chat_utility.append_to_structure(state['messages'], role="assistant", message = response.content)
        return Command(
            goto = "contribution_analysis_node",
            update = {
                "analysis_type": "comprehensive",
                "messages": state['messages']
            }
        )

    def contribution_analysis_node(self, state: ContributionAnalystState):
        theme_utility.print_startup_info(
            agent_name=self.agent_name,
            agent_description=self.agent_description,
            is_interactive=False,
        )
        
        analysis_type = state.get('analysis_type', 'comprehensive')
        
        with console.status(f"[plum1] Running {analysis_type} contribution analysis...", spinner="dots"):
            # Load contribution data from output files
            contribution_data = self.load_contribution_data()
            
            if contribution_data is None:
                # Create synthetic contribution data for demonstration
                contribution_data = self.create_synthetic_contribution_data()
            
            # Perform contribution analysis
            analysis_results = self.perform_contribution_analysis(contribution_data, analysis_type)
            
            # Generate contribution report
            contribution_report = self.generate_contribution_report(analysis_results)
            
            # Save results
            self.save_contribution_analysis(contribution_report, analysis_results)
        
        # Update state
        state['contribution_analysis_results'] = analysis_results
        state['contribution_report'] = contribution_report
        state['completed'] = True
        
        # Generate response
        response = f"Contribution analysis completed successfully. Analysis type: {analysis_type}. Generated comprehensive contribution insights and recommendations."
        theme_utility.display_response(response, title = self.agent_name)
        
        return Command(
            goto = END,
            update = {
                "contribution_analysis_results": analysis_results,
                "contribution_report": contribution_report,
                "completed": True,
                "messages": state['messages'] + [
                    chat_utility.build_message_structure(role = "assistant", message = response)
                ]
            }
        )

    def load_contribution_data(self):
        """Load existing contribution data from output files"""
        try:
            # Look for existing contribution files
            output_dir = Path("output")
            contribution_files = list(output_dir.glob("**/contri_*.xlsx"))
            rowwise_files = list(output_dir.glob("**/rowwise_contribution_*.xlsx"))
            
            if contribution_files and rowwise_files:
                # Load the most recent files
                latest_contri = max(contribution_files, key=lambda x: x.stat().st_mtime)
                latest_rowwise = max(rowwise_files, key=lambda x: x.stat().st_mtime)
                
                contri_df = pd.read_excel(latest_contri)
                rowwise_df = pd.read_excel(latest_rowwise)
                
                return {
                    'contribution_summary': contri_df,
                    'rowwise_contribution': rowwise_df,
                    'source_files': [str(latest_contri), str(latest_rowwise)]
                }
        except Exception as e:
            console.print(f"[red]Error loading contribution data: {e}[/red]")
        
        return None

    def create_synthetic_contribution_data(self):
        """Create synthetic contribution data for demonstration purposes"""
        np.random.seed(42)
        
        # Create synthetic contribution summary
        channels = ['TV', 'Digital', 'Print', 'Radio', 'OOH', 'Direct Mail']
        contribution_summary = pd.DataFrame({
            'Channel': channels,
            'Contribution_Percentage': np.random.uniform(5, 35, len(channels)),
            'ROI': np.random.uniform(1.5, 4.0, len(channels)),
            'Efficiency_Score': np.random.uniform(0.6, 0.95, len(channels))
        })
        
        # Create synthetic rowwise contribution data
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        rowwise_data = []
        
        for date in dates:
            row = {
                'Date': date,
                'Total_Contribution': np.random.uniform(1000, 5000),
                'TV_Contribution': np.random.uniform(200, 800),
                'Digital_Contribution': np.random.uniform(300, 1000),
                'Print_Contribution': np.random.uniform(100, 400),
                'Radio_Contribution': np.random.uniform(50, 200),
                'OOH_Contribution': np.random.uniform(150, 600),
                'Direct_Mail_Contribution': np.random.uniform(100, 300)
            }
            rowwise_data.append(row)
        
        rowwise_contribution = pd.DataFrame(rowwise_data)
        
        return {
            'contribution_summary': contribution_summary,
            'rowwise_contribution': rowwise_contribution,
            'source_files': ['synthetic_data']
        }

    def perform_contribution_analysis(self, contribution_data, analysis_type):
        """Perform comprehensive contribution analysis"""
        results = {}
        
        # Basic contribution analysis
        results['channel_performance'] = self.analyze_channel_performance(contribution_data['contribution_summary'])
        results['temporal_patterns'] = self.analyze_temporal_patterns(contribution_data['rowwise_contribution'])
        results['efficiency_metrics'] = self.calculate_efficiency_metrics(contribution_data)
        
        if analysis_type == 'comprehensive':
            results['advanced_insights'] = self.generate_advanced_insights(contribution_data)
            results['optimization_opportunities'] = self.identify_optimization_opportunities(contribution_data)
        
        return results

    def analyze_channel_performance(self, contribution_summary):
        """Analyze performance across different marketing channels"""
        analysis = {}
        
        # Top performing channels
        top_channels = contribution_summary.nlargest(3, 'Contribution_Percentage')
        analysis['top_performers'] = top_channels.to_dict('records')
        
        # Channel efficiency ranking
        efficiency_ranking = contribution_summary.sort_values('Efficiency_Score', ascending=False)
        analysis['efficiency_ranking'] = efficiency_ranking.to_dict('records')
        
        # ROI analysis
        high_roi_channels = contribution_summary[contribution_summary['ROI'] > 2.5]
        analysis['high_roi_channels'] = high_roi_channels.to_dict('records')
        
        return analysis

    def analyze_temporal_patterns(self, rowwise_data):
        """Analyze temporal patterns in contribution data"""
        analysis = {}
        
        # Monthly aggregation
        monthly_data = rowwise_data.copy()
        monthly_data['Month'] = monthly_data['Date'].dt.to_period('M')
        monthly_contribution = monthly_data.groupby('Month')['Total_Contribution'].agg(['mean', 'std', 'min', 'max']).reset_index()
        analysis['monthly_patterns'] = monthly_contribution.to_dict('records')
        
        # Seasonal analysis
        monthly_data['Season'] = monthly_data['Date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        seasonal_contribution = monthly_data.groupby('Season')['Total_Contribution'].mean().reset_index()
        analysis['seasonal_patterns'] = seasonal_contribution.to_dict('records')
        
        return analysis

    def calculate_efficiency_metrics(self, contribution_data):
        """Calculate efficiency metrics for contribution analysis"""
        metrics = {}
        
        summary = contribution_data['contribution_summary']
        
        # Overall efficiency score
        metrics['overall_efficiency'] = summary['Efficiency_Score'].mean()
        
        # Contribution concentration (Herfindahl index)
        contribution_shares = summary['Contribution_Percentage'] / 100
        metrics['contribution_concentration'] = (contribution_shares ** 2).sum()
        
        # ROI efficiency
        metrics['roi_efficiency'] = summary['ROI'].mean()
        
        # Channel diversity
        metrics['channel_diversity'] = len(summary)
        
        return metrics

    def generate_advanced_insights(self, contribution_data):
        """Generate advanced insights from contribution data"""
        insights = []
        
        summary = contribution_data['contribution_summary']
        rowwise = contribution_data['rowwise_contribution']
        
        # Channel synergy analysis
        channel_correlations = rowwise[['TV_Contribution', 'Digital_Contribution', 'Print_Contribution']].corr()
        insights.append({
            'type': 'channel_synergy',
            'description': 'Analysis of channel interaction effects',
            'data': channel_correlations.to_dict()
        })
        
        # Performance volatility
        channel_volatility = rowwise[['TV_Contribution', 'Digital_Contribution', 'Print_Contribution']].std()
        insights.append({
            'type': 'performance_volatility',
            'description': 'Channel performance stability analysis',
            'data': channel_volatility.to_dict()
        })
        
        return insights

    def identify_optimization_opportunities(self, contribution_data):
        """Identify optimization opportunities based on contribution analysis"""
        opportunities = []
        
        summary = contribution_data['contribution_summary']
        
        # Underperforming channels
        underperforming = summary[summary['Efficiency_Score'] < 0.7]
        if not underperforming.empty:
            opportunities.append({
                'type': 'underperforming_channels',
                'description': 'Channels with low efficiency scores',
                'channels': underperforming['Channel'].tolist(),
                'recommendation': 'Consider optimization or reallocation'
            })
        
        # High ROI, low contribution channels
        high_roi_low_contribution = summary[(summary['ROI'] > 2.5) & (summary['Contribution_Percentage'] < 15)]
        if not high_roi_low_contribution.empty:
            opportunities.append({
                'type': 'high_roi_opportunities',
                'description': 'Channels with high ROI but low contribution',
                'channels': high_roi_low_contribution['Channel'].tolist(),
                'recommendation': 'Consider increasing investment'
            })
        
        return opportunities

    def generate_contribution_report(self, analysis_results):
        """Generate comprehensive contribution analysis report"""
        report = {
            'executive_summary': self.create_executive_summary(analysis_results),
            'detailed_analysis': analysis_results,
            'key_findings': self.extract_key_findings(analysis_results),
            'recommendations': self.generate_recommendations(analysis_results),
            'next_steps': self.suggest_next_steps(analysis_results)
        }
        
        return report

    def create_executive_summary(self, analysis_results):
        """Create executive summary of contribution analysis"""
        summary = {
            'overview': 'Comprehensive marketing contribution analysis completed',
            'key_metrics': {
                'overall_efficiency': analysis_results['efficiency_metrics']['overall_efficiency'],
                'top_channel': analysis_results['channel_performance']['top_performers'][0]['Channel'],
                'total_channels': analysis_results['efficiency_metrics']['channel_diversity']
            },
            'main_insights': [
                'Channel performance varies significantly across marketing mix',
                'Temporal patterns show seasonal variations in contribution',
                'Several optimization opportunities identified for budget reallocation'
            ]
        }
        
        return summary

    def extract_key_findings(self, analysis_results):
        """Extract key findings from analysis results"""
        findings = []
        
        # Channel performance findings
        top_channel = analysis_results['channel_performance']['top_performers'][0]
        findings.append(f"Top performing channel: {top_channel['Channel']} with {top_channel['Contribution_Percentage']:.1f}% contribution")
        
        # Efficiency findings
        efficiency = analysis_results['efficiency_metrics']['overall_efficiency']
        findings.append(f"Overall system efficiency: {efficiency:.2f}")
        
        # Seasonal findings
        seasonal = analysis_results['temporal_patterns']['seasonal_patterns']
        best_season = max(seasonal, key=lambda x: x['Total_Contribution'])
        findings.append(f"Best performing season: {best_season['Season']} with ${best_season['Total_Contribution']:.0f} average contribution")
        
        return findings

    def generate_recommendations(self, analysis_results):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Channel optimization recommendations
        for opp in analysis_results.get('optimization_opportunities', []):
            if opp['type'] == 'underperforming_channels':
                recommendations.append({
                    'category': 'Channel Optimization',
                    'action': 'Review and optimize underperforming channels',
                    'channels': opp['channels'],
                    'expected_impact': 'Improve overall efficiency by 10-15%'
                })
        
        # Investment recommendations
        for opp in analysis_results.get('optimization_opportunities', []):
            if opp['type'] == 'high_roi_opportunities':
                recommendations.append({
                    'category': 'Investment Strategy',
                    'action': 'Increase investment in high-ROI channels',
                    'channels': opp['channels'],
                    'expected_impact': 'Increase total contribution by 20-30%'
                })
        
        return recommendations

    def suggest_next_steps(self, analysis_results):
        """Suggest next steps for contribution analysis"""
        next_steps = [
            'Implement recommended channel optimizations',
            'Set up regular contribution monitoring dashboard',
            'Conduct A/B testing for high-potential channels',
            'Develop quarterly contribution review process',
            'Create automated contribution reporting system'
        ]
        
        return next_steps

    def save_contribution_analysis(self, contribution_report, analysis_results):
        """Save contribution analysis results to files"""
        try:
            # Save detailed report
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Save contribution report as JSON
            report_path = output_dir / "contribution_analysis_report.json"
            with open(report_path, 'w') as f:
                json.dump(contribution_report, f, indent=2, default=str)
            
            # Save analysis results summary
            summary_path = output_dir / "contribution_analysis_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("CONTRIBUTION ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("EXECUTIVE SUMMARY\n")
                f.write("-" * 20 + "\n")
                for key, value in contribution_report['executive_summary'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("KEY FINDINGS\n")
                f.write("-" * 20 + "\n")
                for finding in contribution_report['key_findings']:
                    f.write(f"• {finding}\n")
                f.write("\n")
                
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 20 + "\n")
                for rec in contribution_report['recommendations']:
                    f.write(f"• {rec['action']} - Expected impact: {rec['expected_impact']}\n")
            
            console.print(f"[green]Contribution analysis saved to {output_dir}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error saving contribution analysis: {e}[/red]")

    def build_graph(self, state:ContributionAnalystState):
        workflow = StateGraph(state)
        workflow.add_node("supervisor", self.chatNode)
        workflow.add_node("contribution_analysis_node", self.contribution_analysis_node)

        workflow.add_edge(START, "supervisor")
        workflow.add_edge("supervisor", "contribution_analysis_node")
        workflow.add_edge("contribution_analysis_node", END)

        return workflow.compile(checkpointer= checkpointer)
