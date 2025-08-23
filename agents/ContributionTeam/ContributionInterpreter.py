from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from agent_patterns.states import ContributionInterpreterState
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

InterpreterPrompt = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "contribution_interpreter_prompts",
)

user_messages = utility.load_prompt_config(
    r"prompts\user_messages.yaml",
    "contribution_interpreter",
)


class ContributionInterpreterAgent:
    def __init__(self, agent_name :str, agent_description:str, model:str = "llama3.1"):
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.graph = self.build_graph(ContributionInterpreterState)
        self.llm = ChatOllama(model = model)
        # self.llm = ChatOpenAI(model = "gpt-4o-mini")
        
    def chatNode(self, state:ContributionInterpreterState):
        prompt = InterpreterPrompt['interpreter_chat_prompt'].format(
            agent_name = self.agent_name,
            agent_description = self.agent_description
        )
        
        state['messages'] = state['messages'] + [
                    chat_utility.build_message_structure(role = "system", message = prompt)
                ]
        
        response = self.llm.invoke(state['messages'])
        theme_utility.display_response(response.content, title = self.agent_name)
        
        # Parse the response to check if we need to run interpretation
        parsed = chat_utility.parse_json_from_response(response.content)
        if parsed and isinstance(parsed, dict) and "action" in parsed:
            if parsed['action'] == "run_contribution_interpretation":
                return Command(
                    goto = "contribution_interpretation_node",
                    update = {
                        "interpretation_type": parsed.get("interpretation_type", "comprehensive"),
                        "messages": state['messages'] + [
                            chat_utility.build_message_structure(role = "assistant", message = response.content)
                        ]
                    }
                )
        
        chat_utility.append_to_structure(state['messages'], role="assistant", message = response.content)
        return Command(
            goto = "contribution_interpretation_node",
            update = {
                "interpretation_type": "comprehensive",
                "messages": state['messages']
            }
        )

    def contribution_interpretation_node(self, state: ContributionInterpreterState):
        theme_utility.print_startup_info(
            agent_name=self.agent_name,
            agent_description=self.agent_description,
            is_interactive=False,
        )
        
        interpretation_type = state.get('interpretation_type', 'comprehensive')
        
        with console.status(f"[plum1] Running {interpretation_type} contribution interpretation...", spinner="dots"):
            # Load contribution analysis results
            contribution_data = self.load_contribution_analysis()
            
            if contribution_data is None:
                # Create synthetic interpretation data for demonstration
                contribution_data = self.create_synthetic_interpretation_data()
            
            # Perform contribution interpretation
            interpretation_results = self.perform_contribution_interpretation(contribution_data, interpretation_type)
            
            # Generate business insights
            business_insights = self.generate_business_insights(interpretation_results)
            
            # Generate actionable recommendations
            actionable_recommendations = self.generate_actionable_recommendations(interpretation_results)
            
            # Generate marketing optimization strategies
            marketing_optimization = self.generate_marketing_optimization(interpretation_results)
            
            # Save interpretation results
            self.save_contribution_interpretation(interpretation_results, business_insights, actionable_recommendations, marketing_optimization)
        
        # Update state
        state['contribution_interpretation_results'] = interpretation_results
        state['business_insights'] = business_insights
        state['actionable_recommendations'] = actionable_recommendations
        state['marketing_optimization'] = marketing_optimization
        state['completed'] = True
        
        # Generate response
        response = f"Contribution interpretation completed successfully. Interpretation type: {interpretation_type}. Generated business insights, actionable recommendations, and marketing optimization strategies."
        theme_utility.display_response(response, title = self.agent_name)
        
        return Command(
            goto = END,
            update = {
                "contribution_interpretation_results": interpretation_results,
                "business_insights": business_insights,
                "actionable_recommendations": actionable_recommendations,
                "marketing_optimization": marketing_optimization,
                "completed": True,
                "messages": state['messages'] + [
                    chat_utility.build_message_structure(role = "assistant", message = response)
                ]
            }
        )

    def load_contribution_analysis(self):
        """Load contribution analysis results from files"""
        try:
            # Look for contribution analysis files
            output_dir = Path("output")
            analysis_report = output_dir / "contribution_analysis_report.json"
            analysis_summary = output_dir / "contribution_analysis_summary.txt"
            
            if analysis_report.exists():
                with open(analysis_report, 'r') as f:
                    analysis_data = json.load(f)
                return analysis_data
            elif analysis_summary.exists():
                with open(analysis_summary, 'r') as f:
                    summary_text = f.read()
                return {'summary_text': summary_text}
        except Exception as e:
            console.print(f"[red]Error loading contribution analysis: {e}[/red]")
        
        return None

    def create_synthetic_interpretation_data(self):
        """Create synthetic interpretation data for demonstration purposes"""
        np.random.seed(42)
        
        # Create synthetic contribution analysis results
        interpretation_data = {
            'channel_performance': {
                'top_performers': [
                    {'Channel': 'Digital', 'Contribution_Percentage': 28.5, 'ROI': 3.2, 'Efficiency_Score': 0.89},
                    {'Channel': 'TV', 'Contribution_Percentage': 25.3, 'ROI': 2.8, 'Efficiency_Score': 0.85},
                    {'Channel': 'Print', 'Contribution_Percentage': 18.7, 'ROI': 2.1, 'Efficiency_Score': 0.72}
                ],
                'efficiency_ranking': [
                    {'Channel': 'Digital', 'Efficiency_Score': 0.89},
                    {'Channel': 'TV', 'Efficiency_Score': 0.85},
                    {'Channel': 'OOH', 'Efficiency_Score': 0.78}
                ]
            },
            'temporal_patterns': {
                'seasonal_patterns': [
                    {'Season': 'Spring', 'Total_Contribution': 3200},
                    {'Season': 'Summer', 'Total_Contribution': 2800},
                    {'Season': 'Fall', 'Total_Contribution': 3500},
                    {'Season': 'Winter', 'Total_Contribution': 2400}
                ]
            },
            'efficiency_metrics': {
                'overall_efficiency': 0.78,
                'contribution_concentration': 0.45,
                'roi_efficiency': 2.7,
                'channel_diversity': 6
            },
            'optimization_opportunities': [
                {
                    'type': 'underperforming_channels',
                    'channels': ['Radio', 'Direct Mail'],
                    'recommendation': 'Consider optimization or reallocation'
                },
                {
                    'type': 'high_roi_opportunities',
                    'channels': ['Digital'],
                    'recommendation': 'Consider increasing investment'
                }
            ]
        }
        
        return interpretation_data

    def perform_contribution_interpretation(self, contribution_data, interpretation_type):
        """Perform comprehensive contribution interpretation"""
        results = {}
        
        # Basic interpretation
        results['channel_insights'] = self.interpret_channel_performance(contribution_data)
        results['temporal_insights'] = self.interpret_temporal_patterns(contribution_data)
        results['efficiency_insights'] = self.interpret_efficiency_metrics(contribution_data)
        
        if interpretation_type == 'comprehensive':
            results['strategic_insights'] = self.generate_strategic_insights(contribution_data)
            results['market_context'] = self.analyze_market_context(contribution_data)
            results['competitive_analysis'] = self.perform_competitive_analysis(contribution_data)
        
        return results

    def interpret_channel_performance(self, contribution_data):
        """Interpret channel performance insights"""
        insights = {}
        
        if 'channel_performance' in contribution_data:
            channel_data = contribution_data['channel_performance']
            
            # Top performer insights
            if 'top_performers' in channel_data and channel_data['top_performers']:
                top_channel = channel_data['top_performers'][0]
                insights['top_performer_analysis'] = {
                    'channel': top_channel['Channel'],
                    'strength': f"Strong performance with {top_channel['Contribution_Percentage']:.1f}% contribution",
                    'opportunity': f"High ROI of {top_channel['ROI']:.1f} suggests potential for increased investment",
                    'strategy': "Leverage success factors and consider scaling up"
                }
            
            # Efficiency insights
            if 'efficiency_ranking' in channel_data:
                efficiency_data = channel_data['efficiency_ranking']
                insights['efficiency_analysis'] = {
                    'most_efficient': efficiency_data[0]['Channel'] if efficiency_data else 'N/A',
                    'efficiency_gap': f"Efficiency scores range from {efficiency_data[-1]['Efficiency_Score']:.2f} to {efficiency_data[0]['Efficiency_Score']:.2f}" if len(efficiency_data) > 1 else 'N/A',
                    'optimization_potential': "Significant room for improvement in lower-performing channels"
                }
        
        return insights

    def interpret_temporal_patterns(self, contribution_data):
        """Interpret temporal pattern insights"""
        insights = {}
        
        if 'temporal_patterns' in contribution_data:
            temporal_data = contribution_data['temporal_patterns']
            
            # Seasonal insights
            if 'seasonal_patterns' in temporal_data:
                seasonal_data = temporal_data['seasonal_patterns']
                best_season = max(seasonal_data, key=lambda x: x['Total_Contribution'])
                worst_season = min(seasonal_data, key=lambda x: x['Total_Contribution'])
                
                insights['seasonal_analysis'] = {
                    'best_performing_season': {
                        'season': best_season['Season'],
                        'contribution': best_season['Total_Contribution'],
                        'insight': f"Peak performance during {best_season['Season']} suggests seasonal marketing opportunities"
                    },
                    'worst_performing_season': {
                        'season': worst_season['Season'],
                        'contribution': worst_season['Total_Contribution'],
                        'insight': f"Lower performance in {worst_season['Season']} indicates need for seasonal strategies"
                    },
                    'seasonal_variation': f"Contribution varies by {(best_season['Total_Contribution'] - worst_season['Total_Contribution']) / worst_season['Total_Contribution'] * 100:.0f}% between seasons"
                }
        
        return insights

    def interpret_efficiency_metrics(self, contribution_data):
        """Interpret efficiency metric insights"""
        insights = {}
        
        if 'efficiency_metrics' in contribution_data:
            efficiency_data = contribution_data['efficiency_metrics']
            
            insights['overall_efficiency'] = {
                'score': efficiency_data.get('overall_efficiency', 0),
                'interpretation': self.interpret_efficiency_score(efficiency_data.get('overall_efficiency', 0)),
                'benchmark': "Industry average typically ranges from 0.65 to 0.85"
            }
            
            insights['contribution_concentration'] = {
                'index': efficiency_data.get('contribution_concentration', 0),
                'interpretation': self.interpret_concentration_index(efficiency_data.get('contribution_concentration', 0)),
                'implication': "Lower concentration suggests better diversification"
            }
            
            insights['roi_efficiency'] = {
                'value': efficiency_data.get('roi_efficiency', 0),
                'interpretation': self.interpret_roi_efficiency(efficiency_data.get('roi_efficiency', 0)),
                'benchmark': "Industry average ROI typically ranges from 2.0 to 3.5"
            }
        
        return insights

    def interpret_efficiency_score(self, score):
        """Interpret efficiency score"""
        if score >= 0.85:
            return "Excellent - Top tier performance"
        elif score >= 0.75:
            return "Good - Above average performance"
        elif score >= 0.65:
            return "Average - Room for improvement"
        else:
            return "Below average - Significant optimization needed"

    def interpret_concentration_index(self, index):
        """Interpret concentration index"""
        if index <= 0.3:
            return "Well diversified - Good channel balance"
        elif index <= 0.5:
            return "Moderately diversified - Some concentration risk"
        else:
            return "Highly concentrated - High dependency risk"

    def interpret_roi_efficiency(self, roi):
        """Interpret ROI efficiency"""
        if roi >= 3.0:
            return "Excellent - High return on investment"
        elif roi >= 2.5:
            return "Good - Above average returns"
        elif roi >= 2.0:
            return "Average - Acceptable returns"
        else:
            return "Below average - Low returns"

    def generate_strategic_insights(self, contribution_data):
        """Generate strategic insights from contribution data"""
        insights = []
        
        # Channel strategy insights
        if 'optimization_opportunities' in contribution_data:
            for opp in contribution_data['optimization_opportunities']:
                if opp['type'] == 'high_roi_opportunities':
                    insights.append({
                        'type': 'investment_strategy',
                        'insight': f"High ROI channels ({', '.join(opp['channels'])}) present scaling opportunities",
                        'strategic_implication': "Consider reallocating budget from lower-performing channels",
                        'expected_impact': "Potential 20-30% increase in overall marketing effectiveness"
                    })
        
        # Portfolio optimization insights
        if 'efficiency_metrics' in contribution_data:
            efficiency = contribution_data['efficiency_metrics']
            if efficiency.get('overall_efficiency', 0) < 0.75:
                insights.append({
                    'type': 'portfolio_optimization',
                    'insight': "Overall efficiency below optimal levels",
                    'strategic_implication': "Focus on improving underperforming channels and processes",
                    'expected_impact': "Potential 15-25% improvement in overall marketing efficiency"
                })
        
        return insights

    def analyze_market_context(self, contribution_data):
        """Analyze market context for contribution insights"""
        context = {
            'market_trends': [
                'Digital transformation driving increased digital channel effectiveness',
                'Consumer behavior shifts toward omnichannel experiences',
                'Data-driven marketing enabling better attribution and optimization'
            ],
            'competitive_landscape': [
                'Competitors likely facing similar channel performance challenges',
                'First-mover advantage in optimizing high-ROI channels',
                'Opportunity to differentiate through data-driven optimization'
            ],
            'regulatory_environment': [
                'Privacy regulations affecting data collection and targeting',
                'Need for transparent attribution and measurement',
                'Compliance requirements for marketing effectiveness reporting'
            ]
        }
        
        return context

    def perform_competitive_analysis(self, contribution_data):
        """Perform competitive analysis based on contribution data"""
        analysis = {
            'competitive_positioning': {
                'strengths': [
                    'Data-driven approach to marketing optimization',
                    'Comprehensive channel performance measurement',
                    'Systematic identification of optimization opportunities'
                ],
                'weaknesses': [
                    'Potential gaps in underperforming channels',
                    'Seasonal performance variations',
                    'Efficiency scores below optimal levels'
                ]
            },
            'competitive_advantages': [
                'Advanced contribution analysis capabilities',
                'Real-time performance monitoring',
                'Proactive optimization recommendations'
            ],
            'threats': [
                'Competitors may adopt similar analytical approaches',
                'Market changes could affect channel effectiveness',
                'Technology disruption in marketing channels'
            ]
        }
        
        return analysis

    def generate_business_insights(self, interpretation_results):
        """Generate business insights from interpretation results"""
        insights = []
        
        # Channel performance insights
        if 'channel_insights' in interpretation_results:
            channel_insights = interpretation_results['channel_insights']
            if 'top_performer_analysis' in channel_insights:
                top_analysis = channel_insights['top_performer_analysis']
                insights.append({
                    'category': 'Channel Performance',
                    'insight': top_analysis['strength'],
                    'business_impact': 'High-performing channels provide foundation for growth',
                    'action_required': 'Leverage success factors and consider scaling up'
                })
        
        # Efficiency insights
        if 'efficiency_insights' in interpretation_results:
            efficiency_insights = interpretation_results['efficiency_insights']
            if 'overall_efficiency' in efficiency_insights:
                eff_analysis = efficiency_insights['overall_efficiency']
                insights.append({
                    'category': 'Operational Efficiency',
                    'insight': f"Overall efficiency: {eff_analysis['interpretation']}",
                    'business_impact': 'Efficiency directly affects marketing ROI and budget allocation',
                    'action_required': 'Focus on improving underperforming areas'
                })
        
        # Strategic insights
        if 'strategic_insights' in interpretation_results:
            for strategic in interpretation_results['strategic_insights']:
                insights.append({
                    'category': 'Strategic Planning',
                    'insight': strategic['insight'],
                    'business_impact': strategic['expected_impact'],
                    'action_required': strategic['strategic_implication']
                })
        
        return insights

    def generate_actionable_recommendations(self, interpretation_results):
        """Generate actionable recommendations from interpretation results"""
        recommendations = []
        
        # Channel optimization recommendations
        if 'channel_insights' in interpretation_results:
            channel_insights = interpretation_results['channel_insights']
            if 'efficiency_analysis' in channel_insights:
                eff_analysis = channel_insights['efficiency_analysis']
                recommendations.append({
                    'priority': 'High',
                    'category': 'Channel Optimization',
                    'action': f"Optimize {eff_analysis.get('least_efficient', 'underperforming')} channels",
                    'timeline': '3-6 months',
                    'resources_required': 'Marketing team, analytics tools',
                    'expected_outcome': 'Improve overall efficiency by 10-15%'
                })
        
        # Investment recommendations
        if 'strategic_insights' in interpretation_results:
            for strategic in interpretation_results['strategic_insights']:
                if strategic['type'] == 'investment_strategy':
                    recommendations.append({
                        'priority': 'High',
                        'category': 'Budget Allocation',
                        'action': strategic['strategic_implication'],
                        'timeline': '1-3 months',
                        'resources_required': 'Budget planning, performance monitoring',
                        'expected_outcome': strategic['expected_impact']
                    })
        
        # Process improvement recommendations
        recommendations.append({
            'priority': 'Medium',
            'category': 'Process Improvement',
            'action': 'Implement regular contribution monitoring dashboard',
            'timeline': '2-4 months',
            'resources_required': 'IT team, dashboard tools',
            'expected_outcome': 'Real-time visibility into marketing performance'
        })
        
        return recommendations

    def generate_marketing_optimization(self, interpretation_results):
        """Generate marketing optimization strategies"""
        optimization = {
            'short_term_optimizations': [
                'Reallocate budget from underperforming to high-ROI channels',
                'Optimize creative and messaging for top-performing channels',
                'Implement A/B testing for high-potential campaigns'
            ],
            'medium_term_optimizations': [
                'Develop seasonal marketing strategies based on temporal patterns',
                'Create channel-specific optimization playbooks',
                'Establish performance benchmarks and monitoring systems'
            ],
            'long_term_optimizations': [
                'Build predictive models for contribution forecasting',
                'Develop automated optimization algorithms',
                'Create comprehensive marketing attribution framework'
            ],
            'optimization_metrics': [
                'Channel efficiency scores',
                'ROI improvement',
                'Contribution concentration reduction',
                'Seasonal performance consistency'
            ]
        }
        
        return optimization

    def save_contribution_interpretation(self, interpretation_results, business_insights, actionable_recommendations, marketing_optimization):
        """Save contribution interpretation results to files"""
        try:
            # Save detailed interpretation report
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Save interpretation report as JSON
            interpretation_report = {
                'interpretation_results': interpretation_results,
                'business_insights': business_insights,
                'actionable_recommendations': actionable_recommendations,
                'marketing_optimization': marketing_optimization
            }
            
            report_path = output_dir / "contribution_interpretation_report.json"
            with open(report_path, 'w') as f:
                json.dump(interpretation_report, f, indent=2, default=str)
            
            # Save business insights summary
            insights_path = output_dir / "contribution_business_insights.txt"
            with open(insights_path, 'w') as f:
                f.write("CONTRIBUTION BUSINESS INSIGHTS\n")
                f.write("=" * 50 + "\n\n")
                
                for insight in business_insights:
                    f.write(f"Category: {insight['category']}\n")
                    f.write(f"Insight: {insight['insight']}\n")
                    f.write(f"Business Impact: {insight['business_impact']}\n")
                    f.write(f"Action Required: {insight['action_required']}\n")
                    f.write("-" * 40 + "\n\n")
            
            # Save actionable recommendations
            recs_path = output_dir / "contribution_actionable_recommendations.txt"
            with open(recs_path, 'w') as f:
                f.write("CONTRIBUTION ACTIONABLE RECOMMENDATIONS\n")
                f.write("=" * 50 + "\n\n")
                
                for rec in actionable_recommendations:
                    f.write(f"Priority: {rec['priority']}\n")
                    f.write(f"Category: {rec['category']}\n")
                    f.write(f"Action: {rec['action']}\n")
                    f.write(f"Timeline: {rec['timeline']}\n")
                    f.write(f"Resources: {rec['resources_required']}\n")
                    f.write(f"Expected Outcome: {rec['expected_outcome']}\n")
                    f.write("-" * 40 + "\n\n")
            
            console.print(f"[green]Contribution interpretation saved to {output_dir}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error saving contribution interpretation: {e}[/red]")

    def build_graph(self, state:ContributionInterpreterState):
        workflow = StateGraph(state)
        workflow.add_node("supervisor", self.chatNode)
        workflow.add_node("contribution_interpretation_node", self.contribution_interpretation_node)

        workflow.add_edge(START, "supervisor")
        workflow.add_edge("supervisor", "contribution_interpretation_node")
        workflow.add_edge("contribution_interpretation_node", END)

        return workflow.compile(checkpointer= checkpointer)
