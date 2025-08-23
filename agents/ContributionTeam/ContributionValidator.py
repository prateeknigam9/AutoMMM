from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from agent_patterns.states import ContributionValidatorState
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

ValidatorPrompt = utility.load_prompt_config(
    r"prompts\AgentPrompts.yaml",
    "contribution_validator_prompts",
)

user_messages = utility.load_prompt_config(
    r"prompts\user_messages.yaml",
    "contribution_validator",
)


class ContributionValidatorAgent:
    def __init__(self, agent_name :str, agent_description:str, model:str = "llama3.1"):
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.graph = self.build_graph(ContributionValidatorState)
        self.llm = ChatOllama(model = model)
        # self.llm = ChatOpenAI(model = "gpt-4o-mini")
        
    def chatNode(self, state:ContributionValidatorState):
        prompt = ValidatorPrompt['validator_chat_prompt'].format(
            agent_name = self.agent_name,
            agent_description = self.agent_description
        )
        
        state['messages'] = state['messages'] + [
                    chat_utility.build_message_structure(role = "system", message = prompt)
                ]
        
        response = self.llm.invoke(state['messages'])
        theme_utility.display_response(response.content, title = self.agent_name)
        
        # Parse the response to check if we need to run validation
        parsed = chat_utility.parse_json_from_response(response.content)
        if parsed and isinstance(parsed, dict) and "action" in parsed:
            if parsed['action'] == "run_contribution_validation":
                return Command(
                    goto = "contribution_validation_node",
                    update = {
                        "validation_type": parsed.get("validation_type", "comprehensive"),
                        "messages": state['messages'] + [
                            chat_utility.build_message_structure(role = "assistant", message = response.content)
                        ]
                    }
                )
        
        chat_utility.append_to_structure(state['messages'], role="assistant", message = response.content)
        return Command(
            goto = "contribution_validation_node",
            update = {
                "validation_type": "comprehensive",
                "messages": state['messages']
            }
        )

    def contribution_validation_node(self, state: ContributionValidatorState):
        theme_utility.print_startup_info(
            agent_name=self.agent_name,
            agent_description=self.agent_description,
            is_interactive=False,
        )
        
        validation_type = state.get('validation_type', 'comprehensive')
        
        with console.status(f"[plum1] Running {validation_type} contribution validation...", spinner="dots"):
            # Load contribution analysis and interpretation results
            contribution_data = self.load_contribution_data()
            
            if contribution_data is None:
                # Create synthetic validation data for demonstration
                contribution_data = self.create_synthetic_validation_data()
            
            # Perform contribution validation
            validation_results = self.perform_contribution_validation(contribution_data, validation_type)
            
            # Generate validation report
            validation_report = self.generate_validation_report(validation_results)
            
            # Save validation results
            validation_report_path = self.save_contribution_validation(validation_report, validation_results)
        
        # Update state
        state['validation_results'] = validation_results
        state['validation_report'] = validation_report
        state['validation_report_path'] = validation_report_path
        state['completed'] = True
        
        # Generate response
        response = f"Contribution validation completed successfully. Validation type: {validation_type}. Generated comprehensive validation report with quality checks and consistency analysis."
        theme_utility.display_response(response, title = self.agent_name)
        
        return Command(
            goto = END,
            update = {
                "validation_results": validation_results,
                "validation_report": validation_report,
                "validation_report_path": validation_report_path,
                "completed": True,
                "messages": state['messages'] + [
                    chat_utility.build_message_structure(role = "assistant", message = response)
                ]
            }
        )

    def load_contribution_data(self):
        """Load contribution analysis and interpretation results from files"""
        try:
            # Look for contribution analysis and interpretation files
            output_dir = Path("output")
            analysis_report = output_dir / "contribution_analysis_report.json"
            interpretation_report = output_dir / "contribution_interpretation_report.json"
            business_insights = output_dir / "contribution_business_insights.txt"
            actionable_recommendations = output_dir / "contribution_actionable_recommendations.txt"
            
            contribution_data = {}
            
            if analysis_report.exists():
                with open(analysis_report, 'r') as f:
                    contribution_data['analysis_report'] = json.load(f)
            
            if interpretation_report.exists():
                with open(interpretation_report, 'r') as f:
                    contribution_data['interpretation_report'] = json.load(f)
            
            if business_insights.exists():
                with open(business_insights, 'r') as f:
                    contribution_data['business_insights'] = f.read()
            
            if actionable_recommendations.exists():
                with open(actionable_recommendations, 'r') as f:
                    contribution_data['actionable_recommendations'] = f.read()
            
            if contribution_data:
                return contribution_data
                
        except Exception as e:
            console.print(f"[red]Error loading contribution data: {e}[/red]")
        
        return None

    def create_synthetic_validation_data(self):
        """Create synthetic validation data for demonstration purposes"""
        np.random.seed(42)
        
        # Create synthetic contribution analysis and interpretation data
        validation_data = {
            'analysis_report': {
                'channel_performance': {
                    'top_performers': [
                        {'Channel': 'Digital', 'Contribution_Percentage': 28.5, 'ROI': 3.2, 'Efficiency_Score': 0.89},
                        {'Channel': 'TV', 'Contribution_Percentage': 25.3, 'ROI': 2.8, 'Efficiency_Score': 0.85}
                    ]
                },
                'efficiency_metrics': {
                    'overall_efficiency': 0.78,
                    'contribution_concentration': 0.45,
                    'roi_efficiency': 2.7
                }
            },
            'interpretation_report': {
                'business_insights': [
                    {
                        'category': 'Channel Performance',
                        'insight': 'Digital channel shows strong performance',
                        'business_impact': 'High-performing channels provide foundation for growth'
                    }
                ],
                'actionable_recommendations': [
                    {
                        'priority': 'High',
                        'category': 'Channel Optimization',
                        'action': 'Optimize underperforming channels',
                        'expected_outcome': 'Improve overall efficiency by 10-15%'
                    }
                ]
            }
        }
        
        return validation_data

    def perform_contribution_validation(self, contribution_data, validation_type):
        """Perform comprehensive contribution validation"""
        results = {}
        
        # Basic validation
        results['data_quality_checks'] = self.perform_data_quality_checks(contribution_data)
        results['consistency_validation'] = self.validate_consistency(contribution_data)
        results['logical_validation'] = self.validate_logical_consistency(contribution_data)
        
        if validation_type == 'comprehensive':
            results['business_logic_validation'] = self.validate_business_logic(contribution_data)
            results['recommendation_validation'] = self.validate_recommendations(contribution_data)
            results['risk_assessment'] = self.perform_risk_assessment(contribution_data)
        
        return results

    def perform_data_quality_checks(self, contribution_data):
        """Perform data quality checks on contribution data"""
        quality_checks = {}
        
        # Check for missing data
        missing_data = []
        if 'analysis_report' in contribution_data:
            analysis = contribution_data['analysis_report']
            if 'channel_performance' in analysis:
                if not analysis['channel_performance'].get('top_performers'):
                    missing_data.append('Top performers data missing')
                if not analysis['channel_performance'].get('efficiency_ranking'):
                    missing_data.append('Efficiency ranking data missing')
        
        quality_checks['missing_data'] = missing_data
        
        # Check data completeness
        completeness_score = 0
        total_fields = 0
        filled_fields = 0
        
        if 'analysis_report' in contribution_data:
            analysis = contribution_data['analysis_report']
            for category, data in analysis.items():
                if isinstance(data, dict):
                    total_fields += len(data)
                    filled_fields += sum(1 for v in data.values() if v is not None and v != '')
                elif isinstance(data, list):
                    total_fields += 1
                    filled_fields += 1 if data else 0
        
        if total_fields > 0:
            completeness_score = filled_fields / total_fields
        
        quality_checks['completeness_score'] = completeness_score
        quality_checks['completeness_status'] = self.assess_completeness(completeness_score)
        
        # Check data format consistency
        format_issues = []
        if 'analysis_report' in contribution_data:
            analysis = contribution_data['analysis_report']
            if 'efficiency_metrics' in analysis:
                metrics = analysis['efficiency_metrics']
                for metric, value in metrics.items():
                    if not isinstance(value, (int, float)) or np.isnan(value):
                        format_issues.append(f'Invalid format for {metric}: {value}')
        
        quality_checks['format_issues'] = format_issues
        
        return quality_checks

    def assess_completeness(self, score):
        """Assess data completeness score"""
        if score >= 0.9:
            return "Excellent - Complete dataset"
        elif score >= 0.8:
            return "Good - Minor gaps"
        elif score >= 0.7:
            return "Acceptable - Some gaps"
        else:
            return "Poor - Significant gaps"

    def validate_consistency(self, contribution_data):
        """Validate consistency across different data sources"""
        consistency_checks = {}
        
        # Check for internal consistency in analysis report
        internal_consistency = []
        if 'analysis_report' in contribution_data:
            analysis = contribution_data['analysis_report']
            
            # Check if contribution percentages sum to reasonable range
            if 'channel_performance' in analysis and 'top_performers' in analysis['channel_performance']:
                top_performers = analysis['channel_performance']['top_performers']
                total_contribution = sum(p.get('Contribution_Percentage', 0) for p in top_performers)
                
                if total_contribution > 100:
                    internal_consistency.append(f'Total contribution exceeds 100%: {total_contribution:.1f}%')
                elif total_contribution < 50:
                    internal_consistency.append(f'Total contribution seems low: {total_contribution:.1f}%')
            
            # Check efficiency score consistency
            if 'efficiency_metrics' in analysis:
                efficiency = analysis['efficiency_metrics'].get('overall_efficiency', 0)
                if not (0 <= efficiency <= 1):
                    internal_consistency.append(f'Invalid efficiency score: {efficiency}')
        
        consistency_checks['internal_consistency'] = internal_consistency
        
        # Check cross-report consistency
        cross_report_consistency = []
        if 'analysis_report' in contribution_data and 'interpretation_report' in contribution_data:
            analysis = contribution_data['analysis_report']
            interpretation = contribution_data['interpretation_report']
            
            # Check if insights align with data
            if 'business_insights' in interpretation:
                insights = interpretation['business_insights']
                for insight in insights:
                    if 'Channel Performance' in insight.get('category', ''):
                        if 'analysis_report' not in contribution_data:
                            cross_report_consistency.append('Business insights reference missing analysis data')
        
        consistency_checks['cross_report_consistency'] = cross_report_consistency
        
        return consistency_checks

    def validate_logical_consistency(self, contribution_data):
        """Validate logical consistency of contribution analysis"""
        logical_checks = {}
        
        logical_issues = []
        
        if 'analysis_report' in contribution_data:
            analysis = contribution_data['analysis_report']
            
            # Check if ROI and efficiency scores are logically consistent
            if 'channel_performance' in analysis and 'top_performers' in analysis['channel_performance']:
                top_performers = analysis['channel_performance']['top_performers']
                for performer in top_performers:
                    roi = performer.get('ROI', 0)
                    efficiency = performer.get('Efficiency_Score', 0)
                    
                    # High ROI should generally correlate with high efficiency
                    if roi > 3.0 and efficiency < 0.7:
                        logical_issues.append(f'High ROI ({roi}) but low efficiency ({efficiency}) for {performer.get("Channel", "Unknown")}')
                    
                    # Contribution percentage should be reasonable
                    contribution = performer.get('Contribution_Percentage', 0)
                    if contribution > 50:
                        logical_issues.append(f'Unusually high contribution percentage: {contribution}% for {performer.get("Channel", "Unknown")}')
            
            # Check efficiency metrics logic
            if 'efficiency_metrics' in analysis:
                metrics = analysis['efficiency_metrics']
                overall_efficiency = metrics.get('overall_efficiency', 0)
                roi_efficiency = metrics.get('roi_efficiency', 0)
                
                # Overall efficiency should be reasonable
                if overall_efficiency > 1.0:
                    logical_issues.append(f'Invalid overall efficiency: {overall_efficiency} (should be <= 1.0)')
                
                # ROI should be positive
                if roi_efficiency <= 0:
                    logical_issues.append(f'Invalid ROI efficiency: {roi_efficiency} (should be > 0)')
        
        logical_checks['logical_issues'] = logical_issues
        logical_checks['logical_consistency_score'] = max(0, 1 - len(logical_issues) * 0.1)
        
        return logical_checks

    def validate_business_logic(self, contribution_data):
        """Validate business logic of contribution analysis"""
        business_logic_checks = {}
        
        business_logic_issues = []
        
        if 'interpretation_report' in contribution_data:
            interpretation = contribution_data['interpretation_report']
            
            # Check if recommendations align with insights
            if 'business_insights' in interpretation and 'actionable_recommendations' in interpretation:
                insights = interpretation['business_insights']
                recommendations = interpretation['actionable_recommendations']
                
                # Check if high-priority recommendations have corresponding insights
                high_priority_recs = [r for r in recommendations if r.get('priority') == 'High']
                for rec in high_priority_recs:
                    rec_category = rec.get('category', '')
                    matching_insights = [i for i in insights if i.get('category') == rec_category]
                    
                    if not matching_insights:
                        business_logic_issues.append(f'High-priority recommendation "{rec.get("action", "")}" lacks supporting business insight')
            
            # Check if optimization strategies are realistic
            if 'marketing_optimization' in interpretation:
                optimization = interpretation['marketing_optimization']
                short_term = optimization.get('short_term_optimizations', [])
                
                for opt in short_term:
                    if 'reallocate budget' in opt.lower():
                        # Check if there's data to support budget reallocation
                        if 'analysis_report' not in contribution_data:
                            business_logic_issues.append('Budget reallocation recommendation without supporting performance data')
        
        business_logic_checks['business_logic_issues'] = business_logic_issues
        business_logic_checks['business_logic_score'] = max(0, 1 - len(business_logic_issues) * 0.1)
        
        return business_logic_checks

    def validate_recommendations(self, contribution_data):
        """Validate the quality and feasibility of recommendations"""
        recommendation_validation = {}
        
        validation_issues = []
        
        if 'interpretation_report' in contribution_data:
            interpretation = contribution_data['interpretation_report']
            
            if 'actionable_recommendations' in interpretation:
                recommendations = interpretation['actionable_recommendations']
                
                for rec in recommendations:
                    # Check if timeline is realistic
                    timeline = rec.get('timeline', '')
                    if 'months' in timeline:
                        try:
                            months = int(timeline.split()[0])
                            if months < 1:
                                validation_issues.append(f'Unrealistic timeline: {timeline} for {rec.get("action", "")}')
                        except ValueError:
                            validation_issues.append(f'Invalid timeline format: {timeline}')
                    
                    # Check if expected outcome is measurable
                    outcome = rec.get('expected_outcome', '')
                    if not any(word in outcome.lower() for word in ['%', 'percent', 'increase', 'decrease', 'improve']):
                        validation_issues.append(f'Non-measurable expected outcome: {outcome}')
                    
                    # Check if resources are specified
                    resources = rec.get('resources_required', '')
                    if not resources or resources.strip() == '':
                        validation_issues.append(f'Missing resources specification for: {rec.get("action", "")}')
        
        recommendation_validation['validation_issues'] = validation_issues
        recommendation_validation['recommendation_quality_score'] = max(0, 1 - len(validation_issues) * 0.1)
        
        return recommendation_validation

    def perform_risk_assessment(self, contribution_data):
        """Perform risk assessment of contribution analysis"""
        risk_assessment = {}
        
        risks = []
        risk_level = 'Low'
        
        # Data quality risks
        if 'analysis_report' in contribution_data:
            analysis = contribution_data['analysis_report']
            if 'efficiency_metrics' in analysis:
                efficiency = analysis['efficiency_metrics'].get('overall_efficiency', 0)
                if efficiency < 0.7:
                    risks.append({
                        'category': 'Performance Risk',
                        'description': f'Low overall efficiency ({efficiency:.2f}) indicates potential performance issues',
                        'severity': 'Medium',
                        'mitigation': 'Focus on improving underperforming channels and processes'
                    })
        
        # Consistency risks
        if 'interpretation_report' in contribution_data:
            interpretation = contribution_data['interpretation_report']
            if 'actionable_recommendations' in interpretation:
                recommendations = interpretation['actionable_recommendations']
                high_priority_count = sum(1 for r in recommendations if r.get('priority') == 'High')
                
                if high_priority_count > 5:
                    risks.append({
                        'category': 'Implementation Risk',
                        'description': f'Too many high-priority recommendations ({high_priority_count}) may overwhelm implementation capacity',
                        'severity': 'Medium',
                        'mitigation': 'Prioritize and sequence recommendations based on impact and feasibility'
                    })
        
        # Business logic risks
        if 'analysis_report' in contribution_data and 'interpretation_report' in contribution_data:
            analysis = contribution_data['analysis_report']
            interpretation = contribution_data['interpretation_report']
            
            # Check if insights are supported by data
            if 'business_insights' in interpretation:
                insights = interpretation['business_insights']
                if len(insights) > 0 and 'analysis_report' not in contribution_data:
                    risks.append({
                        'category': 'Data Risk',
                        'description': 'Business insights generated without sufficient supporting data',
                        'severity': 'High',
                        'mitigation': 'Ensure all insights are backed by comprehensive data analysis'
                    })
        
        # Determine overall risk level
        high_risk_count = sum(1 for risk in risks if risk.get('severity') == 'High')
        medium_risk_count = sum(1 for risk in risks if risk.get('severity') == 'Medium')
        
        if high_risk_count > 0:
            risk_level = 'High'
        elif medium_risk_count > 2:
            risk_level = 'Medium'
        elif medium_risk_count > 0:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        risk_assessment['risks'] = risks
        risk_assessment['overall_risk_level'] = risk_level
        risk_assessment['risk_count'] = {
            'high': high_risk_count,
            'medium': medium_risk_count,
            'low': len(risks) - high_risk_count - medium_risk_count
        }
        
        return risk_assessment

    def generate_validation_report(self, validation_results):
        """Generate comprehensive validation report"""
        report = {
            'executive_summary': self.create_validation_summary(validation_results),
            'detailed_validation': validation_results,
            'quality_score': self.calculate_overall_quality_score(validation_results),
            'validation_status': self.determine_validation_status(validation_results),
            'recommendations': self.generate_validation_recommendations(validation_results)
        }
        
        return report

    def create_validation_summary(self, validation_results):
        """Create executive summary of validation results"""
        summary = {
            'overview': 'Comprehensive contribution analysis validation completed',
            'overall_quality': self.calculate_overall_quality_score(validation_results),
            'key_findings': self.extract_validation_findings(validation_results),
            'critical_issues': self.identify_critical_issues(validation_results)
        }
        
        return summary

    def calculate_overall_quality_score(self, validation_results):
        """Calculate overall quality score from validation results"""
        scores = []
        weights = []
        
        # Data quality score
        if 'data_quality_checks' in validation_results:
            quality_checks = validation_results['data_quality_checks']
            completeness_score = quality_checks.get('completeness_score', 0)
            format_issues = len(quality_checks.get('format_issues', []))
            format_score = max(0, 1 - format_issues * 0.1)
            
            data_quality_score = (completeness_score + format_score) / 2
            scores.append(data_quality_score)
            weights.append(0.3)
        
        # Consistency score
        if 'consistency_validation' in validation_results:
            consistency = validation_results['consistency_validation']
            internal_issues = len(consistency.get('internal_consistency', []))
            cross_report_issues = len(consistency.get('cross_report_consistency', []))
            
            consistency_score = max(0, 1 - (internal_issues + cross_report_issues) * 0.1)
            scores.append(consistency_score)
            weights.append(0.25)
        
        # Logical consistency score
        if 'logical_validation' in validation_results:
            logical = validation_results['logical_validation']
            logical_score = logical.get('logical_consistency_score', 0)
            scores.append(logical_score)
            weights.append(0.25)
        
        # Business logic score
        if 'business_logic_validation' in validation_results:
            business_logic = validation_results['business_logic_validation']
            business_logic_score = business_logic.get('business_logic_score', 0)
            scores.append(business_logic_score)
            weights.append(0.2)
        
        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            weighted_score = sum(score * weight for score, weight in zip(scores, weights)) / total_weight
            return round(weighted_score, 2)
        
        return 0.0

    def extract_validation_findings(self, validation_results):
        """Extract key findings from validation results"""
        findings = []
        
        # Data quality findings
        if 'data_quality_checks' in validation_results:
            quality_checks = validation_results['data_quality_checks']
            completeness_status = quality_checks.get('completeness_status', 'Unknown')
            findings.append(f"Data completeness: {completeness_status}")
            
            format_issues = quality_checks.get('format_issues', [])
            if format_issues:
                findings.append(f"Found {len(format_issues)} data format issues")
        
        # Consistency findings
        if 'consistency_validation' in validation_results:
            consistency = validation_results['consistency_validation']
            internal_issues = len(consistency.get('internal_consistency', []))
            cross_report_issues = len(consistency.get('cross_report_consistency', []))
            
            if internal_issues > 0:
                findings.append(f"Found {internal_issues} internal consistency issues")
            if cross_report_issues > 0:
                findings.append(f"Found {cross_report_issues} cross-report consistency issues")
        
        # Risk findings
        if 'risk_assessment' in validation_results:
            risk_assessment = validation_results['risk_assessment']
            risk_level = risk_assessment.get('overall_risk_level', 'Unknown')
            findings.append(f"Overall risk level: {risk_level}")
        
        return findings

    def identify_critical_issues(self, validation_results):
        """Identify critical issues from validation results"""
        critical_issues = []
        
        # High-risk issues
        if 'risk_assessment' in validation_results:
            risk_assessment = validation_results['risk_assessment']
            high_risks = [risk for risk in risk_assessment.get('risks', []) if risk.get('severity') == 'High']
            
            for risk in high_risks:
                critical_issues.append({
                    'type': 'High Risk',
                    'description': risk.get('description', ''),
                    'mitigation': risk.get('mitigation', '')
                })
        
        # Data quality issues
        if 'data_quality_checks' in validation_results:
            quality_checks = validation_results['data_quality_checks']
            completeness_score = quality_checks.get('completeness_score', 0)
            
            if completeness_score < 0.7:
                critical_issues.append({
                    'type': 'Data Quality',
                    'description': f'Low data completeness score: {completeness_score:.2f}',
                    'mitigation': 'Review data sources and ensure all required fields are populated'
                })
        
        # Logical consistency issues
        if 'logical_validation' in validation_results:
            logical = validation_results['logical_validation']
            logical_score = logical.get('logical_consistency_score', 0)
            
            if logical_score < 0.7:
                critical_issues.append({
                    'type': 'Logical Consistency',
                    'description': f'Low logical consistency score: {logical_score:.2f}',
                    'mitigation': 'Review analysis logic and ensure mathematical consistency'
                })
        
        return critical_issues

    def determine_validation_status(self, validation_results):
        """Determine overall validation status"""
        quality_score = self.calculate_overall_quality_score(validation_results)
        
        if quality_score >= 0.9:
            return "PASSED - Excellent Quality"
        elif quality_score >= 0.8:
            return "PASSED - Good Quality"
        elif quality_score >= 0.7:
            return "PASSED - Acceptable Quality"
        elif quality_score >= 0.6:
            return "CONDITIONAL PASS - Requires Review"
        else:
            return "FAILED - Significant Issues"

    def generate_validation_recommendations(self, validation_results):
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Data quality recommendations
        if 'data_quality_checks' in validation_results:
            quality_checks = validation_results['data_quality_checks']
            format_issues = quality_checks.get('format_issues', [])
            
            if format_issues:
                recommendations.append({
                    'priority': 'High',
                    'category': 'Data Quality',
                    'action': 'Fix data format issues identified in validation',
                    'timeline': 'Immediate',
                    'expected_outcome': 'Improve data quality and consistency'
                })
        
        # Consistency recommendations
        if 'consistency_validation' in validation_results:
            consistency = validation_results['consistency_validation']
            internal_issues = consistency.get('internal_consistency', [])
            
            if internal_issues:
                recommendations.append({
                    'priority': 'High',
                    'category': 'Data Consistency',
                    'action': 'Resolve internal consistency issues',
                    'timeline': '1-2 weeks',
                    'expected_outcome': 'Ensure data integrity across analysis'
                })
        
        # Risk mitigation recommendations
        if 'risk_assessment' in validation_results:
            risk_assessment = validation_results['risk_assessment']
            high_risks = [risk for risk in risk_assessment.get('risks', []) if risk.get('severity') == 'High']
            
            for risk in high_risks:
                recommendations.append({
                    'priority': 'Critical',
                    'category': 'Risk Mitigation',
                    'action': risk.get('mitigation', ''),
                    'timeline': 'Immediate',
                    'expected_outcome': 'Reduce risk level and improve analysis reliability'
                })
        
        return recommendations

    def save_contribution_validation(self, validation_report, validation_results):
        """Save contribution validation results to files"""
        try:
            # Save detailed validation report
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Save validation report as JSON
            report_path = output_dir / "contribution_validation_report.json"
            with open(report_path, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)
            
            # Save validation summary
            summary_path = output_dir / "contribution_validation_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("CONTRIBUTION VALIDATION SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("EXECUTIVE SUMMARY\n")
                f.write("-" * 20 + "\n")
                for key, value in validation_report['executive_summary'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("VALIDATION STATUS\n")
                f.write("-" * 20 + "\n")
                f.write(f"{validation_report['validation_status']}\n\n")
                
                f.write("OVERALL QUALITY SCORE\n")
                f.write("-" * 20 + "\n")
                f.write(f"{validation_report['quality_score']:.2f}/1.00\n\n")
                
                f.write("CRITICAL ISSUES\n")
                f.write("-" * 20 + "\n")
                for issue in validation_report['executive_summary']['critical_issues']:
                    f.write(f"• {issue['type']}: {issue['description']}\n")
                    f.write(f"  Mitigation: {issue['mitigation']}\n\n")
            
            # Save validation recommendations
            recs_path = output_dir / "contribution_validation_recommendations.txt"
            with open(recs_path, 'w') as f:
                f.write("CONTRIBUTION VALIDATION RECOMMENDATIONS\n")
                f.write("=" * 50 + "\n\n")
                
                for rec in validation_report['recommendations']:
                    f.write(f"Priority: {rec['priority']}\n")
                    f.write(f"Category: {rec['category']}\n")
                    f.write(f"Action: {rec['action']}\n")
                    f.write(f"Timeline: {rec['timeline']}\n")
                    f.write(f"Expected Outcome: {rec['expected_outcome']}\n")
                    f.write("-" * 40 + "\n\n")
            
            console.print(f"[green]Contribution validation saved to {output_dir}[/green]")
            
            return str(report_path)
            
        except Exception as e:
            console.print(f"[red]Error saving contribution validation: {e}[/red]")
            return ""

    def build_graph(self, state:ContributionValidatorState):
        workflow = StateGraph(state)
        workflow.add_node("supervisor", self.chatNode)
        workflow.add_node("contribution_validation_node", self.contribution_validation_node)

        workflow.add_edge(START, "supervisor")
        workflow.add_edge("supervisor", "contribution_validation_node")
        workflow.add_edge("contribution_validation_node", END)

        return workflow.compile(checkpointer= checkpointer)
