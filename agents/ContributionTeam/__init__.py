"""
Contribution Team Package

This package contains agents specialized in marketing contribution analysis:
- ContributionTeamManager: Manages the overall contribution analysis workflow
- ContributionAnalyst: Analyzes marketing contribution patterns and creates reports
- ContributionInterpreter: Provides business insights and actionable recommendations
- ContributionValidator: Validates analysis quality and ensures consistency
"""

from .ContributionTeamManager import ContributionTeamManagerAgent
from .ContributionAnalyst import ContributionAnalystAgent
from .ContributionInterpreter import ContributionInterpreterAgent
from .ContributionValidator import ContributionValidatorAgent

__all__ = [
    'ContributionTeamManagerAgent',
    'ContributionAnalystAgent',
    'ContributionInterpreterAgent',
    'ContributionValidatorAgent'
]
