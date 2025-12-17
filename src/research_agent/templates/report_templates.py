"""
Template system for context-aware report generation
Supports: Academic, Business, Product, Investment, Technical, General
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import yaml


class ReportContext(Enum):
    """Report context types"""
    ACADEMIC = "academic"
    BUSINESS = "business"
    PRODUCT = "product"
    INVESTMENT = "investment"
    TECHNICAL = "technical"
    GENERAL = "general"


@dataclass
class TemplateSection:
    """Template section definition"""
    name: str
    prompt: str
    required: bool = True
    max_length: int = 1000


class ReportTemplate:
    """Base template for report generation"""
    
    def __init__(
        self,
        context: ReportContext,
        sections: List[TemplateSection],
        citation_style: Optional[str] = None
    ):
        self.context = context
        self.sections = sections
        self.citation_style = citation_style
    
    def get_generation_prompt(
        self,
        question: str,
        context_docs: str,
        section: TemplateSection
    ) -> str:
        """Generate prompt for a specific section"""
        base_prompt = f"""You are generating the "{section.name}" section of a {self.context.value} report.

Research Question: {question}

Context Information:
{context_docs}

Section Requirements:
{section.prompt}

Generate ONLY the content for this section. Be comprehensive, factual, and well-structured.
Maximum length: ~{section.max_length} words.
"""
        if self.citation_style:
            base_prompt += f"\nUse {self.citation_style} citation style where appropriate.\n"
        
        return base_prompt
    
    def format_report(self, sections_content: Dict[str, str]) -> str:
        """Format all sections into final report"""
        report_parts = []
        
        for section in self.sections:
            content = sections_content.get(section.name, "")
            if content:
                report_parts.append(f"## {section.name}")
                report_parts.append(content)
                report_parts.append("")  # Empty line
        
        return "\n".join(report_parts)


class AcademicTemplate(ReportTemplate):
    """Template for academic research reports"""
    
    def __init__(self):
        sections = [
            TemplateSection(
                name="Abstract",
                prompt="""Write a concise abstract (150-250 words) summarizing:
- Research question and objectives
- Key findings and methodologies
- Main conclusions and implications
Be precise and informative.""",
                max_length=250
            ),
            TemplateSection(
                name="Introduction",
                prompt="""Provide comprehensive introduction including:
- Background and context
- Research problem and significance
- Objectives and scope
- Brief overview of existing research
Use formal academic tone.""",
                max_length=800
            ),
            TemplateSection(
                name="Literature Review",
                prompt="""Analyze existing research and findings:
- Key theories and frameworks
- Major studies and their contributions
- Gaps in current knowledge
- How this research fits in
Include citations to sources.""",
                max_length=1000
            ),
            TemplateSection(
                name="Methodology",
                prompt="""Describe research approach:
- Research design and methods
- Data sources and collection
- Analysis techniques
- Limitations and considerations""",
                max_length=600,
                required=False
            ),
            TemplateSection(
                name="Findings & Analysis",
                prompt="""Present and analyze key findings:
- Main results and observations
- Data analysis and interpretation
- Patterns and trends identified
- Statistical or qualitative insights
Support with evidence from sources.""",
                max_length=1200
            ),
            TemplateSection(
                name="Discussion",
                prompt="""Discuss implications and significance:
- Interpretation of findings
- Comparison with existing research
- Theoretical and practical implications
- Limitations and constraints""",
                max_length=800
            ),
            TemplateSection(
                name="Conclusion",
                prompt="""Provide clear conclusions:
- Summary of key findings
- Answer to research question
- Contributions to the field
- Future research directions""",
                max_length=500
            ),
        ]
        
        super().__init__(
            context=ReportContext.ACADEMIC,
            sections=sections,
            citation_style="APA"
        )


class BusinessTemplate(ReportTemplate):
    """Template for business analysis reports"""
    
    def __init__(self):
        sections = [
            TemplateSection(
                name="Executive Summary",
                prompt="""Write executive summary for business stakeholders:
- Key findings and recommendations
- Business impact and opportunities
- Critical action items
- ROI and strategic value
Keep concise and actionable.""",
                max_length=300
            ),
            TemplateSection(
                name="Market Analysis",
                prompt="""Analyze market conditions:
- Market size and growth trends
- Key players and competition
- Market dynamics and forces
- Opportunities and threats""",
                max_length=800
            ),
            TemplateSection(
                name="Strategic Insights",
                prompt="""Provide strategic analysis:
- Strategic positioning
- Competitive advantages
- Value propositions
- Market opportunities""",
                max_length=700
            ),
            TemplateSection(
                name="Financial Overview",
                prompt="""Analyze financial aspects:
- Revenue and profitability trends
- Cost structures
- Investment requirements
- Financial projections""",
                max_length=600,
                required=False
            ),
            TemplateSection(
                name="Risks & Challenges",
                prompt="""Identify and assess risks:
- Market risks
- Operational challenges
- Regulatory concerns
- Mitigation strategies""",
                max_length=500
            ),
            TemplateSection(
                name="Recommendations",
                prompt="""Provide clear, actionable recommendations:
- Strategic recommendations
- Implementation priorities
- Resource requirements
- Expected outcomes and timeline""",
                max_length=600
            ),
        ]
        
        super().__init__(
            context=ReportContext.BUSINESS,
            sections=sections
        )


class ProductTemplate(ReportTemplate):
    """Template for product analysis and reviews"""
    
    def __init__(self):
        sections = [
            TemplateSection(
                name="Product Overview",
                prompt="""Provide comprehensive product overview:
- Product description and purpose
- Target market and use cases
- Key features and capabilities
- Positioning and differentiation""",
                max_length=500
            ),
            TemplateSection(
                name="Features & Specifications",
                prompt="""Detail features and specifications:
- Core features and functionality
- Technical specifications
- Design and user experience
- Performance characteristics""",
                max_length=800
            ),
            TemplateSection(
                name="Market Comparison",
                prompt="""Compare with alternatives:
- Competitive landscape
- Strengths and weaknesses
- Unique selling propositions
- Price positioning""",
                max_length=700
            ),
            TemplateSection(
                name="User Experience",
                prompt="""Analyze user experience:
- Usability and ease of use
- Customer feedback and reviews
- Pain points and limitations
- Overall satisfaction""",
                max_length=600,
                required=False
            ),
            TemplateSection(
                name="Verdict & Recommendations",
                prompt="""Provide final assessment:
- Overall evaluation
- Best use cases
- Who should buy/use
- Value for money
- Final recommendations""",
                max_length=400
            ),
        ]
        
        super().__init__(
            context=ReportContext.PRODUCT,
            sections=sections
        )


class InvestmentTemplate(ReportTemplate):
    """Template for investment analysis"""
    
    def __init__(self):
        sections = [
            TemplateSection(
                name="Investment Thesis",
                prompt="""Present investment thesis:
- Investment opportunity overview
- Key value drivers
- Growth catalysts
- Investment rationale""",
                max_length=400
            ),
            TemplateSection(
                name="Market & Industry Analysis",
                prompt="""Analyze market and industry:
- Industry trends and dynamics
- Market size and growth
- Competitive landscape
- Regulatory environment""",
                max_length=800
            ),
            TemplateSection(
                name="Financial Analysis",
                prompt="""Provide financial analysis:
- Revenue and earnings trends
- Profitability metrics
- Cash flow analysis
- Valuation assessment""",
                max_length=900
            ),
            TemplateSection(
                name="Risk Assessment",
                prompt="""Evaluate investment risks:
- Business risks
- Financial risks
- Market risks
- Regulatory and legal risks
- Risk mitigation factors""",
                max_length=700
            ),
            TemplateSection(
                name="Investment Recommendation",
                prompt="""Provide investment recommendation:
- Buy/Hold/Sell recommendation
- Target price and timeline
- Expected returns
- Key monitoring metrics
- Action items""",
                max_length=400
            ),
        ]
        
        super().__init__(
            context=ReportContext.INVESTMENT,
            sections=sections
        )


class TechnicalTemplate(ReportTemplate):
    """Template for technical documentation"""
    
    def __init__(self):
        sections = [
            TemplateSection(
                name="Technical Overview",
                prompt="""Provide technical overview:
- Technology description
- Architecture and components
- Technical specifications
- Use cases and applications""",
                max_length=600
            ),
            TemplateSection(
                name="Implementation Details",
                prompt="""Detail implementation:
- Setup and configuration
- Dependencies and requirements
- Integration points
- Best practices""",
                max_length=900
            ),
            TemplateSection(
                name="Performance & Scalability",
                prompt="""Analyze performance:
- Performance characteristics
- Scalability considerations
- Bottlenecks and limitations
- Optimization strategies""",
                max_length=600,
                required=False
            ),
            TemplateSection(
                name="Security Considerations",
                prompt="""Address security aspects:
- Security features
- Vulnerabilities and risks
- Security best practices
- Compliance requirements""",
                max_length=500,
                required=False
            ),
            TemplateSection(
                name="Conclusion & Best Practices",
                prompt="""Summarize and recommend:
- Key takeaways
- Recommended approach
- Common pitfalls to avoid
- Additional resources""",
                max_length=400
            ),
        ]
        
        super().__init__(
            context=ReportContext.TECHNICAL,
            sections=sections
        )


class GeneralTemplate(ReportTemplate):
    """Template for general research (flexible LLM-decided format)"""
    
    def __init__(self):
        sections = [
            TemplateSection(
                name="Complete Report",
                prompt="""Generate a comprehensive, well-structured report on the topic.

Use your judgment to organize the content with appropriate sections and subsections.
The report should include:
- Clear introduction and context
- Main findings and analysis
- Supporting evidence and data
- Logical flow and structure
- Conclusion and key takeaways

Format with markdown headers (##, ###) for sections and subsections.
Make it informative, well-organized, and easy to read.""",
                max_length=3000
            ),
        ]
        
        super().__init__(
            context=ReportContext.GENERAL,
            sections=sections
        )


class TemplateFactory:
    """Factory for creating report templates"""
    
    _templates = {
        ReportContext.ACADEMIC: AcademicTemplate,
        ReportContext.BUSINESS: BusinessTemplate,
        ReportContext.PRODUCT: ProductTemplate,
        ReportContext.INVESTMENT: InvestmentTemplate,
        ReportContext.TECHNICAL: TechnicalTemplate,
        ReportContext.GENERAL: GeneralTemplate,
    }
    
    @classmethod
    def create_template(cls, context: ReportContext) -> ReportTemplate:
        """Create template for given context"""
        template_class = cls._templates.get(context)
        if not template_class:
            return GeneralTemplate()
        return template_class()
    
    @classmethod
    def get_available_contexts(cls) -> List[str]:
        """Get list of available context types"""
        return [ctx.value for ctx in ReportContext]


def get_template(context: str) -> ReportTemplate:
    """
    Get template for context string
    
    Args:
        context: Context name (academic, business, etc.)
        
    Returns:
        ReportTemplate instance
    """
    try:
        ctx = ReportContext(context.lower())
        return TemplateFactory.create_template(ctx)
    except ValueError:
        print(f"Unknown context '{context}', using general template")
        return GeneralTemplate()
