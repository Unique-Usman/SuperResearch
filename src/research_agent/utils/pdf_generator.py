"""
PDF Report Generator
Creates professional PDF reports from structured content
"""
from weasyprint import HTML, CSS
from typing import Dict, List
import markdown2
from datetime import datetime
import os


class PDFGenerator:
    """Generate professional PDF reports"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize PDF generator
        
        Args:
            output_dir: Directory to save PDF reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate(
        self,
        title: str,
        content: Dict,
        filename: str,
        metadata: Dict = None
    ) -> str:
        """
        Generate PDF report
        
        Args:
            title: Report title
            content: Structured content dictionary
            filename: Output filename (without .pdf extension)
            metadata: Optional metadata (author, date, etc.)
            
        Returns:
            Path to generated PDF file
        """
        # Create HTML content
        html_content = self._create_html(title, content, metadata)
        
        # Generate PDF
        output_path = os.path.join(self.output_dir, f"{filename}.pdf")
        HTML(string=html_content).write_pdf(
            output_path,
            stylesheets=[CSS(string=self._get_css())]
        )
        
        return output_path
    
    def _create_html(
        self,
        title: str,
        content: Dict,
        metadata: Dict = None
    ) -> str:
        """Create HTML content for PDF"""
        
        # Extract sections
        sections = content.get("sections", [])
        executive_summary = content.get("executive_summary", "")
        conclusion = content.get("conclusion", "")
        sources = content.get("sources", [])
        
        # Build HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
</head>
<body>
    <!-- Cover Page -->
    <div class="cover-page">
        <div class="cover-content">
            <h1 class="cover-title">{title}</h1>
            <div class="cover-meta">
                <p>Research Report</p>
                <p>{metadata.get('date', datetime.now().strftime('%B %d, %Y'))}</p>
            </div>
        </div>
    </div>
    
    <!-- Executive Summary -->
    {self._format_executive_summary(executive_summary)}
    
    <!-- Table of Contents -->
    {self._format_toc(sections)}
    
    <!-- Main Content -->
    {self._format_sections(sections)}
    
    <!-- Conclusion -->
    {self._format_conclusion(conclusion)}
    
    <!-- Sources -->
    {self._format_sources(sources)}
</body>
</html>
"""
        return html
    
    def _format_executive_summary(self, summary: str) -> str:
        """Format executive summary section"""
        if not summary:
            return ""
        
        summary_html = markdown2.markdown(summary)
        return f"""
    <div class="page-break">
        <h2 class="section-title">Executive Summary</h2>
        <div class="executive-summary">
            {summary_html}
        </div>
    </div>
"""
    
    def _format_toc(self, sections: List[Dict]) -> str:
        """Format table of contents"""
        if not sections:
            return ""
        
        toc_items = ""
        for i, section in enumerate(sections, 1):
            title = section.get("title", f"Section {i}")
            toc_items += f'        <div class="toc-item">{i}. {title}</div>\n'
        
        return f"""
    <div class="page-break">
        <h2 class="section-title">Table of Contents</h2>
        <div class="toc">
{toc_items}
        </div>
    </div>
"""
    
    def _format_sections(self, sections: List[Dict]) -> str:
        """Format main content sections"""
        sections_html = ""
        
        for i, section in enumerate(sections, 1):
            title = section.get("title", f"Section {i}")
            content = section.get("content", "")
            subsections = section.get("subsections", [])
            
            # Convert markdown to HTML
            content_html = markdown2.markdown(content, extras=["tables", "fenced-code-blocks"])
            
            sections_html += f"""
    <div class="page-break">
        <h2 class="section-title">{i}. {title}</h2>
        <div class="section-content">
            {content_html}
        </div>
"""
            
            # Add subsections
            for j, subsection in enumerate(subsections, 1):
                sub_title = subsection.get("title", f"Subsection {i}.{j}")
                sub_content = subsection.get("content", "")
                sub_content_html = markdown2.markdown(sub_content, extras=["tables"])
                
                sections_html += f"""
        <h3 class="subsection-title">{i}.{j} {sub_title}</h3>
        <div class="subsection-content">
            {sub_content_html}
        </div>
"""
            
            sections_html += "    </div>\n"
        
        return sections_html
    
    def _format_conclusion(self, conclusion: str) -> str:
        """Format conclusion section"""
        if not conclusion:
            return ""
        
        conclusion_html = markdown2.markdown(conclusion)
        return f"""
    <div class="page-break">
        <h2 class="section-title">Conclusion</h2>
        <div class="conclusion">
            {conclusion_html}
        </div>
    </div>
"""
    
    def _format_sources(self, sources: List[str]) -> str:
        """Format sources/references section"""
        if not sources:
            return ""
        
        sources_html = ""
        for i, source in enumerate(sources, 1):
            sources_html += f"        <div class=\"source-item\">{i}. {source}</div>\n"
        
        return f"""
    <div class="page-break">
        <h2 class="section-title">Sources & References</h2>
        <div class="sources">
{sources_html}
        </div>
    </div>
"""
    
    def _get_css(self) -> str:
        """Get CSS styling for PDF"""
        return """
@page {
    size: A4;
    margin: 2.5cm;
}

body {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
}

/* Cover Page */
.cover-page {
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    page-break-after: always;
}

.cover-content {
    text-align: center;
}

.cover-title {
    font-size: 36pt;
    font-weight: bold;
    color: #1a1a1a;
    margin-bottom: 2cm;
    line-height: 1.3;
}

.cover-meta {
    font-size: 14pt;
    color: #666;
}

.cover-meta p {
    margin: 0.3cm 0;
}

/* Page Breaks */
.page-break {
    page-break-before: always;
}

/* Section Titles */
.section-title {
    font-size: 20pt;
    font-weight: bold;
    color: #2c3e50;
    margin-top: 0;
    margin-bottom: 0.8cm;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.3cm;
}

.subsection-title {
    font-size: 14pt;
    font-weight: bold;
    color: #34495e;
    margin-top: 0.6cm;
    margin-bottom: 0.4cm;
}

/* Executive Summary */
.executive-summary {
    background-color: #f8f9fa;
    padding: 0.8cm;
    border-left: 4px solid #3498db;
    margin-bottom: 1cm;
}

/* Table of Contents */
.toc {
    margin: 1cm 0;
}

.toc-item {
    padding: 0.3cm 0;
    border-bottom: 1px dotted #ccc;
    font-size: 12pt;
}

/* Content Sections */
.section-content, .subsection-content {
    text-align: justify;
}

.section-content p, .subsection-content p {
    margin: 0.4cm 0;
}

.section-content ul, .section-content ol {
    margin: 0.4cm 0;
    padding-left: 1cm;
}

.section-content li {
    margin: 0.2cm 0;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.6cm 0;
}

th, td {
    border: 1px solid #ddd;
    padding: 0.3cm;
    text-align: left;
}

th {
    background-color: #3498db;
    color: white;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #f2f2f2;
}

/* Conclusion */
.conclusion {
    font-style: italic;
    padding: 0.6cm;
    background-color: #f8f9fa;
    border-radius: 5px;
}

/* Sources */
.sources {
    font-size: 10pt;
}

.source-item {
    margin: 0.3cm 0;
    padding-left: 0.5cm;
    word-wrap: break-word;
}

/* Links */
a {
    color: #3498db;
    text-decoration: none;
}

/* Code blocks */
pre {
    background-color: #f4f4f4;
    padding: 0.4cm;
    border-radius: 5px;
    overflow-x: auto;
}

code {
    font-family: 'Courier New', monospace;
    font-size: 10pt;
}

/* Strong emphasis */
strong {
    color: #2c3e50;
}
"""
