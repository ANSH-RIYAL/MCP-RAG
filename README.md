# MCP-RAG: Agentic AI Orchestration for Business Analytics

A lightweight demonstration of **Model Context Protocol (MCP)** combined with **Retrieval-Augmented Generation (RAG)** to orchestrate multi-agent AI workflows for business analysis.

## 🎯 What This Project Demonstrates

This project showcases how to build **agentic AI systems** that can:

1. **Orchestrate Multiple Agents**: MCP servers coordinate different specialized tools
2. **Retrieve Business Knowledge**: RAG provides context-aware information retrieval
3. **Perform Statistical Analysis**: Automated data analysis with natural language queries
4. **Maintain Modularity**: Easy to swap LLM backends and add new capabilities

## 🚀 Key Features

- **MCP-Based Coordination**: Multiple specialized servers working together
- **Business Analytics Tools**: Mean, standard deviation, correlation, linear regression
- **RAG Knowledge Base**: Business terms, policies, and analysis guidelines
- **Modular Design**: Easy to extend with new tools or swap LLM backends
- **Natural Language Interface**: Ask questions like "What's the average earnings from Q1?"

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API key (free tier available) - for future LLM integration
- Basic understanding of MCP and RAG concepts

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ANSH-RIYAL/MCP-RAG.git
   cd MCP-RAG
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (for future LLM integration):
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

## 🎮 Usage

### Quick Demo

Run the demonstration script to see both MCP servers in action:

```bash
python main.py
```

This will show:
- Business analytics tools working with sample data
- RAG knowledge retrieval for business terms
- How the systems can work together

### Business Analytics Tools

The system provides these analysis capabilities:

- **Data Exploration**: Get dataset information and sample data
- **Statistical Analysis**: Mean, standard deviation with filtering
- **Correlation Analysis**: Find relationships between variables
- **Predictive Modeling**: Linear regression for forecasting

### RAG Knowledge Retrieval

Access business knowledge through:

- **Term Definitions**: Look up business concepts
- **Policy Information**: Retrieve company procedures
- **Analysis Guidelines**: Get context for data interpretation

## 🏗️ Architecture

### Project Structure

```
MCP-RAG/
├── data/
│   ├── sample_business_data.csv    # Business dataset for analysis
│   └── business_knowledge.txt      # RAG knowledge base
├── src/
│   └── servers/
│       ├── business_analytics_server.py  # Statistical analysis tools
│       └── rag_server.py                 # Knowledge retrieval tools
├── main.py                         # Demo and orchestration script
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

### Key Components

1. **Business Analytics Server**: MCP server providing statistical analysis tools
2. **RAG Server**: MCP server for business knowledge retrieval
3. **Orchestration Layer**: Coordinates between servers and LLM (future)
4. **Data Layer**: Sample business data and knowledge base

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Gemini API key for LLM integration | None (future feature) |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.0-flash-exp` |

### Sample Data

The system includes:
- **Quarterly Business Data**: Sales, Marketing, Engineering metrics across 4 quarters
- **Business Knowledge Base**: Terms, policies, and analysis guidelines

## 🎯 Use Cases

### For Business Leaders
- **No-Code Analytics**: Ask natural language questions about business data
- **Quick Insights**: Get statistical analysis without technical expertise
- **Context-Aware Reports**: Combine data analysis with business knowledge

### For Data Teams
- **Modular Architecture**: Easy to add new analysis tools
- **LLM Integration**: Ready for natural language query processing
- **Extensible Framework**: Build custom agents for specific needs

### For AI Engineers
- **MCP Protocol**: Learn modern AI orchestration patterns
- **RAG Implementation**: Understand knowledge retrieval systems
- **Agentic Design**: Build multi-agent AI workflows

## 🚀 Future Enhancements

### Planned Features
- [ ] **LLM Integration**: Connect with Gemini, OpenAI, or local models
- [ ] **Natural Language Queries**: Process complex business questions
- [ ] **Advanced Analytics**: Time series analysis, clustering, forecasting
- [ ] **Web Interface**: User-friendly dashboard for non-technical users
- [ ] **Real-time Data**: Connect to live data sources
- [ ] **Custom Knowledge Bases**: Upload company-specific documents

### Integration Possibilities
- **Local LLM API**: Use open-source models with [Local LLM API](https://github.com/ANSH-RIYAL/local-llm-api)
- **Database Connectors**: Connect to SQL databases, data warehouses
- **API Integrations**: Salesforce, HubSpot, Google Analytics
- **Document Processing**: PDF, DOCX, email analysis

## 📝 Example Workflows

### Scenario 1: Sales Analysis
```
User: "What's the average earnings from Q1?"
System: 
1. RAG: Look up "earnings" definition
2. Analytics: Calculate mean of earnings column, filtered by Q1-2024
3. Response: "Average earnings for Q1-2024: $101,667"
```

### Scenario 2: Performance Correlation
```
User: "What's the correlation between sales and expenses?"
System:
1. Analytics: Calculate correlation between sales and expenses columns
2. RAG: Provide context about correlation interpretation
3. Response: "Correlation: 0.923 (strong positive relationship)"
```

### Scenario 3: Predictive Modeling
```
User: "Build a model to predict earnings from sales and employees"
System:
1. Analytics: Create linear regression model
2. RAG: Provide business context for the model
3. Response: "Model created with R² = 0.987"
```

## 🤝 Contributing

This is a foundation for building agentic AI systems. Contributions welcome:

- **New Analysis Tools**: Add statistical methods, ML models
- **Knowledge Base Expansion**: Business domains, industry-specific content
- **LLM Integrations**: Support for different AI models
- **Documentation**: Tutorials, use cases, best practices

## 📄 License

MIT License - feel free to use and modify for your own projects!

## 🔗 Related Projects

- **[Local LLM API](https://github.com/ANSH-RIYAL/local-llm-api)**: Run open-source LLMs locally
- **MCP Protocol**: [Official documentation](https://modelcontextprotocol.io/)

---

**Ready to build your own agentic AI system?** Start with this foundation and extend it for your specific needs. The modular design makes it easy to add new capabilities while maintaining clean architecture.

#AgenticAI #MCP #RAG #BusinessAnalytics #OpenSourceAI 