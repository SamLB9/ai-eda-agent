# 🤖 AI-EDA Agent

**Autonomous Exploratory Data Analysis powered by GPT-5**

An intelligent agent that performs complete EDA automatically in ~2 minutes. Powered by OpenAI's GPT-5, it generates insights, visualizations, and statistical analysis with zero manual intervention.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-green.svg)](https://openai.com)

## ✨ Features

- **🤖 Autonomous Agent**: No manual intervention needed - the AI plans, executes, and explains the analysis
- **⚡ Lightning Fast**: Complete EDA in approximately 2 minutes
- **📊 Smart Visualizations**: Auto-generated plots with descriptive titles and axis labels
- **🎯 Pre-loaded Datasets**: Three diverse demo datasets ready to use
- **🚀 One-Click Execution**: Upload your CSV or choose a demo dataset and run
- **📈 Comprehensive Analysis**: Dataset summary, statistical insights, correlations, and visualizations
- **🔒 Safe Execution**: Sandboxed code execution with timeout protection
- **💡 Intelligent Fallbacks**: Robust error handling with graceful degradation

## 🎯 What It Does

The AI-EDA Agent automatically:

1. **Analyzes your dataset** - shape, data types, missing values
2. **Generates visualizations** - histograms, box plots, scatter plots, correlations
3. **Provides statistical insights** - descriptive statistics, correlations, grouped analysis
4. **Creates comprehensive reports** - organized into clear sections with explanations
5. **Handles errors gracefully** - continues analysis even if some operations fail

## 📊 Demo Datasets

### 🎵 Spotify Features Dataset
- **Size**: 32MB (232,725 tracks × 18 features)
- **Source**: [Spotify Web API](https://developer.spotify.com/documentation/web-api/)
- **Features**: Popularity, danceability, energy, acousticness, instrumentalness, etc.
- **Use Case**: Music analysis, feature correlation, genre insights

### 🚢 Titanic Survival Dataset
- **Size**: 43KB (888 passengers × 12 features)
- **Source**: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- **Features**: Age, sex, class, fare, survival status, etc.
- **Use Case**: Classification analysis, survival prediction, demographic insights

### 🎮 Video Game Sales Dataset
- **Size**: 1.3MB (16,598 games × 11 features)
- **Source**: [VGChartz](https://www.vgchartz.com/) via Kaggle
- **Features**: Platform, genre, publisher, global sales, year, etc.
- **Use Case**: Sales analysis, platform comparison, market trends

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-eda-agent.git
   cd ai-eda-agent
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key**
   ```bash
   # Create .env file in project root
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

5. **Run the application**
   ```bash
   cd src
   streamlit run app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

## 📖 Usage

### Step 1: Choose Your Data
- **Use Demo Datasets**: Select from Spotify, Titanic, or Video Game Sales
- **Upload Your Own**: Drag and drop any CSV file
- **Start with Example**: Use the built-in sample dataset

### Step 2: Set Your Focus
Enter what you want to analyze, for example:
- "Find drivers of customer churn"
- "Analyze sales performance by region"
- "Identify factors affecting music popularity"

### Step 3: Run Analysis
Click **"Run Autonomous EDA"** and watch the AI agent work!

### Step 4: Review Results
The agent will provide:
- **Dataset Summary**: Basic info, missing values, data types
- **Visualizations**: 5-8 relevant plots with proper labels
- **Statistical Insights**: Correlations, grouped statistics, key relationships
- **Final Explanation**: Summary of findings and recommendations

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **AI Engine**: OpenAI GPT-5 for planning and code generation
- **Execution**: Sandboxed Python environment with timeout protection
- **Data Processing**: Pandas, NumPy, Matplotlib, Seaborn

### Security Features
- **Restricted Imports**: Only safe libraries allowed
- **Code Sandboxing**: Isolated execution environment
- **Timeout Protection**: Prevents infinite loops
- **Error Handling**: Graceful degradation on failures

### Performance Optimizations
- **Smart Sampling**: Large datasets automatically sampled for visualization
- **Efficient Operations**: Optimized for speed without sacrificing quality
- **Memory Management**: Automatic cleanup of large objects

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
OPENAI_ORG=your_organization_id            # Optional
```

### Customization
- **Model Selection**: Currently uses GPT-5-mini (configurable in code)
- **Timeout Settings**: Adjustable execution timeouts
- **Plot Limits**: Configurable number of visualizations

## 📁 Project Structure

```
ai-eda-agent/
├── src/
│   ├── app.py              # Streamlit main application
│   ├── llm.py              # OpenAI API integration
│   ├── executor.py          # Code execution engine
│   ├── session_runner.py    # EDA session orchestration
│   └── requirements.txt     # Python dependencies
├── data/
│   └── uploads/            # Demo datasets
│       ├── SpotifyFeatures.csv
│       ├── titanic.csv
│       └── vgsales.csv
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests
pytest

# Format code
black src/

# Lint code
flake8 src/
```

## 🐛 Troubleshooting

### Common Issues

**"Failed to load dataset"**
- Check file paths in the data directory
- Ensure CSV files are properly formatted
- Verify file permissions

**"OpenAI API error"**
- Verify your API key is correct
- Check your OpenAI account balance
- Ensure API key has proper permissions

**"Execution timeout"**
- Large datasets may take longer
- Check your internet connection
- Verify OpenAI API is responding

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Wiki**: Detailed documentation and examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing the GPT-5 API
- **Streamlit** for the amazing web framework
- **Pandas/NumPy/Matplotlib** for data processing and visualization
- **Kaggle** for providing the demo datasets
- **Open Source Community** for inspiration and support

## 📞 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Project**: [AI-EDA Agent](https://github.com/yourusername/ai-eda-agent)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-eda-agent/issues)

---

**⭐ Star this repository if you find it helpful!**

**🤖 Let the AI-EDA Agent analyze your data automatically!**
