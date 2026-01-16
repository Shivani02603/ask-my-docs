# 📚 RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system that allows you to ask questions about PDF documents using AI-powered search.

![RAG System](https://img.shields.io/badge/RAG-System-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)

## 🚀 Features

- 📄 **PDF Document Processing**: Automatically extracts and indexes content from PDF files
- 🔍 **Semantic Search**: Uses vector embeddings for intelligent document retrieval
- 🤖 **AI-Powered Answers**: Optional OpenAI integration for natural language responses
- 💡 **Context Display**: Shows relevant document excerpts even without API key
- 🎨 **Beautiful UI**: Clean, modern Streamlit interface
- ⚡ **Fast Performance**: ChromaDB vector store for quick searches

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **LLM Framework**: LangChain
- **Vector Store**: ChromaDB
- **Embeddings**: HuggingFace Sentence Transformers (free, no API key needed)
- **LLM**: OpenAI GPT-3.5-turbo (optional)
- **PDF Processing**: PyPDF

## 📦 Installation

### Local Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rag-tutorial-v2
```

2. **Install dependencies**
```bash
pip install -r requirements_streamlit.txt
```

3. **Add your PDF documents**
Place your PDF files in the `data/` folder (currently includes Monopoly and Ticket to Ride rulebooks as examples)

4. **Initialize the database**
```bash
python setup_database.py
```

5. **Run the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🌐 Deployment

### Deploy to Streamlit Cloud (Recommended)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Deploy!

**Note**: The database will be automatically initialized on first deployment using the PDFs in the `data/` folder.

### Deploy to Other Platforms

#### Render
- Create a new Web Service
- Connect your repository
- Build Command: `pip install -r requirements_streamlit.txt && python setup_database.py`
- Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

#### Heroku
```bash
heroku create your-app-name
git push heroku main
```

## 🎯 Usage

### Without OpenAI API (Free)
- Just type your question and click "Search"
- You'll get relevant excerpts from the documents

### With OpenAI API (AI-Generated Answers)
1. Get an API key from [OpenAI](https://platform.openai.com)
2. Enter your API key in the sidebar
3. Ask questions to get AI-generated answers based on your documents

### Example Questions
- "What are the basic rules of Monopoly?"
- "How do players win in Ticket to Ride?"
- "What happens when you land on a property?"
- "Explain the game setup"

## 📁 Project Structure

```
rag-tutorial-v2/
├── app.py                      # Main Streamlit application
├── setup_database.py           # Database initialization script
├── data/                       # PDF documents folder
│   ├── monopoly.pdf
│   └── ticket_to_ride.pdf
├── chroma/                     # Vector database (auto-generated)
├── requirements_streamlit.txt  # Python dependencies
└── README.md                   # This file
```

## 🔧 Configuration

### Changing the Embedding Model
Edit `app.py` and modify the `get_embedding_function()`:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="your-preferred-model"
)
```

### Changing Chunk Size
Edit `setup_database.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Adjust this
    chunk_overlap=80,    # And this
)
```

### Using Different LLMs
You can modify `app.py` to use other LLMs like:
- Anthropic Claude
- Google PaLM
- Local models via Ollama
- HuggingFace models

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new features
- Improve the UI
- Fix bugs
- Add more document types support

## 📝 License

MIT License - feel free to use this project for personal or commercial purposes.

## 🙏 Acknowledgments

- Original tutorial inspiration from [pixegami](https://github.com/pixegami)
- Built with [LangChain](https://langchain.com)
- Powered by [Streamlit](https://streamlit.io)
- Vector storage by [ChromaDB](https://www.trychroma.com)

## 📧 Support

If you have questions or issues, please open an issue on GitHub.

---

Made with ❤️ using LangChain + ChromaDB + Streamlit
