# AI Tutor System

An intelligent tutoring system built with Retrieval-Augmented Generation (RAG) using FAISS for efficient vector similarity search. This system allows you to create a personalized AI tutor that can answer questions based on your own documents and knowledge base.

## 🚀 Features

- **RAG-based Architecture**: Combines the power of large language models with your custom knowledge base
- **FAISS Vector Store**: Fast and efficient similarity search using Facebook's FAISS library
- **Gradio Interface**: User-friendly web interface for easy interaction
- **Multi-format Support**: Works with various document formats and URLs
- **GPU Accelerated**: Optimized for GPU usage for faster processing

## 📋 Prerequisites

- **GPU Required**: This system requires a CUDA-compatible GPU for optimal performance
- Python 3.8 or higher
- UV package manager

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd ai-tutor-system
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

## 📚 Setup Your Knowledge Base

1. **Create the documents folder**
   ```bash
   mkdir documents
   ```

2. **Add your content**
   - Place all your documents (PDF, TXT, DOCX, etc.) in the `documents/` folder
   - Add any URLs you want to include in your knowledge base
   - Supported formats: PDF, TXT, DOCX, MD, HTML, and more

3. **Build the vector store**
   ```bash
   uv run rag_backend.py
   ```
   
   This process will:
   - Parse all documents in the `documents/` folder
   - Create embeddings for the content
   - Build and save the FAISS vector index
   - Prepare the knowledge base for querying

## 🚀 Running the AI Tutor

1. **Start the Gradio interface**
   ```bash
   uv run app.py
   ```

2. **Access the application**
   - Open your web browser and navigate to the provided local URL (typically `http://127.0.0.1:7860`)
   - Start asking questions based on your uploaded documents!

## 💡 Usage Tips

- **Quality Documents**: The tutor's effectiveness depends on the quality and relevance of your documents
- **Specific Questions**: Ask specific questions for better, more focused answers
- **Context Matters**: The system will retrieve relevant document chunks to provide contextual answers
- **Iterative Learning**: Add more documents and rebuild the vector store to expand the tutor's knowledge

## 🏗️ System Architecture

```
Documents → Text Processing → Embeddings → FAISS Index
                                              ↓
User Query → Embedding → Similarity Search → Context + LLM → Response
```

## 📁 Project Structure

```
ai-tutor-system/
├── documents/              # Your knowledge base documents
├── rag_backend.py         # Vector store creation and RAG logic
├── app.py                 # Gradio web interface
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## ⚙️ Configuration

You can customize the system by modifying parameters in the configuration files:

- **Embedding Model**: Change the model used for creating embeddings
- **Chunk Size**: Adjust how documents are split for processing
- **Similarity Threshold**: Modify the relevance threshold for retrieved documents
- **GPU Settings**: Configure GPU memory usage and batch sizes

## 🔧 Troubleshooting

### Common Issues

1. **Out of GPU Memory**
   - Reduce batch size in the configuration
   - Use a smaller embedding model
   - Process documents in smaller chunks

2. **No Documents Found**
   - Ensure documents are placed in the `documents/` folder
   - Check file formats are supported
   - Verify file permissions

3. **Slow Performance**
   - Ensure GPU is properly configured
   - Check CUDA installation
   - Consider reducing the number of documents for initial testing

### System Requirements

- **Minimum GPU**: 4GB VRAM
- **Recommended GPU**: 8GB+ VRAM
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: At least 2GB free space for the vector index

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the logs for error messages
3. Open an issue on the repository with detailed information about your problem

## 🙏 Acknowledgments

- FAISS by Facebook Research for efficient similarity search
- Gradio for the intuitive web interface
- The open-source community for the underlying ML models

---

**Happy Learning with your AI Tutor! 🎓**
