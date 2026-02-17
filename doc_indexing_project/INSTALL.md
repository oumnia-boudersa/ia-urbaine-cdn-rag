# Installation Guide

## Install Dependencies

If installation fails (often due to network issues), try one of the following methods.

### Method 1: Use a PyPI mirror

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Or use the Aliyun mirror:

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### Method 2: Install core packages one by one

```bash
# Document processing
pip install PyPDF2 python-docx openpyxl markdown beautifulsoup4 lxml

# Text processing
pip install langchain langchain-text-splitters tiktoken

# Sparse indexing
pip install rank-bm25 scikit-learn jieba

# Dense (vector) indexing
pip install sentence-transformers faiss-cpu numpy

# Utilities
pip install tqdm python-dotenv
```

### Method 3: Use conda (if available)

```bash
conda install -c conda-forge pypdf2 python-docx openpyxl scikit-learn numpy
conda install -c conda-forge langchain langchain-text-splitters tiktoken rank-bm25 jieba sentence-transformers faiss-cpu tqdm python-dotenv
```

## Verify Installation

Run the following command to verify:

```bash
python -c "import PyPDF2; import langchain; import rank_bm25; import sentence_transformers; import faiss; print('All dependencies are installed!')"
```

## Notes

1. **jieba tokenization**: If you work with Chinese documents, it is strongly recommended to install `jieba`:
   ```bash
   pip install jieba
   ```

2. **FAISS**: 
   - CPU build: `pip install faiss-cpu`
   - GPU build (if you have a GPU): `pip install faiss-gpu`

3. **Sentence Transformers**: Models are downloaded automatically on first use; make sure you have a working internet connection.

4. **Memory requirements**:
   - Small docs (<100MB): at least 2GB RAM
   - Medium docs (100MB–1GB): at least 4GB RAM
   - Large docs (>1GB): at least 8GB RAM
