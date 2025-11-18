
---

## 🔍 Key Features

- 📄 **Multi-format Parsing**: Supports PDF, DOCX, and TXT using [LlamaParse](https://llamaindex.ai/).
- 🧠 **AI-Powered Reasoning**: Uses **Google Gemini 2.0 Flash Thinking Exp** for deep contextual analysis.
- 📑 **Section Breakdown**: Extracts requirements, timelines, compliance info, and more.
- 📊 **Table Extraction**: Detects and summarizes tables embedded in the document.
- ⚡ **Streamlit Interface**: Clean and responsive web UI.
- 🔐 **Secure API Key Management** with `.env`.

---

## 🖼️ Screenshot

<img width="1907" height="972" alt="image" src="https://github.com/user-attachments/assets/f93d852a-b803-493c-b0a1-782f9f4a1f03" />

---

## 🧠 Why It Stands Out

RFP Insights combines two powerful engines:

- **LlamaParse** for high-fidelity text + table extraction from complex documents.
- **Gemini 2.0 Flash Thinking Exp** from Google — a cutting-edge model that interprets and reasons through RFP content far beyond simple keyword matching.

---

## 🛠️ Tech Stack

| Component             | Description                                 |
| --------------------- | ------------------------------------------- |
| `google.generativeai` | Gemini 2.0 Flash Thinking Exp (LLM backend) |
| `llama_parse`         | Structured document parsing                 |
| `streamlit`           | UI framework                                |
| `pandas`              | Table processing and summarization          |
| `dotenv`              | Secure API key management                   |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/pranav-kadam/rfpanalysis.git
cd rfpanalysis
```

````

### 2. Set Up a Virtual Environment

```bash
python -m venv venv

# On Linux/macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add:

```env
GEMINI_API_KEY=your_gemini_api_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
```

> 🗝️ You’ll need valid API keys from:
>
> - [Google AI Studio (Gemini)](https://aistudio.google.com/app/apikey)
> - [Llama Cloud](https://cloud.llamaindex.ai/)

### 5. Run the App

```bash
streamlit run main.py
```

Visit `http://localhost:8501` in your browser.

---

## 🧪 How to Use

1. 📁 Upload your RFP document.
2. 🎯 Choose an analysis type:

   - Full Document
   - Requirements
   - Timeline
   - Compliance

3. 📊 View the AI-generated analysis and extracted tables.
4. 🔍 Expand optional sections to explore raw content and tables.

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests.

📬 Connect with me on [LinkedIn](https://www.linkedin.com/in/suryawanshi-yash)

---



---
````
