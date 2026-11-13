# Energy Operations RAG Assistant

## Project Overview

The **Energy Operations RAG Assistant** is a simple question-answering application designed to help users find information from renewable-energy operational documents.

In many energy operations, important instructions and guidelines are spread across different documents, such as maintenance manuals, safety instructions, and standard operating procedures (SOPs). Finding the right information manually can take time.

This application makes that process easier. Users can ask questions in everyday language, and the system searches its available documents to find relevant information. It then presents the answer along with the source documents and supporting information.

When an AI language model is configured, it can also generate a natural-language answer based on the retrieved information.

**Important:** This project is a document-based question-answering demonstration. It is not a renewable-energy forecasting system, and its current small document collection does not establish production-level accuracy.

## 1. What Does the Application Do?

The application follows a simple process:

1. It reads four renewable-energy operational documents.
2. It divides the documents into smaller sections so information can be searched more easily.
3. When a user asks a question, it searches for the most relevant sections.
4. If an AI model is configured, it uses those sections to generate an answer.
5. It displays the answer, the documents used as references, and a basic check to see whether the answer shares information with the retrieved content.

The application can still retrieve information even when the AI model is not configured. In that situation, it provides a response based directly on the retrieved document content.

### How It Works

```text
Operational Documents
        |
        v
Read and Organize Information
        |
        v
Break Documents into Smaller Sections
        |
        v
Convert Text into Searchable Data
        |
        v
User Asks a Question
        |
        v
Find the Most Relevant Sections
        |
        v
AI Generates an Answer (if configured)
        |
        v
Check Answer Against Retrieved Information
        |
        v
Display Answer and Sources
```

## 2. Main Components

| Component | What It Does |
|---|---|
| Document Collection | Contains four renewable-energy operational documents. |
| Text Processing | Organizes documents into smaller sections for searching. |
| Text Embeddings | Converts text into numerical representations so the system can compare information. |
| Information Retrieval | Finds up to three sections that are most relevant to the user's question. |
| AI Answer Generation | Uses a Groq-hosted language model to prepare an answer from the retrieved information, when configured. |
| Answer Support Check | Checks whether the generated answer shares important words with the retrieved documents. |
| Backend | Handles questions, document searching, and answer generation using FastAPI. |
| User Interface | Provides a simple webpage where users can enter questions and view answers. |

## 3. Documents Used

The application currently uses four sample documents stored in the `data/` folder:

- **Solar Inverter SOP:** Information about solar inverter operations and maintenance.
- **Wind Turbine Maintenance:** Information about wind turbine maintenance.
- **Hydro Safety Brief:** Safety-related information for hydro operations.
- **Grid Operations Note:** Information related to grid operations.

These documents form the information source for the application.

At present, each document is shorter than the configured section size. Therefore, each document produces only one searchable section. This means the current demonstration mainly searches across complete documents rather than testing searches across many sections of longer manuals.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language. |
| FastAPI | Handles communication between the webpage and the application. |
| HTML, CSS, JavaScript | Builds the user interface. |
| NumPy | Supports numerical calculations during information retrieval. |
| Sentence Transformers | Converts text into numerical representations that help identify related information. |
| TF-IDF | Provides an alternative keyword-based text comparison method. |
| Groq API | Connects the application to an AI language model for answer generation. |

### How the Search Works

The application supports two methods for finding relevant information:

**1. Sentence Transformers**

This method converts sentences into numerical representations called embeddings. These representations help the system identify information that is related in meaning, even when the exact words are different.

The configured model is `all-MiniLM-L6-v2`.

**2. TF-IDF**

This is a simpler, keyword-based method. It compares the words used in the question with the words found in the documents.

If Sentence Transformers is unavailable, the application can use TF-IDF as a fallback.

The system compares the question with the available document sections and uses cosine similarity to identify the closest matches.

## 5. Repository Structure

The project is organized into the following folders and files:

```text
Greenko-Energy-Rag/
|
|-- backend/
|   |-- app.py
|
|-- src/
|   |-- chunker.py
|   |-- embedder.py
|   |-- retriever.py
|   |-- llm.py
|   |-- hallucination_check.py
|   |-- pipeline.py
|
|-- scripts/
|   |-- evaluate.py
|
|-- data/
|   |-- solar_inverter_sop.md
|   |-- wind_turbine_maintenance.md
|   |-- hydro_safety_brief.md
|   |-- grid_operations_note.md
|   |-- questions.csv
|
|-- outputs/
|   |-- evaluation_results.json
|
|-- frontend/
|   |-- index.html
|
|-- requirements.txt
|-- requirements-minimal.txt
|-- Procfile
|-- railway.json
```

### What These Files Do

- `backend/app.py`: Starts the application and handles incoming questions.
- `src/chunker.py`: Reads documents and divides them into searchable sections.
- `src/embedder.py`: Converts document text into numerical representations.
- `src/retriever.py`: Searches for the most relevant document sections.
- `src/llm.py`: Connects to the Groq AI model and prepares answers.
- `src/hallucination_check.py`: Performs a basic check to compare the answer with the retrieved information.
- `src/pipeline.py`: Connects the different steps into one complete question-answering process.
- `scripts/evaluate.py`: Tests the application using prepared questions.
- `frontend/index.html`: Provides the webpage used to interact with the application.
- `outputs/evaluation_results.json`: Stores the results of the evaluation.

## 6. How to Set Up the Application

The application requires Python 3.10 or later.

First, create a virtual environment:

```bash
python -m venv .venv
```

Activate it using the appropriate command.

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

The complete installation includes Sentence Transformers and PyTorch, which may require additional time and storage.

For a lighter installation without PyTorch, use:

```bash
python -m pip install -r requirements-minimal.txt
```

In this case, the application uses the TF-IDF method for searching.

## 7. Configuring the AI Model (Optional)

The application uses the Groq API to connect to a language model.

To enable AI-generated answers:

1. Create a `.env` file using the provided `.env.example` file.
2. Add your Groq API key to the `.env` file.

```dotenv
GROQ_API_KEY=your_key_here
```

The API key should remain private and should not be uploaded to GitHub.

The current code uses `openai/gpt-oss-120b` as its default model name.

**What happens if the API key is not configured?**

The application still works. It can search the documents and return relevant information, but it will not generate a new AI-written answer. Instead, it provides a clearly identified response based on the retrieved document content.

If the Groq request fails, the application also falls back to the retrieved information and reports the relevant status.

## 8. Running the Application

From the main project folder, run:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Once the server starts, open the following address in your browser:

**Application:** http://localhost:8000

**API Documentation:** http://localhost:8000/docs

### Available API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Opens the application webpage. |
| `POST /api/run` | Accepts a question and returns the answer and supporting information. |
| `GET /api/health` | Checks whether the application is running and reports its configuration. |
| `GET /api/demo` | Displays the current application configuration and available endpoints. |

### Example Question

A user can enter:

> What should be checked for inverter derating?

The application searches the available documents, identifies relevant information, and returns the matching content.

The response can include:

- The answer or retrieved-text fallback.
- References to the source documents.
- The document sections used to prepare the response.
- The answer support-check result.
- Whether the AI model generated the answer or the system used a fallback.
- The selected search method and response time.

## 9. Project Evaluation

The project includes an evaluation script that tests the retrieval process using prepared questions.

To run the evaluation:

```bash
python -m scripts.evaluate
```

The script reads questions from `data/questions.csv` and saves the results in:

```text
outputs/evaluation_results.json
```

### Current Evaluation Results

<text> </text>

| Evaluation Item | Result |
|---|---|
| Total test questions | 2 |
| Correct top-source matches | 2 |
| Source attribution accuracy | 100% |
| Embedding method used in the saved test | TF-IDF |
| AI-generated answers in the saved test | 0 |

The saved evaluation shows that the system retrieved the expected source document for both test questions.

However, this result should be understood as a basic functionality test, not proof that the application will perform accurately in real-world situations.

The evaluation uses only two questions and four documents. A much larger and more varied test set would be needed to measure the application's actual performance.

The saved test also did not use the Groq AI model because an API key was not configured. The answers were generated using the retrieved document content rather than an AI model.

## 10. How Does the Application Check Its Answers?

The application includes a basic check to see whether the answer is supported by the retrieved document content.

It compares important words in the answer with words found in the retrieved sections. The configured threshold is 70%.

This helps identify answers that may not have enough overlap with the source material.

However, this is only a basic text-comparison method. It does not fully understand the meaning of a sentence or verify whether every statement is factually correct.

For example, an answer could use similar words but still misunderstand the original instruction. Similarly, a correct answer written using different words could receive a low support score.

Therefore, this check should be treated as an additional warning mechanism, not a guarantee that an answer is correct.

## 11. Does This Project Use an AI Agent?

No. This project uses a **Retrieval-Augmented Generation (RAG) approach**, not an autonomous AI agent.

The application follows a predefined sequence:

1. Receive a question.
2. Search the available documents.
3. Retrieve relevant information.
4. Send the retrieved information to the language model, if configured.
5. Return the answer with its sources.

The system does not independently choose tools, plan multiple actions, or perform operational tasks.

Its purpose is to help users find information from existing documents, not to make decisions or control energy infrastructure.

## 12. Limitations and Future Improvements

The current application is a working demonstration, but it has some limitations.

**Current limitations:**

- **Small document collection:** Only four sample documents are included.
- **Limited evaluation:** Only two prepared questions are used for testing.
- **Basic document size:** The documents are short, so the application does not yet demonstrate searching through large manuals.
- **AI model dependency:** AI-generated answers require a valid Groq API key.
- **Basic answer verification:** The support check compares words rather than deeply understanding the meaning of the answer.
- **No document upload feature:** New documents must be added manually to the `data/` folder, followed by restarting the application.
- **No conversation memory:** Each question is handled independently, without remembering previous questions.
- **No user authentication:** The current demonstration does not include login or document-level access restrictions.
- **No forecasting capability:** The application does not predict renewable-energy generation or calculate actual-versus-predicted generation errors.

**Possible future improvements:**

1. Add more operational documents, including longer manuals and maintenance reports.
2. Increase the number and variety of test questions.
3. Improve the evaluation process to measure both retrieval quality and answer accuracy.
4. Introduce stronger methods to verify whether each answer is genuinely supported by its source.
5. Add document-upload functionality.
6. Introduce user authentication and document access controls.
7. Explore conversation history and more advanced question-answering capabilities.

## 13. Deployment

The repository includes configuration files for deploying the application using Railway.

However, having deployment configuration does not mean that the application has already been successfully deployed.

To deploy it, the required environment variables, including `GROQ_API_KEY` if AI-generated answers are needed, must be configured in Railway.

After deployment, the application should be tested using its health endpoint and by submitting sample questions.

No live deployment URL or successful deployment result is currently included in the saved project evaluation.

## 14. Important Safety Considerations

This application is intended to demonstrate how AI can make operational information easier to access.

It should not replace approved operating procedures, official maintenance instructions, or decisions made by qualified energy-sector professionals.

Users should always refer to the original documents and follow the organization's approved safety procedures before taking any operational action.

## Conclusion

The Energy Operations RAG Assistant demonstrates how document search and generative AI can be combined to make information easier to access.

Instead of manually searching through multiple operational documents, users can ask questions in natural language and receive relevant information along with its source references.

The project demonstrates the basic working principles of document processing, information retrieval, AI-assisted question answering, and answer-support checking.

Although the current implementation is limited in document size and evaluation coverage, it provides a foundation that can be improved with larger document collections, more comprehensive testing, and stronger answer-verification methods.
