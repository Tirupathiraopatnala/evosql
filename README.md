# 🧬 EvoSQL: Self‑Evolving SQL Query Optimizer

## 👥 Team Details

- **Team Name:** The Evolvers  
- **Members:**  
  - Tirupathirao Patnala  
  - Chandra Javvaji  
- **Domain Category:** Multi-Agent Systems / Evolutionary AI  
- **Demo Video:** [SharePoint URL of your MVP demo]  

---

## 🎯 Problem Statement

Slow queries in Azure Synapse waste compute resources and delay critical insights.  
Manual tuning is:

- Time‑consuming  
- Dependent on scarce expertise  
- Rarely discovers optimal rewrites  

Traditional optimizers are static and cannot adapt to changing data or workload patterns.

---

## 💡 Solution Overview

**EvoSQL** is an autonomous multi‑agent system where AI agents, each with a distinct optimization strategy (encoded in a **genome**), compete to rewrite SQL queries.  
Agents call **Azure OpenAI** to generate rewrites guided by schema metadata and generational feedback.  
Rewrites are validated for correctness and executed on **Azure Synapse**.  
The **fittest** agents (fastest correct rewrites) survive and **reproduce** via crossover and mutation, driving continuous improvement across generations.  

The system:

- Discovers non‑intuitive rewrite strategies  
- Adapts automatically to data growth and workload changes  
- Provides explainable evolution (what changed and why it helped)  

---

## 🏗 Architecture

### Components

- **User Interface** – Streamlit dashboard for input, live progress, and results  
- **Evolution Engine** – Orchestrates generations, manages population, selection, reproduction  
- **Strategy Agent** – Holds a genome, requests LLM rewrites, tracks status and fitness  
- **Genome** – Encodes optimization biases (predicate pushdown, shuffle avoidance, etc.)  
- **Synapse Client** – Executes SQL, retrieves `EXPLAIN` plans, computes result checksums  
- **Schema Extractor** – Queries Synapse metadata (distributions, indexes, partitions) for prompt context  
- **Safety Governor** – Validates SQL against forbidden operations and schema preservation  
- **Query Validator** – Compares result sets (row count, column count, checksum) for semantic equivalence  
- **Fitness Evaluator** – Computes fitness = execution time in seconds (lower = better)  
- **Azure OpenAI** – LLM that generates rewritten SQL based on strategy instructions + context  

### Flow

1. User submits SQL query via Streamlit.  
2. Evolution Engine executes baseline query → captures metrics + checksum.  
3. **For each generation:**  
   - Agents (with unique genomes) request rewrites from Azure OpenAI.  
   - Rewrites are safety‑checked and schema‑validated.  
   - Valid rewrites are executed on Synapse; execution time and result sets are captured.  
   - Results are validated against baseline; valid agents receive fitness score.  
   - Top 2 agents become elites (carried forward unchanged).  
   - Remaining population filled via crossover + mutation of elite genomes.  
   - Feedback (winner + failed strategies) is injected into next generation’s prompts.  
4. After all generations, the fastest correct rewrite is presented as the winner.  

📁 **Architecture Diagram:** `/architecture/architecture.png` (to be added)

---

## 🛠 Tech Stack

| Layer          | Technology                         |
|----------------|------------------------------------|
| Backend        | Python 3.11                        |
| Frontend       | Streamlit                          |
| AI Model       | Azure OpenAI GPT-4                 |
| Database       | Azure Synapse Dedicated SQL Pool   |
| Orchestration  | Custom Python (Evolution Engine)   |

---

## 📂 Project Structure

evosql/
├── agent.py
├── app.py
├── debug_logger.py
├── evolution.py
├── evosql_debug.log
├── fitness.py
├── genome.py
├── requirements.txt
├── safety.py
├── schema_extractor.py
├── synapse_client.py
└── validator.py

---

## ⚙️ Setup Instructions

## 1️ Verify Required Software

- Programming Language: <Python / Node / Java / etc>
- Required Version: Exact Version
- Package Manager: <pip / npm / etc>

### 1️⃣ Clone Repository

```bash
git clone https://github.com/dataforge-ai/autonomous-data-agent
cd autonomous-data-agent
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create `.env` file from `.env.example`

Example:

```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_MODEL=gpt-4

SYNAPSE_SERVER=syn-your-server.sql.azuresynapse.net
SYNAPSE_DATABASE=your_db
SYNAPSE_USERNAME=your_user
SYNAPSE_PASSWORD=your_password
```

---

## ▶️ Entry Point

Run the application:

```bash
streamlit run src/main.py
```

Application will start at:

```
http://localhost:8501
```

---

## 🔄 Application Flow

1. **Input** – Paste a slow SQL query into the text area and click **Start Evolution**.  
2. **Baseline** – The original query is executed on Synapse; metrics and a result checksum are captured.  
3. **Generations** – The evolution loop runs (configurable population size & generations):  
   - **Rewriting** – Each agent (except elites) calls Azure OpenAI to generate a rewritten SQL based on its genome, schema metadata, and previous feedback.  
   - **Safety Check** – Rewrites are scanned for forbidden keywords and schema preservation.  
   - **Execution** – Valid rewrites are executed; execution time and result rows are recorded.  
   - **Validation** – Result sets are compared to baseline (row count, column count, checksum).  
   - **Fitness** – Valid agents receive fitness = execution time (seconds).  
   - **Selection & Reproduction** – Top 2 agents become elites; children are created via crossover + mutation.  
   - **Feedback** – Winner and failed strategies are summarized and fed into next generation’s prompts.  
4. **Completion** – The fastest correct rewrite across all generations is displayed, along with improvement %, diff summary, and plan metrics.

---

## 🧪 How to Test

Paste any complex SQL that runs on your Synapse instance.  
The system will attempt to optimize it across multiple generations.

**Example query snippet** (from sample):
```sql
SELECT ...
FROM fact_sales f
JOIN dim_product p ON f.product_id = p.product_id
WHERE p.category = 'Electronics'
  AND f.order_date >= '2023-01-01'
```
---

## ⚠️ Known Limitations

- Requires active Azure OpenAI and Synapse connections.  
- Feedback is prompt‑based; the LLM is not fine‑tuned.  
- Checksum validation is currently disabled due to precision mismatches (will be re‑enabled after normalization improvements).  

## 🔮 Future Improvements

- Integrate **Azure Fabric** for automated pipeline deployment.  
- Add **reinforcement learning** loop to adjust genome mutation rates based on historical success.  
- Support for **multiple database platforms** (Snowflake, BigQuery).  
- **Multi‑user** session management and result persistence.  
- **Explainability enhancements**: show which parts of the genome contributed most to improvements.


