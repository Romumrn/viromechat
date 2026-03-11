# 🦠 Virus Dataset AI Agent

## Project Context

This project is developed within the framework of **SHAPE-Med@Lyon** and contributes to the structuring research initiative [**Virome@tlas**](https://www.shape-med-lyon.fr/projets/structurants-vague-1/virometlas/).

*Virome@tlas* aims to build an integrated digital platform for large-scale exploration and surveillance of the global virosphere. The project leverages publicly available sequencing data to analyze virus diversity, virus–host interactions, and ecological distribution patterns within a transdisciplinary **One Health** framework spanning human, animal, and environmental health.

The Virus Dataset AI Agent supports this effort by providing a controlled, reproducible interface for structured exploration of viral taxonomy and virus–host datasets. It is designed as a research companion tool that combines:

* Deterministic dataset querying
* Transparent visualization generation
* Controlled external knowledge retrieval
* Strict grounding of all biological statements

By constraining the language model to explicit data sources and documented tool calls, the system aims to reduce hallucination risk while preserving interpretability and scientific traceability.

##  Data Sources

The system operates on structured viral datasets:

```
data/
├── viral_taxo.csv
└── virushostdb.tsv
```

### Viral Taxonomy Dataset (`viral_taxo.csv`)

* Taxonomic hierarchy (species, genus, family, order, etc.)
* Associated genomic and metadata attributes
* Derived from SRA taxonomy records

### Virus–Host Dataset (`virushostdb.tsv`)

* Virus–host relationships
* Derived from VirusHostDB ([genome.jp](https://www.genome.jp/virushostdb/)) 

All quantitative results originate strictly from these datasets.


## Agent Architecture

The agent uses a controlled tool-calling loop powered by a local LLM via Ollama.

### tools:

#### `query_dataframe`

Executes validated pandas code on the dataset.

* Returns structured DataFrame output
* Restricts access to known columns
* Enforces assignment to a `result` variable
* Provides reproducible previews

####  `create_visualization`

Generates Plotly figures.

* Only permitted after structured queries
* Requires explicit `fig` assignment
* Produces deterministic visual outputs

####  `wikipedia_search`

Retrieves biological summaries from Wikipedia.

* Plain-text extraction only
* Character-limited responses
* Explicit source URLs returned
* No interpretation layer added


##  Scientific Constraints

The AI operates under strict rules trought the prompt and parameters:

* No invention of taxa, species counts, or biological claims
* No use of implicit or hidden knowledge
* No speculative interpretation
* All biological statements must originate from:

  * dataset queries, or
  * documented Wikipedia tool calls
* If information is unavailable, the agent states explicitly:

> “This information is not available in the current dataset or sources.”

This constraint framework is designed to reduce hallucination risk and increase reproducibility.


##  Capabilities

* Natural language querying of viral taxonomy
* Aggregation and comparative analysis
* Species, genus, and family counts
* Structured filtering and grouping
* Interactive Plotly visualizations
* Virus–host relationship exploration
* Explicit source documentation
* and more ...

## Local Execution

The system runs entirely locally:

* LLM inference via Ollama
* No external LLM APIs required
* Wikipedia access via public API
* All dataset processing performed locally

This enables reproducibility and data control in research environments.


## Installation

### Requirements

* Python 3.9+
* Ollama installed and running
* A tool-capable model (e.g., `gpt-oss`, check ollama webpage tools )
* Python packages:

  * streamlit
  * pandas
  * numpy
  * requests
  * plotly


### Commands

```bash
pip install streamlit pandas numpy requests plotly

# Pull a Model
ollama pull gpt-oss

# Run the Application
streamlit run app.py
```


## Example Research Queries

* “Summarize Orthopoxvirus (family, genus, species count)”
* “List virus families with more than 100 recorded species”
* “Compare species counts between Orthomyxoviridae and Coronaviridae”
* “Show a pie chart of genus distribution in Poxviridae”
* “Give me hosts of Orthopoxvirus Abatino”


## Transparency & Reproducibility

* Executed pandas code can be inspected
* Generated visualizations are deterministic
* Wikipedia URLs are explicitly displayed
* Tool calls are limited and traceable
* Dataset access is column-restricted

This architecture enables auditability and reproducible AI-assisted analysis.


## ⚠️ Disclaimer

This system is intended for exploratory and research support purposes only.
All outputs should be independently verified before use in scientific or medical contexts.


## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).