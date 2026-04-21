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

---

## Data Sources

The system operates on structured viral datasets:

```
data/
├── viral_taxo.csv        # Viral taxonomy (NCBI/SRA)
└── virushostdb.tsv       # Virus–host occurrences (SRA)
```

### Viral Taxonomy Dataset (`viral_taxo.csv`)

* Taxonomic hierarchy (species, genus, family, order, etc.)
* Associated genomic and ecological metadata
* Derived from SRA taxonomy records
* Key columns: `TAX_ID`, `ORGANISM_NAME`, `RANK`, `GENUS_NAME`, `FAMILY_NAME`, `DIVISION_NAME`

### Virus–Host Dataset (`virushostdb.tsv`)

* Virus–host occurrence records with geographic coordinates
* Derived from VirusHostDB ([genome.jp](https://www.genome.jp/virushostdb/)) and SRA
* Key columns: `DATA_ID`, `VIRAL_TAX_ID`, `VIRAL_SPECIES`, `TAX_ID` (host), `lon`, `lat`
* Host names are resolved via `TAX_ID` join with `viral_taxo.csv` — the `SPECIES_NAME` column is intentionally sparse

All quantitative results originate strictly from these datasets.

---

## Agent Architecture

The agent uses a controlled tool-calling loop powered by the **Albert API** (French government sovereign AI infrastructure), using the `gpt-oss-120b` model via an OpenAI-compatible endpoint.

### Tools

#### `query_dataframe`

Executes validated pandas code on the dataset.

* Returns structured DataFrame output
* Restricts access to known columns
* Enforces assignment to a `result` variable
* Result cache prevents redundant identical calls

#### `create_visualization`

Generates interactive Plotly figures.

* Requires explicit `fig` assignment
* Produces deterministic visual outputs

#### `create_map`

Generates geographic Plotly maps from occurrence coordinates.

* Triggered automatically for spatial/distribution questions
* Uses `lon`/`lat` columns from the virus–host dataset

#### `wikipedia_search`

Retrieves biological summaries from Wikipedia.

* Plain-text extraction only
* Character-limited responses (configurable)
* Explicit source URLs returned
* Fuzzy matching for approximate name resolution

#### `pubmed_search`

Retrieves scientific article abstracts from PubMed.

* Returns title, authors, journal, year, abstract, DOI, PMID
* Only confirmed PMIDs from actual tool calls are allowed in responses
* Hallucinated PMIDs are automatically detected and stripped before display

---

## Scientific Constraints

The AI operates under strict rules enforced through the system prompt and post-processing:

* No invention of taxa, species counts, or biological claims
* No implicit or hidden knowledge — all statements must be grounded in data or tool output
* No speculative interpretation
* **Host lookup is two-step**: host name → `TAX_ID` via `viral_taxo.csv`, then `TAX_ID` match in the occurrence table
* **PMID hallucination guard**: a whitelist of real PMIDs is built from actual `pubmed_search` calls; any PMID absent from this whitelist is stripped from the final response and logged as a warning
* If information is unavailable, the agent states explicitly:

> "This information is not available in the current dataset or sources."

---

## Capabilities

* Natural language querying of viral taxonomy and virus–host relationships
* Two-step host resolution (name → TAX_ID → occurrence records)
* Aggregation, comparative analysis, species/genus/family counts
* Interactive Plotly visualizations and geographic maps
* Wikipedia and PubMed integration for biological context
* Error reporting system (user-facing feedback button)
* Full session logging with tool call tracing

---

## Deployment

### Requirements

* Python 3.9+
* An **Albert API key** ([albert.api.etalab.gouv.fr](https://albert.api.etalab.gouv.fr))
* Python packages:

```
streamlit
pandas
numpy
requests
plotly
```

### Configuration

Create `.streamlit/secrets.toml`:

```toml
ALBERT_API_KEY = "your_key_here"

# Optional access protection
PASSWORD_ENABLED = true
ACCESS_CODE = "your_access_code"
```

### Commands

```bash
pip install streamlit pandas numpy requests plotly

streamlit run app_albert.py
```

---

## Example Research Queries

* "Summarize Orthopoxvirus (family, genus, species count)"
* "List virus families with more than 100 recorded species"
* "Compare species counts between Orthomyxoviridae and Coronaviridae"
* "Show a pie chart of genus distribution in Poxviridae"
* "World distribution of Poxviridae"
* "What viruses infect Bos taurus?"
* "Tell me more about feline viruses"

---

## Transparency & Reproducibility

* Executed pandas code is visible in the "Sources" expander of each response
* Generated visualizations are deterministic given the same dataset
* Wikipedia and PubMed URLs are explicitly displayed per response
* Tool calls are limited, numbered, and fully traced in logs (`agent_YYYY-MM.log`)
* All PMID citations are verified against actual tool call output — hallucinated PMIDs are logged and removed
* Error reports (question, answer, executed code, log excerpt) can be submitted via the in-app feedback button

---

## ⚠️ Disclaimer

This system is intended for exploratory and research support purposes only.
All outputs should be independently verified before use in scientific or medical contexts.
The agent may miss viruses absent from the SRA-derived dataset, and dataset coverage does not reflect epidemiological prevalence or clinical severity.

---

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).