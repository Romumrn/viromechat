import os
import re
import json
import traceback
import requests
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from xml.etree import ElementTree as ET
from fastmcp import FastMCP
from config import TAXO_DB_PATH, DEFAULT_PREVIEW_ROWS, DEFAULT_WIKIPEDIA_LIMIT, MCP_ENV_PATH, load_env_file

# ==================== BASE_DIR (défini immédiatement) ==================== #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== CHEMINS DES FICHIERS DE DESCRIPTION ==================== #
HOST_COL_DESC_PATH = os.path.join(BASE_DIR, "data", "v@_columns_description.csv")
TAXO_COL_DESC_PATH = os.path.join(BASE_DIR, "data", "TAXONOMY_columns_description.json")

# Variables globales pour les descriptions (peuplées par load_column_descriptions,
# exposées aux clients MCP via les resources définies plus bas)
host_col_descriptions = {}
taxo_col_descriptions = {}


def load_column_descriptions():
    """Charge les descriptions sémantiques des colonnes depuis les fichiers."""
    global host_col_descriptions, taxo_col_descriptions

    if os.path.exists(HOST_COL_DESC_PATH):
        df_desc = pd.read_csv(HOST_COL_DESC_PATH)
        # Colonnes: column_name, description, Type
        host_col_descriptions = df_desc.set_index('column_name').to_dict(orient='index')
        print(f"[server_mcp] Loaded {len(host_col_descriptions)} host column descriptions.")
    else:
        print(f"[server_mcp] Warning: host column description file not found: {HOST_COL_DESC_PATH}")

    if os.path.exists(TAXO_COL_DESC_PATH):
        with open(TAXO_COL_DESC_PATH, 'r', encoding='utf-8') as f:
            taxo_col_descriptions = json.load(f)
        print(f"[server_mcp] Loaded taxonomy schema ({len(taxo_col_descriptions.get('columns', []))} columns).")
    else:
        print(f"[server_mcp] Warning: taxonomy column description file not found: {TAXO_COL_DESC_PATH}")


mcp = FastMCP("ViromeChatMCP")

load_env_file(MCP_ENV_PATH)

# ==================== DATA SOURCES (from .env.mcp / environment) ==================== #
S3_ENDPOINT = os.environ.get("ENDPOINT", "")
S3_ACCESS_KEY = os.environ.get("ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("SECRET_KEY", "")
S3_BUCKET = os.environ.get("BUCKET", "")
S3_HOST_KEY = os.environ.get("VIRAL_HOST_DATASET", "*.parquet")
S3_REGION = os.environ.get("REGION", "fr")
S3_URL_STYLE = os.environ.get("S3_URL_STYLE", "path")

S3_HOST_PARQUET_PATH = f"s3://{S3_BUCKET}/{S3_HOST_KEY}"


# ==================== GLOBAL DATA CONTEXT ==================== #
class DataContext:
    def __init__(self):
        self.df_taxo: pd.DataFrame | None = None
        self.host_con: duckdb.DuckDBPyConnection | None = None
        self.host_columns: list[str] | None = None
        self.last_host_result: pd.DataFrame | None = None

    def load_taxo(self, taxo_path: str, **read_csv_kwargs):
        self.df_taxo = pd.read_csv(taxo_path, **read_csv_kwargs)

    def connect_host_s3(
        self,
        parquet_path: str,
        access_key: str,
        secret_key: str,
        region: str,
        endpoint: str = "",
        url_style: str = "path",
        use_ssl: bool = True,
    ):
        con = duckdb.connect(database=":memory:")
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        # Needed for ST_X/ST_Y — the `geometry` column is a native GEOMETRY
        # type (WGS84 points), not plain lat/lon columns.
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")

        # A bit more slack than the 30s default: this dataset lives behind a
        # slow S3-compatible gateway, and a single filtered query can involve
        # several sequential range requests.
        con.execute("SET http_timeout=60;")
        con.execute("SET http_retries=3;")

        if region:
            con.execute(f"SET s3_region='{region}';")
        if endpoint:
            con.execute(f"SET s3_endpoint='{endpoint}';")
            con.execute(f"SET s3_url_style='{url_style}';")
        con.execute(f"SET s3_use_ssl={'true' if use_ssl else 'false'};")
        if access_key and secret_key:
            con.execute(f"SET s3_access_key_id='{access_key}';")
            con.execute(f"SET s3_secret_access_key='{secret_key}';")

        con.execute(f"CREATE OR REPLACE VIEW host AS SELECT * FROM read_parquet('{parquet_path}');")

        self.host_con = con
        self.host_columns = [row[0] for row in con.execute("DESCRIBE host").fetchall()]


ctx = DataContext()


# ==================== INITIALIZATION (called at server startup) ==================== #
def initialize_datasets():
    """Charge les données au démarrage du serveur. En cas d'échec, le serveur s'arrête."""
    print("[server_mcp] Loading taxonomy...")
    ctx.load_taxo(TAXO_DB_PATH)
    print(f"[server_mcp] df_taxo loaded: {ctx.df_taxo.shape} ({TAXO_DB_PATH})")

    print("[server_mcp] Loading column descriptions...")
    load_column_descriptions()

    print("[server_mcp] Connecting to S3 / Parquet dataset...")
    ctx.connect_host_s3(
        S3_HOST_PARQUET_PATH,
        S3_ACCESS_KEY,
        S3_SECRET_KEY,
        S3_REGION,
        endpoint=S3_ENDPOINT,
        url_style=S3_URL_STYLE,
    )
    print(
        f"[server_mcp] df_host connected via DuckDB/S3: {S3_HOST_PARQUET_PATH} "
        f"(endpoint={S3_ENDPOINT or 'AWS default'}, {len(ctx.host_columns)} columns, "
        f"not loaded into memory)"
    )


# ==================== RESPONSE HELPERS ==================== #

def _ok(content: str, artifacts: list | None = None) -> dict:
    return {"success": True, "content": content, "artifacts": artifacts or []}


def _fail(content: str) -> dict:
    return {"success": False, "content": content, "artifacts": []}


def _check_figure_has_data(fig: go.Figure) -> bool:
    """
    Returns True if the figure contains at least one trace with actual data points.
    Catches empty figures produced from empty dataframes.
    """
    for trace in fig.data:
        x = getattr(trace, "x", None)
        lat = getattr(trace, "lat", None)
        lon = getattr(trace, "lon", None)
        values = getattr(trace, "values", None)  # pie chart

        if lat is not None and lon is not None:
            if hasattr(lat, "__len__") and len(lat) > 0:
                return True
        if x is not None:
            if hasattr(x, "__len__") and len(x) > 0:
                return True
        if values is not None:
            if hasattr(values, "__len__") and len(values) > 0:
                return True
    return False


def _check_hover_has_column(fig: go.Figure, column: str) -> bool:
    """
    Returns True if `column` appears in some trace's hovertemplate — Plotly
    Express renders each `hover_data` column as a literal `column=%{...}`
    label in the template, so this reliably detects whether a given column
    was actually included in hover_data (not just present in the dataframe).
    """
    for trace in fig.data:
        tmpl = getattr(trace, "hovertemplate", None)
        if tmpl and column in tmpl:
            return True
    return False


def _check_data_not_empty(env: dict):
    """
    After exec(), inspect variables named 'data'/'result'/etc. in env.
    If a DataFrame is found and is empty, return an error message.
    Returns None if everything looks fine.
    """
    for varname in ("data", "result", "df_filtered", "df"):
        val = env.get(varname)
        if isinstance(val, pd.DataFrame) and val.empty:
            return (
                f"Error: the DataFrame '{varname}' used to build the figure is empty (0 rows). "
                "The search term returned no results in the dataset. "
                "Try a broader term, the full scientific species name instead of an acronym, "
                "or use str.contains() with a partial name match."
            )
    return None


def _figure_artifact(fig: go.Figure) -> dict:
    """Build the standard {"type": "plotly", "figure": {...}} artifact from a Plotly figure."""
    return {"type": "plotly", "figure": json.loads(fig.to_json())}


def _figure_title(fig: go.Figure) -> str | None:
    return fig.layout.title.text if fig.layout.title else None


def _table_result(result: pd.DataFrame, preview_rows: int) -> dict:
    """Build the standard {success, content, artifacts:[{"type": "table", ...}]} response
    for a query that produced a pandas DataFrame."""
    total = len(result)
    preview_df = result.head(preview_rows)
    preview_text = preview_df.to_string(index=False)
    if total > preview_rows:
        preview_text += f"\n... and {total - preview_rows} more rows"

    rows = json.loads(preview_df.to_json(orient="records", date_format="iso"))
    content = f"Query returned {total} row{'s' if total != 1 else ''}.\nPreview:\n{preview_text}"
    return _ok(content, [{
        "type": "table",
        "rows": rows,
        "columns": list(result.columns),
        "total_rows": total,
    }])


_FORBIDDEN_SQL_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|COPY|PRAGMA|EXPORT|IMPORT|CALL)\b",
    re.IGNORECASE,
)


# ==================== MCP RESOURCES ==================== #

@mcp.resource(
    "resource://datasets/host/schema",
    name="Host dataset schema",
    description=(
        "Column-by-column description of the virus-host observation table (`host`), "
        "which lives as a Parquet dataset on S3. Sourced from data/v@_columns_description.csv. "
        "Read this once to learn the exact column names, types, and semantics before writing "
        "SQL against `host` (via query_host_sql) or pandas code against df_host."
    ),
    mime_type="application/json",
)
def host_schema_resource() -> str:
    return json.dumps(host_col_descriptions, ensure_ascii=False)


@mcp.resource(
    "resource://datasets/taxonomy/schema",
    name="Taxonomy dataset schema",
    description=(
        "Column-by-column description of the NCBI taxonomy dataset (df_taxo), including the "
        "dataset's primary key and row definition. Sourced from "
        "data/TAXONOMY_columns_description.json. Read this once to learn the exact column "
        "names and semantics before writing pandas code against df_taxo."
    ),
    mime_type="application/json",
)
def taxonomy_schema_resource() -> str:
    return json.dumps(taxo_col_descriptions, ensure_ascii=False)


# ==================== MCP TOOLS ==================== #

@mcp.tool
def wikipedia_search(search_term: str, wikipedia_limit: int = DEFAULT_WIKIPEDIA_LIMIT) -> dict:
    """
    Search Wikipedia for biological or scientific information.

    First attempts an exact page match, then falls back to the closest
    relevant page if no exact match is found. Useful when searching for
    precise scientific names (e.g. virus species, taxa) that may not have
    a dedicated Wikipedia page — the tool automatically returns the most
    relevant related article instead.

    Args:
        search_term: The scientific or biological term to search for
            (e.g. 'Sprivivirus cyprinus', 'SARS-CoV-2', 'Rabies lyssavirus').
        wikipedia_limit: Maximum number of characters to return from the
            article extract.
    """
    term = re.sub(r"[^\w\s\-]", "", search_term.strip())

    api_url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "VirusAgent/1.0"}

    def _build_content(title: str, extract: str, url: str, fuzzy: bool, original: str) -> str:
        if len(extract) > wikipedia_limit:
            extract = extract[:wikipedia_limit] + "... [truncated]"
        content = f"**{title}**\n\n{extract}\n\n🔗 {url}"
        if fuzzy:
            content += f"\n\n> ⚠️ No exact page found for *{original}* — showing closest match."
        return content

    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "redirects": 1, "titles": term,
    }

    try:
        r = requests.get(api_url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        pages = data.get("query", {}).get("pages", {})
        if pages:
            page = next(iter(pages.values()))
            if "missing" not in page:
                page_title = page.get("title", term)
                extract = page.get("extract", "")
                url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                content = _build_content(page_title, extract, url, fuzzy=False, original=search_term)
                return _ok(content, [{"type": "url", "url": url}])

    except requests.RequestException:
        pass

    search_params = {
        "action": "query", "format": "json", "list": "search",
        "srsearch": term, "srlimit": 1,
    }

    try:
        sr = requests.get(api_url, params=search_params, headers=headers, timeout=10)
        sr.raise_for_status()
        search_results = sr.json().get("query", {}).get("search", [])

        if not search_results:
            return _fail(f"No Wikipedia article found for {search_term}")

        best_match_title = search_results[0]["title"]

        page_params = {
            "action": "query", "format": "json", "prop": "extracts",
            "explaintext": 1, "redirects": 1, "titles": best_match_title,
        }

        pr = requests.get(api_url, params=page_params, headers=headers, timeout=10)
        pr.raise_for_status()
        pages = pr.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))

        if "missing" in page:
            return _fail(f"No Wikipedia article found for {search_term}")

        page_title = page.get("title", best_match_title)
        extract = page.get("extract", "")
        url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        content = _build_content(page_title, extract, url, fuzzy=True, original=search_term)
        return _ok(content, [{"type": "url", "url": url}])

    except requests.RequestException:
        return _fail(f"Wikipedia request failed for {search_term}")


@mcp.tool
def pubmed_search(query: str, max_results: int = 5) -> dict:
    """
    Search PubMed for scientific articles and retrieve abstracts.

    Returns the most relevant articles matching the query with their
    titles, abstracts, authors, and publication details. Useful for
    finding recent research, clinical studies, or detailed scientific
    information about viruses, diseases, or biological topics. The search
    uses PubMed's relevance sorting to find the most pertinent articles.

    Args:
        query: Search query for PubMed (e.g. 'SARS-CoV-2 spike protein',
            'influenza vaccine efficacy', 'rabies virus transmission').
        max_results: Maximum number of articles to retrieve (max recommended: 5).
    """
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed", "term": query, "retmode": "json",
            "retmax": max_results, "sort": "relevance",
        }

        search_response = requests.get(
            search_url, params=search_params, headers={"User-Agent": "VirusAgent/1.0"}
        )

        if search_response.status_code != 200:
            return _fail(f"PubMed search failed with status {search_response.status_code}")

        search_data = search_response.json()
        id_list = search_data.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return _fail(f"No articles found on PubMed for '{query}'")

        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"}

        fetch_response = requests.get(
            fetch_url, params=fetch_params, headers={"User-Agent": "VirusAgent/1.0"}
        )

        if fetch_response.status_code != 200:
            return _fail("Failed to fetch article details from PubMed")

        root = ET.fromstring(fetch_response.content)
        articles = []

        for article in root.findall(".//PubmedArticle"):
            pmid = article.find(".//PMID")
            pmid_text = pmid.text if pmid is not None else "N/A"

            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title available"

            abstract_elem = article.find(".//Abstract/AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"

            authors = []
            for author in article.findall(".//Author"):
                last_name = author.find("LastName")
                fore_name = author.find("ForeName")
                if last_name is not None:
                    name = last_name.text
                    if fore_name is not None:
                        name = f"{fore_name.text} {name}"
                    authors.append(name)

            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "Unknown journal"

            year_elem = article.find(".//PubDate/Year")
            year = year_elem.text if year_elem is not None else "Unknown year"

            doi_elem = article.find(".//ArticleId[@IdType='doi']")
            doi = doi_elem.text if doi_elem is not None else None

            articles.append({
                "pmid": pmid_text, "title": title, "abstract": abstract,
                "authors": authors[:3], "journal": journal, "year": year,
                "doi": doi, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_text}/",
            })

        parts = [f"Found {len(articles)} article(s) on PubMed for '{query}':\n"]
        pmids = []
        for i, article in enumerate(articles, 1):
            if article["pmid"].isdigit():
                pmids.append(int(article["pmid"]))

            authors_str = ", ".join(article["authors"]) if article["authors"] else "Unknown authors"
            doi_str = f"DOI: {article['doi']}" if article["doi"] else ""

            parts.append(
                f"\n--- Article {i} ---\n"
                f"**{article['title']}**\n"
                f"Authors: {authors_str} et al.\n"
                f"Journal: {article['journal']} ({article['year']})\n"
                f"PMID: {article['pmid']} {doi_str}\n\n"
                f"Abstract:\n{article['abstract']}\n"
                f"🔗 {article['url']}"
            )

        return _ok("\n".join(parts), [{"type": "pubmed", "pmids": pmids}])

    except Exception as e:
        return _fail(f"Error searching PubMed: {str(e)}\n{traceback.format_exc()}")


_NCBI_TAXONOMY_SYNONYM_TAGS = (
    "Synonym", "Acronym", "EquivalentName", "GenbankSynonym",
    "GenbankAcronym", "GenbankCommonName", "CommonName", "Inpart",
)


@mcp.tool
def ncbi_taxonomy_search(name: str) -> dict:
    """
    Resolve a taxon name against the NCBI Taxonomy database — the
    authoritative source for scientific names, ranks, and classification of
    any organism (virus, host, or otherwise). Matches on scientific names,
    common names, acronyms, and synonyms alike (e.g. 'HIV', 'MPOX').

    Use this FIRST whenever you are unsure of:
    - the exact scientific name behind an acronym or common name
      (e.g. 'HIV' resolves to 'Human immunodeficiency virus 1', genus
      'Lentivirus', species 'Lentivirus humimdef1' in its lineage),
    - whether a name is a species, genus, family, etc. — check the `rank`
      of the returned match,
    - the full taxonomic lineage of an organism.

    Then use the resolved scientific name (not the acronym) in
    query_host_sql, query_dataframe, wikipedia_search, or pubmed_search.

    A name can match multiple taxa (e.g. 'HIV' matches several related
    entries) — all matches are returned, ranked by NCBI relevance.

    Args:
        name: An organism/taxon name, acronym, or common name to resolve
            (e.g. 'HIV', 'MPOX', 'Orthopoxvirus', 'Poxviridae').
    """
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {"db": "taxonomy", "term": name, "retmode": "json", "retmax": 5}
        sr = requests.get(search_url, params=search_params, headers={"User-Agent": "VirusAgent/1.0"}, timeout=15)
        sr.raise_for_status()
        id_list = sr.json().get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return _fail(f"No NCBI Taxonomy entry found for '{name}'.")

        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fr = requests.get(
            fetch_url, params={"db": "taxonomy", "id": ",".join(id_list), "retmode": "xml"},
            headers={"User-Agent": "VirusAgent/1.0"}, timeout=15,
        )
        fr.raise_for_status()

        root = ET.fromstring(fr.content)
        taxa = []
        # Direct children only — nested <Taxon> elements also appear inside
        # each result's <LineageEx> (one per ancestor rank) and must be
        # excluded, or a single query balloons into dozens of "matches".
        for taxon in root.findall("Taxon"):
            other_names = taxon.find("OtherNames")
            synonyms = []
            if other_names is not None:
                for tag in _NCBI_TAXONOMY_SYNONYM_TAGS:
                    synonyms.extend(el.text for el in other_names.findall(tag) if el.text)

            taxa.append({
                "tax_id": taxon.findtext("TaxId"),
                "name": taxon.findtext("ScientificName"),
                "rank": taxon.findtext("Rank") or "no rank",
                "division": taxon.findtext("Division") or "",
                "lineage": taxon.findtext("Lineage") or "",
                "synonyms": synonyms,
            })

        if not taxa:
            return _fail(f"No NCBI Taxonomy entry found for '{name}'.")

        parts = [f"Found {len(taxa)} NCBI Taxonomy match(es) for '{name}':\n"]
        for i, t in enumerate(taxa, 1):
            syn_str = f"\nSynonyms/acronyms: {', '.join(t['synonyms'][:6])}" if t["synonyms"] else ""
            parts.append(
                f"\n--- Match {i} ---\n"
                f"**{t['name']}** (TaxID {t['tax_id']}, rank: {t['rank']}, division: {t['division']})\n"
                f"Lineage: {t['lineage']}{syn_str}\n"
                f"🔗 https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={t['tax_id']}"
            )

        top = taxa[0]
        top_url = f"https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={top['tax_id']}"
        return _ok("\n".join(parts), [{"type": "ncbi_taxonomy", "url": top_url, "tax_id": top["tax_id"]}])

    except Exception as e:
        return _fail(f"Error querying NCBI Taxonomy: {str(e)}\n{traceback.format_exc()}")


_BARE_SELECT_STAR = re.compile(r"SELECT\s+\*\s+FROM", re.IGNORECASE)


@mcp.tool
def query_host_sql(sql: str, preview_rows: int = DEFAULT_PREVIEW_ROWS) -> dict:
    """
    Run a read-only SQL SELECT against the virus-host observation table
    (`host`), which lives as a Parquet dataset on S3 and is NEVER fully
    loaded into memory. DuckDB pushes your filter down to S3: it only reads
    the Parquet row-groups and columns your query actually needs.

    This is the REQUIRED first step before using query_dataframe,
    create_visualization, or create_map with df_host: those tools then
    operate on the (small, already-filtered) result of your last
    query_host_sql call, exactly like a normal pandas DataFrame.

    Read the `resource://datasets/host/schema` resource for the exact column
    names, types, and semantics before writing your query — column names are
    case-sensitive and guessing them (or the acronym instead of the full
    scientific name) is the most common cause of a 0-row or slow query.

    RULES:
    - SELECT-only. INSERT/UPDATE/DELETE/DDL/PRAGMA/etc. are rejected.
    - NEVER `SELECT *`. Always list the exact columns you need. `host` has
      ~65 columns including a heavy `geometry` blob; pulling all of them
      for thousands of matching rows over S3 is what causes timeouts.
    - `geometry` is a native GEOMETRY point (WGS84), not plain lat/lon
      columns. To get coordinates, extract them explicitly:
      `ST_X(geometry) AS lon, ST_Y(geometry) AS lat` — never select
      `geometry` itself.
    - ALWAYS filter with WHERE — never run an unfiltered query, that would
      scan the entire S3 dataset for nothing.
    - Prefer `ILIKE '%term%'` for case-insensitive partial name matching
      over `=`.
    - Use the FULL scientific name (family/genus/species), never an
      acronym (e.g. 'Orthopoxvirus' not 'MPOX'; 'Lentivirus humimdef1' not
      'HIV').
    - Add a LIMIT (a few thousand rows) unless you really need everything
      matching the filter.

    Examples:
        -- Counts / composition
        SELECT VIRUS_GENUS, COUNT(*) AS n
        FROM host
        WHERE VIRUS_FAMILY ILIKE '%poxviridae%'
        GROUP BY VIRUS_GENUS
        LIMIT 1000

        -- Rows for query_dataframe / create_visualization
        SELECT VIRUS_SPECIES, HOST_TAX_NAME, COUNTRY, DISEASE_STATUS
        FROM host
        WHERE VIRUS_FAMILY ILIKE '%poxviridae%'
        LIMIT 10000

        -- Rows for create_map (coordinates MUST be extracted like this;
        -- primary_id MUST always be included — it's the sample identifier
        -- create_map requires in hover_data)
        #Here we want to get the coordinates of all host-virus observations for the Poxviridae family,
        # but we only want the first 50,000 rows to avoid overloading the system.
        # We also want to make sure that we only include rows where the geometry is not null, since we can't map points without coordinates.
        SELECT ST_X(geometry) AS lon, ST_Y(geometry) AS lat,
               primary_id, VIRUS_SPECIES, HOST_TAX_NAME, COUNTRY
        FROM host
        WHERE VIRUS_FAMILY ILIKE '%poxviridae%' AND geometry IS NOT NULL
        LIMIT 50000

    Args:
        sql: A single read-only SELECT statement against the `host` table.
        preview_rows: How many rows of the result to include in the preview.
    """
    stripped = sql.strip()
    if not re.match(r"^\(?\s*SELECT\b", stripped, re.IGNORECASE):
        return _fail("Error: only SELECT statements are allowed.")
    if _FORBIDDEN_SQL_KEYWORDS.search(stripped):
        return _fail("Error: query contains a forbidden keyword — only a read-only SELECT is allowed.")
    if _BARE_SELECT_STAR.search(stripped):
        return _fail(
            "Error: `SELECT *` is not allowed — it pulls every column (including "
            "the heavy `geometry` blob) for every matching row over S3 and will "
            "time out. List only the columns you actually need. Read the "
            "resource://datasets/host/schema resource for exact column names, "
            "and use ST_X(geometry) AS lon, ST_Y(geometry) AS lat if you need "
            "coordinates."
        )

    try:
        result = ctx.host_con.execute(stripped).fetch_df()
    except Exception:
        return _fail(traceback.format_exc())

    if result.empty:
        return _fail(
            "Error: query returned 0 rows. Try a broader ILIKE filter, "
            "check exact values first with e.g. "
            "\"SELECT DISTINCT VIRUS_FAMILY FROM host LIMIT 50\", "
            "or use the full scientific name instead of an acronym."
        )

    ctx.last_host_result = result
    return _table_result(result, preview_rows)


@mcp.tool
def query_dataframe(code: str, preview_rows: int = DEFAULT_PREVIEW_ROWS) -> dict:
    """
    Execute pandas code to query and extract data from the viral datasets.

    Available variables inside your code: df_taxo, df_host, pd, np

    - df_taxo is the full taxonomy table (in-memory). Read the
      `resource://datasets/taxonomy/schema` resource for exact column names.
      Notably: search taxon names via ORGANISM_NAME, never SPECIES_NAME
      (SPECIES_NAME is almost always empty — it's only populated for a
      minority of rows). If a search on the full species name returns 0
      rows, retry with the genus only, e.g.
      `df_taxo['ORGANISM_NAME'].str.contains('Orthopoxvirus', case=False)`.
    - df_host is the result of your MOST RECENT `query_host_sql(...)` call
      — a small, already-filtered subset of the S3 host table. If you
      haven't called query_host_sql yet and your code references df_host,
      this will fail with a clear message telling you to do that first.
    - NEVER search with an acronym (HIV, HBV, MPOX, SARS, etc.) — resolve it
      to the full scientific name first with `ncbi_taxonomy_search`,
      e.g. HIV → 'Lentivirus humimdef1', MPOX → 'Orthopoxvirus'.

    You MUST assign your result to the variable 'result' (a pandas DataFrame).

    Examples:
        result = df_taxo[df_taxo['ORGANISM_NAME'].str.contains('Orthopoxvirus', case=False)]
        result = df_host.groupby('VIRUS_FAMILY').size().reset_index(name='count')

    Args:
        code: Pandas code. Must assign its output to 'result'.
        preview_rows: How many rows of the result to include in the preview.
    """
    try:
        if "df_host" in code and ctx.last_host_result is None:
            return _fail(
                "Error: df_host is not available yet — it lives on S3 and is no "
                "longer preloaded. Call query_host_sql(...) first with a SQL "
                "SELECT + WHERE filter to fetch the subset you need, then retry."
            )

        env = {"df_taxo": ctx.df_taxo, "df_host": ctx.last_host_result, "pd": pd, "np": np}
        exec(code, {}, env)

        if "result" not in env:
            return _fail("Error: code must assign 'result' variable (pandas DataFrame)")

        result = env["result"]
        if not isinstance(result, pd.DataFrame):
            return _fail(f"Error: 'result' must be a pandas DataFrame, got {type(result)}")

        if result.empty:
            return _fail(
                "Error: query returned 0 rows. "
                "The search term yielded no results in the dataset. "
                "Try a broader term, use the full scientific species name instead of an acronym, "
                "or use str.contains() with a partial/genus-level name match."
            )

        return _table_result(result, preview_rows)

    except Exception:
        return _fail(traceback.format_exc())


@mcp.tool
def create_visualization(code: str) -> dict:
    """
    Execute pandas/Plotly code to create a visualization (chart, graph, plot).

    Available variables inside your code: df_taxo, df_host, pd, np, px, go

    - df_taxo is the full taxonomy table (in-memory).
    - df_host is the result of your MOST RECENT `query_host_sql(...)` call
      — a small, already-filtered subset of the S3 host table. Call
      query_host_sql first if you haven't yet and need df_host here.

    You MUST assign your Plotly figure to the variable 'fig'.

    Example:
        data = df_taxo.groupby('FAMILY_NAME').size().reset_index(name='count')
        fig = px.bar(data, x='FAMILY_NAME', y='count', title='Species per Family')

    IMPORTANT: NEVER search with acronyms (HIV, HBV, etc.). Always use the
    full scientific name. If the tool returns an error about empty data or
    empty figure, retry with a broader or alternate term.

    Args:
        code: Pandas/Plotly code. Must assign the figure to 'fig'.
    """
    try:
        if "df_host" in code and ctx.last_host_result is None:
            return _fail(
                "Error: df_host is not available yet — it lives on S3 and is no "
                "longer preloaded. Call query_host_sql(...) first with a SQL "
                "SELECT + WHERE filter to fetch the subset you need, then retry."
            )

        env = {"df_taxo": ctx.df_taxo, "df_host": ctx.last_host_result, "pd": pd, "np": np, "px": px, "go": go}
        exec(code, {}, env)

        empty_err = _check_data_not_empty(env)
        if empty_err:
            return _fail(empty_err)

        if "fig" not in env:
            return _fail("Error: code must assign 'fig' variable (Plotly figure)")

        fig = env["fig"]
        if not isinstance(fig, (go.Figure, go.FigureWidget)):
            return _fail(f"Error: 'fig' must be a Plotly figure, got {type(fig)}")

        if not _check_figure_has_data(fig):
            return _fail(
                "Error: the figure was created but contains no data points (empty chart). "
                "The filter likely returned 0 rows. "
                "Use the full scientific species name instead of an acronym (e.g. 'Lentivirus humimdef1' not 'HIV'). "
                "Try str.contains() with a partial or genus-level term, or call query_dataframe first "
                "to verify which names exist in the dataset."
            )

        title = _figure_title(fig)
        content = f"Visualization generated: {title}." if title else "Visualization generated successfully."
        return _ok(content, [_figure_artifact(fig)])

    except Exception:
        return _fail(traceback.format_exc())


@mcp.tool
def create_map(code: str) -> dict:
    """
    Execute pandas/Plotly code to create a geographic map showing
    virus-host observation locations.

    MANDATORY: Use ONLY px.scatter_mapbox() and mapbox_style. NEVER use
    scatter_map or map_style.

    Available variables inside your code: df_taxo, df_host, pd, np, px, go

    df_host is the result of your MOST RECENT `query_host_sql(...)` call —
    a small, already-filtered subset of the S3 host table. This means the
    heavy filtering (by species/genus/family/country/etc.) should already
    have happened in your SQL query; here you're just further narrowing
    that subset (if needed) and plotting it.

    IMPORTANT: `host` has no plain lat/lon columns — coordinates live in a
    `geometry` point column. Your PRECEDING query_host_sql call MUST extract
    them explicitly: `SELECT ST_X(geometry) AS lon, ST_Y(geometry) AS lat, ...`.
    If df_host doesn't have 'lat'/'lon' columns, call query_host_sql again
    with that extraction before using this tool.

    MANDATORY — sample identifier: every point plotted MUST be traceable back
    to its exact sample. Your preceding query_host_sql call MUST SELECT
    `primary_id` (the BioSample accession), and `primary_id` MUST be included
    in `hover_data` below. A map without it will be REJECTED.

    EXACT TEMPLATE TO FOLLOW (mandatory, adapt only the title and any
    extra narrowing — but NEVER drop primary_id from hover_data):
        data = df_host.dropna(subset=['lat', 'lon'])
        fig = px.scatter_mapbox(
            data, lat='lat', lon='lon',
            hover_name='VIRUS_SPECIES',
            hover_data=['primary_id', 'HOST_TAX_NAME', 'COUNTRY'],
            color='VIRUS_SPECIES',
            zoom=1, title='TITLE'
        )
        fig.update_layout(mapbox_style='open-street-map')

    Replace TITLE with a descriptive title. You MUST assign the figure to 'fig'.

    IMPORTANT: NEVER search with acronyms (HIV, HBV, etc.). Always use the
    full scientific name — ideally already filtered in your query_host_sql
    call. If the tool returns an error about empty data or empty map, call
    query_host_sql again with a broader filter.

    Args:
        code: Plotly code using scatter_mapbox. Must assign the figure to 'fig'.
    """
    try:
        if "df_host" in code and ctx.last_host_result is None:
            return _fail(
                "Error: df_host is not available yet — it lives on S3 and is no "
                "longer preloaded. Call query_host_sql(...) first with a SQL "
                "SELECT + WHERE filter to fetch the subset you need, then retry."
            )

        env = {
            "df_taxo": ctx.df_taxo, "df_host": ctx.last_host_result,
            "pd": pd, "np": np, "px": px, "go": go,
        }
        exec(code, {}, env)

        empty_err = _check_data_not_empty(env)
        if empty_err:
            return _fail(empty_err)

        if "fig" not in env:
            return _fail("Error: code must assign 'fig' variable (Plotly figure)")

        fig = env["fig"]
        if not isinstance(fig, (go.Figure, go.FigureWidget)):
            return _fail(f"Error: 'fig' must be a Plotly figure, got {type(fig)}")

        if not _check_figure_has_data(fig):
            return _fail(
                "Error: the map was created but contains no data points (empty map). "
                "The filter likely returned 0 rows — no matching entries found in df_host. "
                "Use the full scientific species name instead of an acronym (e.g. 'Orthopoxvirus' not 'MPOX'). "
                "Call query_host_sql again with a broader ILIKE filter, or read the "
                "resource://datasets/host/schema resource to inspect what columns exist."
            )

        if not _check_hover_has_column(fig, "primary_id"):
            return _fail(
                "Error: the map is missing the sample identifier (primary_id) — every "
                "point MUST be traceable back to its exact BioSample sample. Fix BOTH: "
                "1) your query_host_sql SELECT must include primary_id, and "
                "2) hover_data=[...] in px.scatter_mapbox must include 'primary_id'. "
                "Retry create_map with primary_id in hover_data."
            )

        title = _figure_title(fig)
        content = f"A map has been generated — {title}." if title else "A map has been generated showing host locations."
        return _ok(content, [_figure_artifact(fig)])

    except Exception:
        return _fail(traceback.format_exc())


# ==================== ENTRYPOINT ==================== #

if __name__ == "__main__":
    initialize_datasets()
    mcp.run(transport="http", host="0.0.0.0", port=8000, path="/mcp")
