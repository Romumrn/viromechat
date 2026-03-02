import re
import traceback
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ==================== TOOL IMPLEMENTATIONS ==================== #

def wikipedia_search(search_term: str, wikipedia_limit: int) -> dict:
    term = re.sub(r"[^\w\s\-]", "", search_term.strip())
    title = requests.utils.quote(term)
    
    # 1. Tentative de récupération de la page exacte
    api_url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&format=json&prop=extracts&explaintext=1"
        f"&titles={title}"
    )
    r = requests.get(api_url, headers={"User-Agent": "VirusAgent/1.0"})
    
    if r.status_code == 200:
        pages = r.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        
        if "missing" not in page:
            page_title = page.get("title", term)
            extract = page.get("extract", "")
            url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
            if len(extract) > wikipedia_limit:
                extract = extract[:wikipedia_limit] + "... [truncated]"
            return {"success": True, "title": page_title, "extract": extract, "url": url}
    
    # 2. Fallback : recherche de pages proches via opensearch ou search
    search_url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&format=json&list=search"
        f"&srsearch={title}&srlimit=1&srprop=snippet"
    )
    sr = requests.get(search_url, headers={"User-Agent": "VirusAgent/1.0"})
    
    if sr.status_code != 200:
        return {"success": False, "message": f"No Wikipedia article found for {search_term}"}
    
    search_results = sr.json().get("query", {}).get("search", [])
    if not search_results:
        return {"success": False, "message": f"No Wikipedia article found for {search_term}"}
    
    # 3. Récupération du contenu de la page la plus proche
    best_match_title = search_results[0]["title"]
    best_match_title_encoded = requests.utils.quote(best_match_title)
    
    page_url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&format=json&prop=extracts&explaintext=1"
        f"&titles={best_match_title_encoded}"
    )
    pr = requests.get(page_url, headers={"User-Agent": "VirusAgent/1.0"})
    
    if pr.status_code != 200:
        return {"success": False, "message": f"No Wikipedia article found for {search_term}"}
    
    pages = pr.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    
    if "missing" in page:
        return {"success": False, "message": f"No Wikipedia article found for {search_term}"}
    
    page_title = page.get("title", best_match_title)
    extract = page.get("extract", "")
    url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
    if len(extract) > wikipedia_limit:
        extract = extract[:wikipedia_limit] + "... [truncated]"
    
    return {
        "success": True,
        "title": page_title,
        "extract": extract,
        "url": url,
        "fuzzy_match": True,           # indique que c'est une page approchée
        "original_search": search_term  # terme original pour traçabilité
    }


def query_dataframe(code: str, df_taxo: pd.DataFrame, df_host: pd.DataFrame, preview_rows ) -> dict:
    try:
        env = {"df_taxo": df_taxo, "df_host": df_host, "pd": pd, "np": np}
        exec(code, {}, env)

        if "result" not in env:
            return {"success": False, "message": "Error: code must assign 'result' variable (pandas DataFrame)"}

        result = env["result"]
        if not isinstance(result, pd.DataFrame):
            return {"success": False, "message": f"Error: 'result' must be a pandas DataFrame, got {type(result)}"}

        preview = (
            result.to_string(index=False) if len(result) <= preview_rows
            else result.head(preview_rows).to_string(index=False) + f"\n... and {len(result) - preview_rows} more rows"
        )
        return {"success": True, "result": result, "shape": result.shape, "columns": list(result.columns), "preview": preview}

    except Exception:
        return {"success": False, "message": traceback.format_exc()}


def create_visualization(code: str, df_taxo: pd.DataFrame, df_host: pd.DataFrame) -> dict:
    try:
        env = {"df_taxo": df_taxo, "df_host": df_host, "pd": pd, "np": np, "px": px, "go": go}
        exec(code, {}, env)

        if "fig" not in env:
            return {"success": False, "message": "Error: code must assign 'fig' variable (Plotly figure)"}

        fig = env["fig"]
        if not isinstance(fig, (go.Figure, go.FigureWidget)):
            return {"success": False, "message": f"Error: 'fig' must be a Plotly figure, got {type(fig)}"}

        return {"success": True, "figure": fig}

    except Exception:
        return {"success": False, "message": traceback.format_exc()}

def create_map(code: str, df_taxo: pd.DataFrame, df_host: pd.DataFrame) -> dict:
    try:
        env = {
            "df_taxo": df_taxo, "df_host": df_host,
            "pd": pd, "np": np, "px": px, "go": go,
        }
        exec(code, {}, env)

        if "fig" not in env:
            return {"success": False, "message": "Error: code must assign 'fig' variable (Plotly figure)"}

        fig = env["fig"]
        if not isinstance(fig, (go.Figure, go.FigureWidget)):
            return {"success": False, "message": f"Error: 'fig' must be a Plotly figure, got {type(fig)}"}

        return {"success": True, "figure": fig}

    except Exception:
        return {"success": False, "message": traceback.format_exc()}
    
# ==================== TOOL SPECIFICATIONS ==================== #

TOOLS_SPEC = [
    {
    "type": "function",
    "function": {
        "name": "wikipedia_search",
        "description": (
            "Search Wikipedia for biological or scientific information. "
            "First attempts an exact page match, then falls back to the closest relevant page if no exact match is found. "
            "Useful when searching for precise scientific names (e.g. virus species, taxa) that may not have a dedicated Wikipedia page — "
            "the tool will automatically return the most relevant related article instead. "
            "The response includes a 'fuzzy_match' field (bool) indicating whether the result is an exact match or an approximation, "
            "and an 'original_search' field with the original query for traceability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "The scientific or biological term to search for (e.g. 'Sprivivirus cyprinus', 'SARS-CoV-2', 'Rabies lyssavirus')"
                }
            },
            "required": ["search_term"]
        }
    }
    },
    {
        "type": "function",
        "function": {
            "name": "query_dataframe",
            "description": (
                "Execute pandas code to query and extract data from viral datasets. "
                "Use this tool when you need to retrieve, filter, aggregate, or analyze data.\n\n"
                "Available variables: df_taxo, df_host, pd, np\n\n"
                "You MUST assign your result to the variable 'result' (pandas DataFrame)\n\n"
                "Examples:\n"
                "result = df_taxo.groupby('family').size().reset_index(name='count')\n"
                "result = df_taxo[df_taxo['genus'] == 'Orthopoxvirus']"
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Pandas code. Must assign result to 'result'."}},
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_visualization",
            "description": (
                "Execute pandas/Plotly code to create a visualization. "
                "Use this tool when you need to create charts, graphs, or plots.\n\n"
                "Available variables: df_taxo, df_host, pd, np, px, go\n\n"
                "You MUST assign your Plotly figure to the variable 'fig'\n\n"
                "Examples:\n"
                "data = df_taxo.groupby('family').size().reset_index(name='count')\n"
                "fig = px.bar(data, x='family', y='count', title='Species per Family')"
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Pandas/Plotly code. Must assign figure to 'fig'."}},
                "required": ["code"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "create_map",
        "description": (
            "Execute pandas/Plotly code to create a geographic map showing virus-host observation locations.\n\n"
            "MANDATORY: Use ONLY px.scatter_mapbox() and mapbox_style. NEVER use scatter_map or map_style.\n\n"
            "Available variables: df_taxo, df_host, pd, np, px, go\n\n"
            "PREFER filter with .str.contains(term, case=False, na=False) — NEVER use == for name matching.\n"
            "EXACT TEMPLATE TO FOLLOW (mandatory, adapt only the filter term and title):\n"
            "```\n"
            "data = df_host[\n"
            "    df_host['VIRAL_SPECIES'].str.contains('TERM', case=False, na=False) &\n"
            "    (df_host['LOCALISATION_RESOLUTION'] == 'local')\n"
            "].dropna(subset=['lat', 'lon'])\n"
            "fig = px.scatter_mapbox(\n"
            "    data, lat='lat', lon='lon',\n"
            "    hover_name=...',\n"
            "    hover_data=...,\n"
            "    color='ORGANISM_NAME',\n"
            "    zoom=1, title='TITLE'\n"
            ")\n"
            "fig.update_layout(mapbox_style='open-street-map')\n"
            "Replace TERM with the relevant species/genus/family/id keyword from the user query.\n"
            "Replace TITLE with a descriptive title.\n"
            "You MUST assign the figure to 'fig'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Plotly code using scatter_mapbox. Must assign figure to 'fig'."}
            },
            "required": ["code"]
        }
    }
}
]