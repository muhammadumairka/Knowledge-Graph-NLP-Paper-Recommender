import sys
import logging
from typing import List, Tuple

import pandas as pd
import networkx as nx
import arxiv

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MAX_PAPERS = 20
DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.5

# Combined ranking weight
SEMANTIC_WEIGHT = 0.7
PAGERANK_WEIGHT = 0.3


# -----------------------------
# Logging Setup
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# -----------------------------
# arXiv Paper Fetching
# -----------------------------

def fetch_arxiv_papers(query: str, max_results: int = DEFAULT_MAX_PAPERS) -> pd.DataFrame:
    """
    Fetch recent research papers from arXiv based on the user's query.

    Parameters:
        query (str): Research topic entered by the user.
        max_results (int): Maximum number of papers to fetch.

    Returns:
        pd.DataFrame: DataFrame containing paper title, abstract, published date, and URL.

    Raises:
        ValueError: If query is empty or max_results is invalid.
        RuntimeError: If papers cannot be fetched from arXiv.
    """

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if max_results <= 0:
        raise ValueError("max_results must be greater than 0.")

    try:
        logging.info("Fetching recent papers from arXiv...")

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers = []

        for result in search.results():
            title = result.title.strip() if result.title else ""
            abstract = result.summary.replace("\n", " ").strip() if result.summary else ""
            published = result.published.strftime("%Y-%m-%d") if result.published else "N/A"
            url = result.entry_id if result.entry_id else "N/A"

            # Skip papers with missing title or abstract
            if not title or not abstract:
                continue

            papers.append({
                "title": title,
                "abstract": abstract,
                "published": published,
                "url": url
            })

        df = pd.DataFrame(papers)

        if df.empty:
            logging.warning("No valid papers found for this topic.")
            return df

        # Remove duplicate papers based on title
        df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

        logging.info(f"Fetched {len(df)} valid papers.")

        return df

    except Exception as error:
        raise RuntimeError(
            "Failed to fetch papers from arXiv. "
            "Please check your internet connection or try a different query."
        ) from error


# -----------------------------
# Model Loading
# -----------------------------

def load_sentence_transformer(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Load the Sentence Transformer model.

    Parameters:
        model_name (str): Name of the pretrained Sentence Transformer model.

    Returns:
        SentenceTransformer: Loaded transformer model.

    Raises:
        RuntimeError: If model loading fails.
    """

    try:
        logging.info(f"Loading Sentence Transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        return model

    except Exception as error:
        raise RuntimeError(
            "Failed to load the Sentence Transformer model. "
            "During first run, this model needs to be downloaded from the internet. "
            "Please check your internet connection."
        ) from error


# -----------------------------
# Embedding Generation
# -----------------------------

def generate_embeddings(model: SentenceTransformer, abstracts: List[str]):
    """
    Generate semantic embeddings for paper abstracts.

    Parameters:
        model (SentenceTransformer): Loaded Sentence Transformer model.
        abstracts (List[str]): List of paper abstracts.

    Returns:
        numpy.ndarray: Embedding vectors for each abstract.

    Raises:
        ValueError: If abstracts list is empty.
        RuntimeError: If embedding generation fails.
    """

    if not abstracts:
        raise ValueError("No abstracts available for embedding generation.")

    try:
        logging.info("Generating semantic embeddings...")

        embeddings = model.encode(
            abstracts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return embeddings

    except Exception as error:
        raise RuntimeError("Failed to generate semantic embeddings.") from error


# -----------------------------
# Knowledge Graph Construction
# -----------------------------

def build_graph(
    df: pd.DataFrame,
    embeddings,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> nx.Graph:
    """
    Build a knowledge graph where:
    - Each paper is represented as a node.
    - An edge is created between two papers if their cosine similarity is above the threshold.

    Parameters:
        df (pd.DataFrame): DataFrame containing paper data.
        embeddings: Semantic embeddings of paper abstracts.
        similarity_threshold (float): Minimum similarity required to create an edge.

    Returns:
        nx.Graph: Similarity-based knowledge graph.

    Raises:
        ValueError: If input data is invalid.
        RuntimeError: If graph construction fails.
    """

    if df.empty:
        raise ValueError("Cannot build graph because the paper dataframe is empty.")

    if len(df) != len(embeddings):
        raise ValueError("Number of papers and embeddings do not match.")

    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1.")

    try:
        logging.info("Building knowledge graph...")

        graph = nx.Graph()

        # Add each paper as a node in the graph
        for i, row in df.iterrows():
            graph.add_node(
                i,
                title=row["title"],
                published=row["published"],
                url=row["url"]
            )

        # Compute pairwise cosine similarity once for better performance
        similarity_matrix = cosine_similarity(embeddings)

        edge_count = 0

        # Add weighted edges between semantically similar papers
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = float(similarity_matrix[i][j])

                if similarity >= similarity_threshold:
                    graph.add_edge(i, j, weight=similarity)
                    edge_count += 1

        logging.info(
            f"Graph created with {graph.number_of_nodes()} nodes and {edge_count} edges."
        )

        if edge_count == 0:
            logging.warning(
                "No edges were created in the graph. "
                "The similarity threshold may be too high. "
                "Recommendations will mainly depend on semantic similarity."
            )

        return graph

    except Exception as error:
        raise RuntimeError("Failed to build the knowledge graph.") from error


# -----------------------------
# PageRank Calculation
# -----------------------------

def compute_pagerank(graph: nx.Graph) -> dict:
    """
    Compute PageRank centrality scores for papers in the graph.

    Parameters:
        graph (nx.Graph): Knowledge graph.

    Returns:
        dict: PageRank scores for graph nodes.

    Raises:
        ValueError: If graph is empty.
        RuntimeError: If PageRank calculation fails.
    """

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot compute PageRank because the graph has no nodes.")

    try:
        logging.info("Computing PageRank scores...")
        pagerank_scores = nx.pagerank(graph, weight="weight")
        return pagerank_scores

    except nx.PowerIterationFailedConvergence:
        logging.warning(
            "PageRank did not converge with default settings. "
            "Retrying with more iterations..."
        )

        try:
            return nx.pagerank(graph, weight="weight", max_iter=500)
        except Exception as error:
            raise RuntimeError("PageRank failed even after increasing iterations.") from error

    except Exception as error:
        raise RuntimeError("Failed to compute PageRank scores.") from error


# -----------------------------
# Recommendation Logic
# -----------------------------

def calculate_recommendations(
    query: str,
    df: pd.DataFrame,
    model: SentenceTransformer,
    embeddings,
    pagerank_scores: dict,
    top_k: int = DEFAULT_TOP_K
) -> List[Tuple[int, float, float, float]]:
    """
    Calculate final recommendation scores.

    Final score formula:
        final_score = 70% semantic similarity + 30% PageRank score

    Parameters:
        query (str): User research topic.
        df (pd.DataFrame): Paper dataframe.
        model (SentenceTransformer): Loaded Sentence Transformer model.
        embeddings: Embeddings of paper abstracts.
        pagerank_scores (dict): PageRank scores of graph nodes.
        top_k (int): Number of top recommendations to return.

    Returns:
        List[Tuple[int, float, float, float]]:
        Each tuple contains:
        - paper index
        - final score
        - semantic similarity score
        - PageRank score
    """

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if df.empty:
        raise ValueError("No papers available for recommendation.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    # Avoid requesting more papers than available
    top_k = min(top_k, len(df))

    try:
        logging.info("Calculating query similarity...")

        query_embedding = model.encode([query], convert_to_numpy=True)
        similarity_scores = cosine_similarity(query_embedding, embeddings)[0]

        final_scores = []

        for i, similarity in enumerate(similarity_scores):
            graph_score = pagerank_scores.get(i, 0)

            final_score = (
                SEMANTIC_WEIGHT * float(similarity)
                + PAGERANK_WEIGHT * float(graph_score)
            )

            final_scores.append(
                (
                    i,
                    final_score,
                    float(similarity),
                    float(graph_score)
                )
            )

        final_scores = sorted(
            final_scores,
            key=lambda x: x[1],
            reverse=True
        )

        return final_scores[:top_k]

    except Exception as error:
        raise RuntimeError("Failed to calculate recommendations.") from error


# -----------------------------
# Output Display
# -----------------------------

def display_recommendations(
    recommendations: List[Tuple[int, float, float, float]],
    df: pd.DataFrame
) -> None:
    """
    Display recommended papers in a clean terminal format.

    Parameters:
        recommendations: List of recommendation tuples.
        df (pd.DataFrame): Paper dataframe.
    """

    if not recommendations:
        print("\nNo recommendations found.")
        return

    print("\nTop Recommended Papers:\n")

    for rank, (idx, final_score, similarity, graph_score) in enumerate(
        recommendations,
        start=1
    ):
        print(f"{rank}. {df.loc[idx, 'title']}")
        print(f"   Published: {df.loc[idx, 'published']}")
        print(f"   Final Score: {final_score:.3f}")
        print(f"   Semantic Similarity: {similarity:.3f}")
        print(f"   PageRank Score: {graph_score:.3f}")
        print(f"   URL: {df.loc[idx, 'url']}")
        print()


# -----------------------------
# Main Recommendation Function
# -----------------------------

def recommend(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    max_papers: int = DEFAULT_MAX_PAPERS
) -> None:
    """
    Complete recommendation pipeline.

    Parameters:
        query (str): User research topic.
        top_k (int): Number of recommended papers to display.
        max_papers (int): Number of recent arXiv papers to fetch.
    """

    try:
        # Step 1: Fetch recent papers from arXiv
        df = fetch_arxiv_papers(query, max_results=max_papers)

        if df.empty:
            print("\nNo papers found for this topic. Try a broader query.")
            return

        # Step 2: Load NLP model
        model = load_sentence_transformer()

        # Step 3: Generate embeddings from paper abstracts
        embeddings = generate_embeddings(
            model,
            df["abstract"].tolist()
        )

        # Step 4: Build semantic knowledge graph
        graph = build_graph(
            df,
            embeddings,
            similarity_threshold=SIMILARITY_THRESHOLD
        )

        # Step 5: Compute PageRank scores
        pagerank_scores = compute_pagerank(graph)

        # Step 6: Calculate final recommendation scores
        recommendations = calculate_recommendations(
            query=query,
            df=df,
            model=model,
            embeddings=embeddings,
            pagerank_scores=pagerank_scores,
            top_k=top_k
        )

        # Step 7: Display recommendations
        display_recommendations(recommendations, df)

    except ValueError as error:
        logging.error(error)

    except RuntimeError as error:
        logging.error(error)

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")
        sys.exit(0)

    except Exception as error:
        logging.error(f"Unexpected error occurred: {error}")


# -----------------------------
# Program Entry Point
# -----------------------------

if __name__ == "__main__":
    try:
        user_query = input("Enter research topic: ").strip()

        if not user_query:
            print("Search topic cannot be empty.")
            sys.exit(1)

        recommend(user_query)

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")
        sys.exit(0)
