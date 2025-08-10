import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy
import json
import re
from datetime import datetime

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder (now smart!) ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    """
    Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
    """
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))

# --- Fact-checking (batch) tool: fact_check_batch ---
# environment variable: GOOGLE_FACT_CHECK_API_KEY
FACTCHECK_API_KEY = os.environ.get("GOOGLE_FACT_CHECK_API_KEY")
FACTCHECK_BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def simple_sentence_split(text: str) -> list[str]:
    """
    Very small heuristic sentence splitter.
    Keeps sentences long enough to be real claims.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", (text or "")).strip()
    # Split on sentence endings but keep abbreviations naive guard
    candidates = re.split(r'(?<=[\.\?\!])\s+', text)
    claims = []
    for s in candidates:
        s = s.strip()
        # remove trailing punctuation
        s = re.sub(r'^[\-\—\–\•\:\;]+', '', s).strip()
        s = re.sub(r'[\n\r]+', ' ', s)
        if len(s) >= 20 and len(s.split()) >= 3:
            claims.append(s)
    return claims

async def google_factcheck_search(client: httpx.AsyncClient, claim: str, language: str = "en"):
    """Call Google Fact Check Tools API for one claim. Returns parsed 'claims' or None."""
    if not FACTCHECK_API_KEY:
        return None
    params = {"query": claim, "key": FACTCHECK_API_KEY, "languageCode": language}
    try:
        r = await client.get(FACTCHECK_BASE, params=params, timeout=20)
    except httpx.HTTPError as e:
        return {"error": f"HTTP error calling Fact Check API: {e}"}
    if r.status_code != 200:
        # Some errors return 400/403; return parsed error
        try:
            return {"error": r.text}
        except Exception:
            return {"error": f"status {r.status_code}"}
    try:
        return r.json()
    except Exception as e:
        return {"error": f"failed to parse JSON: {e}"}

def extract_ld_json_from_html(html: str) -> list[dict]:
    """
    Find <script type="application/ld+json"> blocks and parse JSON. Return list of objects.
    """
    blocks = []
    for match in re.finditer(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
                             html, flags=re.IGNORECASE | re.DOTALL):
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                blocks.extend(parsed)
            else:
                blocks.append(parsed)
        except Exception:
            # try to clean some common errors (trailing commas)
            try:
                cleaned = re.sub(r',\s*}', '}', raw)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    blocks.extend(parsed)
                else:
                    blocks.append(parsed)
            except Exception:
                continue
    return blocks

def build_schema_from_google_claim(google_claim: dict) -> dict:
    """
    Map one Google claims -> Schema.org ClaimReview (JSON-LD)
    Picks the first claimReview entry in the google_claim.
    """
    claim_text = google_claim.get("text", "")
    crs = google_claim.get("claimReview", [])
    if not crs:
        return {
            "@context": "https://schema.org",
            "@type": "ClaimReview",
            "claimReviewed": claim_text,
            "reviewRating": {
                "@type": "Rating",
                "alternateName": "No review found"
            }
        }
    review = crs[0]
    publisher = review.get("publisher", {}) or {}
    textualRating = review.get("textualRating") or review.get("textRating") or None
    review_date = review.get("reviewDate") or review.get("publicationDate") or None

    # attempt to assign ratingValue (best effort)
    rating_value = None
    alt = textualRating or ""
    alt_l = alt.lower()
    if "true" in alt_l or "correct" in alt_l or "confirmed" in alt_l:
        rating_value = "1"
    elif "false" in alt_l or "incorrect" in alt_l or "pants on fire" in alt_l or "fabricated" in alt_l:
        rating_value = "0"
    else:
        rating_value = None

    schema = {
        "@context": "https://schema.org",
        "@type": "ClaimReview",
        "datePublished": review_date,
        "url": review.get("url"),
        "claimReviewed": claim_text,
        "author": {
            "@type": "Organization" if publisher.get("name") else "Person",
            "name": publisher.get("name") or review.get("publisher", {}).get("site") or "Unknown"
        },
        "reviewRating": {
            "@type": "Rating",
            "alternateName": textualRating or "Unknown",
        }
    }
    if rating_value is not None:
        schema["reviewRating"]["ratingValue"] = rating_value
        schema["reviewRating"]["bestRating"] = "1"
        schema["reviewRating"]["worstRating"] = "0"
    return schema

def build_schema_no_match(claim_text: str, user_name: str | None = None) -> dict:
    return {
        "@context": "https://schema.org",
        "@type": "ClaimReview",
        "datePublished": datetime.utcnow().strftime("%Y-%m-%d"),
        "claimReviewed": claim_text,
        "author": {
            "@type": "Person" if user_name else "Organization",
            "name": user_name or "FactChecker AI"
        },
        "reviewRating": {
            "@type": "Rating",
            "alternateName": "No matching ClaimReview found"
        }
    }

# MCP tool
FACTCHECK_DESCRIPTION = RichToolDescription(
    description="Batch fact-check tool: extract claims from text (or accept a single claim), check Google Fact Check API, fallback to search+page fetch, and output Schema.org ClaimReview JSON-LD plus a human summary.",
    use_when="Use when user gives an article, text, or claim to verify multiple statements.",
    side_effects="Queries external web/data sources (Google Fact Check API, and websites) and returns structured JSON-LD."
)

@mcp.tool(description=FACTCHECK_DESCRIPTION.model_dump_json())
async def fact_check_batch(
    text_or_claim: Annotated[str, Field(description="Either a single claim or a longer text/article to extract claims from.")],
    user_name: Annotated[str | None, Field(description="Optional - name of the user/author for itemReviewed.author")]=None,
    max_claims: Annotated[int, Field(description="Maximum number of claims to extract (default 5)")] = 5,
    language: Annotated[str, Field(description="Language code for Fact Check API (default 'en')")] = "en",
) -> str:
    """
    - If `text_or_claim` looks like multiple sentences, we extract up to max_claims claims.
    - For each claim:
      1) Query Google Fact Check Tools API.
      2) If no results, do a search (Fetch.google_search_links) and fetch pages to find ClaimReview JSON-LD.
      3) Build Schema.org ClaimReview JSON-LD and a short human-readable summary.
    Returns a JSON string with keys: 'summary' (text) and 'schema' (list of JSON-LD objects).
    """
    if not text_or_claim or not text_or_claim.strip():
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide text or a claim to verify."))

    # decide if it's one claim or many
    sentences = simple_sentence_split(text_or_claim)
    # If the input is short (<200 chars) treat it as a single claim
    if len(text_or_claim) < 200 and (len(sentences) == 0 or len(sentences) == 1):
        claims = [text_or_claim.strip()]
    else:
        claims = sentences[:max_claims] if sentences else [text_or_claim.strip()]

    async with httpx.AsyncClient() as client:
        results = []
        human_lines = []
        for i, claim in enumerate(claims, start=1):
            try:
                # 1) Ask Google Fact Check API
                gresp = await google_factcheck_search(client, claim, language=language)
                if not gresp:
                    # no key or no response - treat as no match
                    schema_obj = build_schema_no_match(claim, user_name)
                    human_lines.append(f"({i}) No fact-check results found for: '{claim}'")
                    results.append(schema_obj)
                    continue

                if isinstance(gresp, dict) and gresp.get("error"):
                    # API error response - treat gracefully but include error note
                    human_lines.append(f"({i}) Fact Check API error for claim: '{claim}' — {gresp.get('error')}")
                    # proceed to fallback search below to try to get something
                    found_schema = None
                else:
                    claims_list = gresp.get("claims", []) if isinstance(gresp, dict) else []
                    if claims_list:
                        # Map first claim result to Schema
                        schema_obj = build_schema_from_google_claim(claims_list[0])
                        human_lines.append(f"({i}) Found fact-check for: '{claim}' — verdict: {schema_obj.get('reviewRating', {}).get('alternateName', 'Unknown')}. Source: {schema_obj.get('url')}")
                        results.append(schema_obj)
                        continue
                    else:
                        found_schema = None

                # 2) Fallback: search + fetch pages; look for ClaimReview JSON-LD
                links = await Fetch.google_search_links(claim, num_results=4)
                ld_found = None
                # if links is an error marker, note that
                if links and isinstance(links, list) and links[0].startswith("<error>"):
                    human_lines.append(f"({i}) Search failed for claim: '{claim}' — fallback search error.")
                    results.append(build_schema_no_match(claim, user_name))
                    continue

                for link in links:
                    # fetch raw HTML
                    try:
                        page_raw, maybe_note = await Fetch.fetch_url(link, Fetch.USER_AGENT, force_raw=True)
                    except McpError as e:
                        # couldn't fetch this page, try next
                        continue
                    # try to extract JSON-LD
                    ld_blocks = extract_ld_json_from_html(page_raw)
                    for b in ld_blocks:
                        # if block itself is ClaimReview or contains it
                        def is_claimreview(obj):
                            try:
                                t = obj.get("@type") if isinstance(obj, dict) else None
                                if isinstance(t, list):
                                    return "ClaimReview" in t
                                return (isinstance(t, str) and "ClaimReview" in t)
                            except Exception:
                                return False
                        if isinstance(b, dict) and is_claimreview(b):
                            ld_found = b
                            break
                        # sometimes LD block is a graph
                        if isinstance(b, dict) and b.get("@graph"):
                            for node in b.get("@graph", []):
                                if is_claimreview(node):
                                    ld_found = node
                                    break
                            if ld_found:
                                break
                    if ld_found:
                        # Map minimal fields to Schema (we can re-use as-is)
                        # ensure @context/@type exist
                        if "@context" not in ld_found:
                            ld_found["@context"] = "https://schema.org"
                        human_lines.append(f"({i}) Found ClaimReview on page: {link}")
                        results.append(ld_found)
                        break

                if not ld_found:
                    human_lines.append(f"({i}) No ClaimReview found for: '{claim}'. Returning no-match Schema.")
                    results.append(build_schema_no_match(claim, user_name))
            except McpError as e:
                # surface MCP errors as part of result
                human_lines.append(f"({i}) Error verifying claim: '{claim}' -> {str(e)}")
                results.append(build_schema_no_match(claim, user_name))

    output = {
        "summary": "\n".join(human_lines),
        "schema": results
    }
    # return compact JSON string (MCP tool returns str)
    return json.dumps(output, ensure_ascii=False, indent=2)


# Image inputs and sending images

MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    import base64
    import io

    from PIL import Image

    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
