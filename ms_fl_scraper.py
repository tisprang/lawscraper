"""
File to be used for scraping statutes from FindLaw. This was made to be used primarily for scraping the ND, KY, and PA statutes.

FindLaw requires interaction with Javascript to get to the actual statutes, so this scraper uses Playwright to handle that.


The output format is identical to the other scrapers, with each statute being a JSON object written to a JSONL file. For example:
{"url": "https://law.justia.com/codes/kansas/2023/chapter-1/article-2/section-1-201/", "state": "KS", "path": "Justia\u203aU.S. Law\u203aU.S. Codes and Statutes\u203aKansas Statutes\u203a2023 Kansas Statutes\u203aChapter 1 - Accountants, Certified Public\u203aArticle 2 - State Board Of Accountancy\u203a1-201 Membership; appointment; qualifications; term; vacancies; removal.", "title": "2023 Kansas Statutes \u203a Chapter 1 - Accountants, Certified Public \u203a Article 2 - State Board Of Accountancy \u203a 1-201 Membership; appointment; qualifications; term; vacancies; removal.", "univ_cite": true, "citation": "KS Stat \u00a7 1-201 (2023)", "content": "1-201.\nMembership; appointment; qualifications; term; vacancies; removal.\n(a) There is hereby created a board of accountancy,...", "lex_path": [0, 0, 0]}
"""

import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

FL_BASE_URL = "https://codes.findlaw.com/{state}"
MISSING_STATES = ["nd", "ky", "pa"]

DEFAULT_DIR = (
    "findlaw_codes"  # Directory to save the scraped data, e.g. findlaw_codes/ND.jsonl
)


def _chunk_list(seq, n):
    """Yield n contiguous chunks from seq (as balanced as possible)."""
    if n <= 1 or len(seq) == 0:
        yield seq
        return
    k, m = divmod(len(seq), n)
    start = 0
    for i in range(n):
        end = start + k + (1 if i < m else 0)
        if start < end:
            yield seq[start:end]
        start = end

def clean_paragraphs(soup):
    paragraphs = []
    for p in soup.select("div.codes-content p"):
        text = p.get_text(" ", strip=True)          # Whitespace kollabieren
        text = re.sub(r" {2,}", " ", text)           # mehrfache Spaces entfernen
        text = re.sub(r"\n+", " ", text)             # Zeilenbrüche innerhalb eines p entfernen
        if text.lower().startswith("cite this article"):  # Zitationshinweis rausfiltern
            continue
        paragraphs.append(text)
    return "\n\n".join(paragraphs)


NODE_KIND_PATTERNS = [
    (r"^title\b", "title"),
    (r"^subtitle\b", "subtitle"),
    (r"^chapter\b", "chapter"),
    (r"^subchapter\b", "subchapter"),
    (r"^article\b", "article"),
    (r"^part\b", "part"),
    (r"^division\b", "division"),
    (r"^subdivision\b", "subdivision"),
    (r"^section\b", "section_group"),
    (r"^rule\b", "rule_group"),
]


def _guess_node_kind(label: str, level: int) -> str:
    txt = (label or "").strip().lower()
    if level == 0 and re.fullmatch(r"[a-z]{2}", txt):
        return "state"
    for pattern, kind in NODE_KIND_PATTERNS:
        if re.match(pattern, txt):
            return kind
    if "constitution" in txt:
        return "constitution"
    return "node"


def _parse_statute_heading(heading: str, fallback_label: str = "") -> Dict[str, str]:
    """
    Parse statute heading text from <h1> into structured parts.

    Example input:
    "Washington Revised Code Title 29A. Elections § 29A.04.001. Scope of definitions"
    """
    raw = (heading or "").strip()
    out = {
        "heading": raw,
        "article_code": "",
        "article_title": "",
    }
    if not raw:
        return out

    m = re.search(
        r"(?:§|\uFFFD|section\s+|sec\.\s*)([^\.\s]+(?:\.[^\.\s]+)*)\.?\s*(.*)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        out["article_code"] = m.group(1).strip()
        out["article_title"] = m.group(2).strip(" .")
        return out

    # Generic fallback for headings that don't include §: use last sentence fragment.
    parts = [p.strip() for p in raw.split(".") if p.strip()]
    if len(parts) >= 2:
        out["article_title"] = parts[-1]
    else:
        out["article_title"] = raw
    out["article_code"] = (fallback_label or "").strip()
    return out


def _build_hierarchy_metadata(
    path_so_far: List[str],
    lex_path: List[int],
    state: str,
    *,
    leaf_title: str = "",
    leaf_heading: str = "",
    leaf_code: str = "",
) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    for i, label in enumerate(path_so_far):
        nodes.append(
            {
                "level": i,
                "label": label,
                "kind": _guess_node_kind(label, i),
                "lex_index": lex_path[i] if i < len(lex_path) else None,
            }
        )

    parent_nodes = nodes[:-1] if len(nodes) > 1 else []
    leaf_node = nodes[-1] if nodes else None
    if leaf_node is not None:
        if leaf_title:
            leaf_node["title"] = leaf_title
        if leaf_heading:
            leaf_node["heading"] = leaf_heading
        if leaf_code:
            leaf_node["code"] = leaf_code

    slots: Dict[str, str] = {}
    for n in parent_nodes:
        k = n["kind"]
        if k not in ("node", "state") and k not in slots:
            slots[k] = n["label"]

    return {
        "schema": "findlaw_hierarchy_v1",
        "state": state.upper(),
        "path_nodes": path_so_far,
        "path_depth": len(path_so_far),
        "parent_nodes": parent_nodes,
        "leaf_node": leaf_node,
        "slots": slots,
    }


def _worker_scrape_sections(
    state: str,
    code_title: str,
    state_url: str,
    sections_slice: list,
    output_part_path: str,
    progress_queue=None,
    record_queue=None,
    threads: int = 4,
):
    # Worker runs without rendering tqdm bars; parent process owns console progress output
    """
    Worker process: launches its own Playwright browser and threadpool, scrapes only the given sections.
    `sections_slice` is a list of dicts: {"idx": int, "name": str, "url": str}
    Writes JSONL lines into `output_part_path`.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from urllib.parse import urljoin

    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright

    # leaf HTTP session for this process
    MAX_WORKERS = max(1, int(threads))
    leaf_session = requests.Session()
    leaf_session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Referer": "https://www.google.com/",
        }
    )
    adapter = HTTPAdapter(
        pool_connections=MAX_WORKERS * 2,
        pool_maxsize=MAX_WORKERS,
        max_retries=Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=False,
        ),
    )
    leaf_session.mount("https://", adapter)
    leaf_session.mount("http://", adapter)

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/Chicago",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        page = context.new_page()
        page.wait_for_timeout(200)
        stealth_js = """
        Object.defineProperty(navigator, 'webdriver', {get: () => false});
        window.navigator.chrome = { runtime: {} };
        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
        Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
        """
        page.add_init_script(stealth_js)

        for s in sections_slice:
            section_name = s["name"]
            section_url = s["url"]
            idx = s["idx"]
            logging.info(
                f"[pid={os.getpid()}] Scraping section: {section_name} - {section_url}"
            )
            # lightweight retry nav
            ok = False
            for attempt in range(3):
                try:
                    page.goto(section_url, timeout=60000, wait_until="domcontentloaded")
                    page.wait_for_selector("body", timeout=15000)
                    ok = True
                    break
                except PlaywrightTimeoutError:
                    logging.warning(
                        f"[pid={os.getpid()}] Timeout loading {section_url} (attempt {attempt+1}/3)"
                    )
                    try:
                        page.reload(timeout=30000, wait_until="domcontentloaded")
                    except Exception:
                        pass
                    time.sleep(2 * (attempt + 1))
                except Exception as e:
                    logging.warning(f"[pid={os.getpid()}] Navigation error: {e}")
                    time.sleep(2 * (attempt + 1))
            if not ok:
                continue

            try:
                scrape_section(
                    page,
                    state,
                    section_name,
                    section_url,
                    [code_title, section_name],
                    [idx + 1],
                    None,
                    parallel=True,
                    executor=executor,
                    session=leaf_session,
                    futures=futures,
                    return_work=False,
                    tqdm_position=0,
                    tqdm_disable=True,
                    record_queue=record_queue,
                )
                # notify parent that this section finished scheduling
                try:
                    if progress_queue is not None:
                        progress_queue.put(1)
                except Exception:
                    pass
                # workers do not report progress directly; parent updates on future completion
            except Exception as e:
                logging.error(
                    f"[pid={os.getpid()}] Error while scraping section {section_name}: {e}"
                )

        # drain futures
        if futures:
            for _ in as_completed(futures):
                pass
        executor.shutdown(wait=True)
        context.close()
        browser.close()


def scrape_state(state: str, output_dir: str, *, processes: int = 6, threads: int = 8, chunks_per_proc: int = 4) -> None:
    """
    Scrape statutes for a given state from FindLaw and save them to a JSONL file.

    Args:
    - state (str): The state abbreviation (e.g., 'nd', 'ky', 'pa').
    - output_dir (str): The directory to save the output JSONL file.

    At the first step (the state page), we can use requests and BeautifulSoup. The first sections of the code will have class fl-list-item-link within a div with class landingContent
    After that, we need to use Playwright to handle the Javascript.
    """

    def _goto_with_retry(page: Page, url: str, attempts: int = 3) -> bool:
        """
        Navigate to a URL with retries, using looser wait conditions than networkidle.
        Returns True on success, False on repeated timeout.
        """
        import logging

        for i in range(attempts):
            try:
                # Add a console logger to see page-side errors in Python logs
                # try:
                #     page.on("console", lambda msg: logging.warning(f"PAGE CONSOLE: {msg.type} :: {msg.text}"))
                # except Exception:
                #     pass
                # 'domcontentloaded' is more reliable for JS-heavy pages than 'load' or 'networkidle'
                page.goto(url, timeout=60000, wait_until="domcontentloaded")
                # Wait for a generic body element as a lighter signal the page has painted
                page.wait_for_selector("body", timeout=15000)
                return True
            except PlaywrightTimeoutError:
                logging.warning(
                    f"Timeout loading {url} (attempt {i+1}/{attempts}); backing off and retrying…"
                )
                # Try a soft reload once after a failed attempt
                try:
                    page.reload(timeout=30000, wait_until="domcontentloaded")
                except Exception:
                    pass
                time.sleep(2 * (i + 1))
            except Exception as e:
                logging.warning(
                    f"Navigation error on {url} (attempt {i+1}/{attempts}): {e}"
                )
                time.sleep(2 * (i + 1))
        logging.error(f"Timeout while loading page after {attempts} attempts: {url}")
        return False

    state = state.lower()
    FL_STATES = ["al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nj", "nm", "ny", "nc", "nd", "oh", "or", "pa", "ri", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"]
    if state not in FL_STATES:
        raise ValueError(
            f"State {state} is not in the list of FindLaw states: {FL_STATES}"
        )

    state_url = FL_BASE_URL.format(state=state)

    # Use a session and realistic browser headers to reduce 403 responses from FindLaw
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Referer": "https://www.google.com/",
        }
    )

    try:
        response = session.get(state_url, timeout=30)
    except requests.RequestException as e:
        logging.error(f"Network error retrieving state page for {state}: {e}")
        return
    if response.status_code != 200:
        logging.error(
            f"Failed to retrieve state page for {state}. Status code: {response.status_code}"
        )
        return

    soup = BeautifulSoup(response.content, "html.parser")
    # code_title = soup.select('div.landingContent h3')[0].get_text(strip=True)
    code_title = soup.select("div.fl-cases-content-list h3")[0].get_text(strip=True)
    # sections = soup.select('div.landingContent a.fl-list-item-link')
    sections = soup.select("div.fl-cases-content-list a.fl-list-item-link")
    section_urls = [a.get("href") for a in sections]

    if not sections:
        logging.warning(f"No sections found for state {state} at {state_url}")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{state.upper()}.jsonl")

    # Global executor and session (producer → consumer streaming)
    MAX_WORKERS = max(1, int(threads))
    leaf_session = requests.Session()
    leaf_session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Referer": "https://www.google.com/",
        }
    )
    adapter = HTTPAdapter(
        pool_connections=MAX_WORKERS * 2,
        pool_maxsize=MAX_WORKERS,
        max_retries=Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=False,
        ),
    )
    leaf_session.mount("https://", adapter)
    leaf_session.mount("http://", adapter)

    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    all_futures = []
    MAX_BROWSERS = max(1, int(processes))

    if MAX_BROWSERS <= 1:
        # === single-browser path (existing behavior) ===
        with sync_playwright() as p:
            user_agent = (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            browser = p.chromium.launch(
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            context = browser.new_context(
                viewport={"width": 1366, "height": 768},
                user_agent=user_agent,
                locale="en-US",
                timezone_id="America/Chicago",
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            page = context.new_page()
            page.wait_for_timeout(300)
            stealth_js = """
            Object.defineProperty(navigator, 'webdriver', {get: () => false});
            window.navigator.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
            """
            page.add_init_script(stealth_js)

            from tqdm import tqdm

            sections_bar = tqdm(
                total=len(sections),
                desc="Sections",
                unit="section",
                dynamic_ncols=True,
                position=0,
                leave=True,
            )
            with open(output_file, "w", encoding="utf-8") as f_out:
                for idx, section in enumerate(sections):
                    section_name = section.get_text(strip=True)
                    section_url = urljoin(state_url, section.get("href"))
                    logging.info(f"Scraping section: {section_name} - {section_url}")
                    if _goto_with_retry(page, section_url, attempts=3):
                        try:
                            scrape_section(
                                page,
                                state,
                                section_name,
                                section_url,
                                [code_title, section_name],
                                [idx + 1],
                                f_out,
                                parallel=True,
                                executor=executor,
                                session=leaf_session,
                                futures=all_futures,
                                return_work=False,
                                tqdm_position=1,
                            )
                            sections_bar.update(1)
                        except Exception as e:
                            logging.error(
                                f"Error while scraping section {section_name}: {e}"
                            )
                    else:
                        continue
                sections_bar.close()

                if all_futures:
                    for _ in as_completed(all_futures):
                        pass
                executor.shutdown(wait=True)
            context.close()
            browser.close()
    else:
        # === multi-browser path (N processes, each with its own browser) ===
        # Precompute the sections list we’ll distribute to workers
        sections_info = []
        for idx, section in enumerate(sections):
            name = section.get_text(strip=True)
            href = section.get("href")
            if not href:
                continue
            sections_info.append(
                {"idx": idx, "name": name, "url": urljoin(state_url, href)}
            )

        from tqdm import tqdm

        sections_bar = tqdm(
            total=len(sections_info),
            desc="Sections",
            unit="section",
            dynamic_ncols=True,
            position=0,
            leave=True,
        )
        # Create queues and writer thread for single-file output
        from queue import Empty
        from threading import Thread
        # Cap processes to number of sections
        proc_count = min(MAX_BROWSERS, max(1, len(sections_info)))
        # Improve perceived progress by splitting into more chunks than processes
        chunk_factor = max(1, int(chunks_per_proc))
        total_chunks = min(len(sections_info), proc_count * chunk_factor)
        ctx = get_context("spawn")
        manager = ctx.Manager()
        progress_q = manager.Queue(maxsize=1000)
        record_q = manager.Queue(maxsize=10000)

        # Writer thread
        def _writer():
            with open(output_file, "w", encoding="utf-8") as fout:
                while True:
                    item = record_q.get()
                    if item is None:
                        break
                    try:
                        fout.write(item)
                    except Exception:
                        pass

        writer_t = Thread(target=_writer, daemon=True)
        writer_t.start()

        with ProcessPoolExecutor(max_workers=proc_count, mp_context=ctx) as pool:
            futures = []
            for pi, chunk in enumerate(_chunk_list(sections_info, total_chunks)):
                # part_path kept for arg shape but unused in worker when record_queue is provided
                fut = pool.submit(
                    _worker_scrape_sections,
                    state,
                    code_title,
                    state_url,
                    chunk,
                    "",
                    progress_q,
                    record_q,
                    threads,
                )
                futures.append(fut)
            # Drain per-section progress from queue while workers run
            total_expected = len(sections_info)
            processed = 0
            done_workers = 0
            total_workers = len(futures)
            while processed < total_expected and done_workers < total_workers:
                try:
                    inc = progress_q.get(timeout=0.5)
                    if isinstance(inc, int) and inc > 0:
                        sections_bar.update(inc)
                        processed += inc
                except Empty:
                    pass
                # update done_workers snapshot
                done_workers = sum(1 for f in futures if f.done())
            # Drain any remaining queued increments without blocking
            try:
                while True:
                    inc = progress_q.get_nowait()
                    if isinstance(inc, int) and inc > 0:
                        sections_bar.update(inc)
                        processed += inc
            except Empty:
                pass
            # Ensure all workers have completed
            for fut in as_completed(futures):
                fut.result()
        # stop writer
        record_q.put(None)
        writer_t.join()
        try:
            manager.shutdown()
        except Exception:
            pass
        sections_bar.close()


def _wait_links_or_subaccordions(scope, timeout=6000):
    """
    Wait (briefly) for either links (leaves) or nested accordion-items to appear under `scope`.
    Returns True if something is present/attaches, False on soft-timeout.
    """
    sel = (
        ":scope > .fl-accordion-content .fl-recursive-tree-accordion a[href], "
        ":scope > .fl-accordion-content .fl-recursive-tree-accordion-list .fl-accordion-item"
    )
    # fast path: anything already there?
    try:
        if scope.locator(sel).count() > 0:
            return True
    except Exception:
        pass
    # slow path: wait briefly for first match to attach
    try:
        scope.locator(sel).first.wait_for(state="attached", timeout=timeout)
        return True
    except Exception:
        return False


def scrape_section(
    page: Page,
    state: str,
    code_name: str,
    section_url: str,
    path_so_far: List[str],
    lex_order: List[int],
    f_out,
    parallel: bool = True,
    executor=None,
    session=None,
    futures=None,
    return_work: bool = False,
    tqdm_position: int = 0,
    tqdm_disable: bool = False,
    record_queue=None,
):
    """
    Scrape a specific section of law from FindLaw.

    Args:
    - page (Page): The Playwright page object.
    - state (str): The state abbreviation.
    - section_name (str): The name of the section.
    - section_url (str): The URL of the section.
    - f_out: The output file handle.
    """
    # Load & ensure top-level accordion items exist
    page.wait_for_load_state("domcontentloaded", timeout=30000)
    page.wait_for_selector(
        ".fl-expandable-tree-accordion > .fl-accordion-item", timeout=30000
    )

    def _collect_links(scope, base_path, base_lex, section_url):
        """DFS over any number of accordion layers; collect (sec_name, url, path, lex)."""
        results = []

        # If nothing is present yet, wait briefly for either links or nested accordions.
        if (
            scope.locator(
                ":scope > .fl-accordion-content .fl-recursive-tree-accordion a[href], "
                ":scope > .fl-accordion-content .fl-recursive-tree-accordion-list .fl-accordion-item"
            ).count()
            == 0
        ):
            _wait_links_or_subaccordions(scope, timeout=6000)

        # Case A: links directly under this scope
        link_list = scope.locator(
            ":scope > .fl-accordion-content .fl-recursive-tree-accordion a[href]"
        )
        direct_count = 0
        try:
            direct_count = link_list.count()
        except Exception:
            direct_count = 0

        if direct_count > 0:
            for k in range(direct_count):
                a = link_list.nth(k)
                sec_name = a.inner_text().strip()
                href = a.get_attribute("href")
                if not href:
                    continue
                url = urljoin(section_url, href)
                results.append(
                    (sec_name, url, base_path + [sec_name], base_lex + [k + 1])
                )

        # Case B: deeper accordions under this scope
        nested_items = scope.locator(
            ":scope > .fl-accordion-content .fl-recursive-tree-accordion-list .fl-accordion-item"
        )
        nested_count = 0
        try:
            nested_count = nested_items.count()
        except Exception:
            nested_count = 0

        for j in range(nested_count):
            n_item = nested_items.nth(j)
            n_btn = n_item.locator("button.fl-accordion-button")
            try:
                n_btn.wait_for(state="attached", timeout=3000)
            except Exception:
                continue

            # label for this node
            try:
                label = n_btn.locator(".fl-text-left").inner_text(timeout=2000).strip()
            except Exception:
                label = f"Section {j+1}"

            # expand if collapsed
            if (n_btn.get_attribute("aria-expanded") or "").lower() != "true":
                n_btn.click()

            # After expanding, wait briefly for content under this node to show up (links or more accordions)
            _wait_links_or_subaccordions(n_item, timeout=6000)

            # Recurse
            results.extend(
                _collect_links(
                    n_item, base_path + [label], base_lex + [j + 1], section_url
                )
            )
        return results

    items = page.locator(".fl-expandable-tree-accordion > .fl-accordion-item")
    count = items.count()

    # Build work for this section using DFS helper
    work = []
    for i in range(count):
        item = items.nth(i)
        btn = item.locator(":scope > h2 .fl-accordion-button")
        btn.wait_for(state="attached", timeout=10000)
        btn.scroll_into_view_if_needed()

        # Get the visible header text for top-level
        try:
            top_label = btn.locator(".fl-text-left").inner_text(timeout=3000).strip()
        except Exception:
            top_label = f"Section {i+1}"

        # Expand if needed
        if (btn.get_attribute("aria-expanded") or "").lower() != "true":
            btn.click()
        # After expanding, wait briefly for either links or nested items under this top-level
        _wait_links_or_subaccordions(item, timeout=6000)

        # Collect all links at any depth under this top-level item
        work.extend(
            _collect_links(
                item,
                base_path=path_so_far + [top_label],
                base_lex=lex_order + [i + 1],
                section_url=section_url,
            )
        )

    # Guard: optionally just return the work list for upper-level management
    if return_work:
        return work

    # Stream to shared executor (minimal change to your existing parallel path)
    if parallel:
        if executor is None or session is None or futures is None:
            raise RuntimeError(
                "Parallel mode requires shared executor, session, and futures list."
            )

        section_bar = tqdm(
            total=len(work),
            desc=f"{code_name} - leaves",
            unit="leaf",
            dynamic_ncols=True,
            position=tqdm_position if tqdm_position is not None else 0,
            leave=False,
            disable=tqdm_disable,
        )
        from threading import Lock

        _bar_lock = Lock()

        def _mk_done_cb(bar):
            def _done_cb(_future):
                with _bar_lock:
                    if bar.disable:
                        return
                    bar.update(1)
                    if bar.n >= bar.total:
                        try:
                            bar.close()
                        except Exception:
                            pass

            return _done_cb

        done_cb = _mk_done_cb(section_bar)

        for sec, url, p, lp in work:
            fut = executor.submit(
                fetch_leaf_threadsafe,
                sec,
                url,
                p,
                lp,
                state,
                session,
                f_out,
                record_queue,
            )
            fut.add_done_callback(done_cb)
            futures.append(fut)
    else:
        total_leaves = len(work)
        bar = tqdm(
            total=total_leaves,
            desc=f"{code_name} - leaves",
            unit="leaf",
            dynamic_ncols=True,
            position=tqdm_position if tqdm_position is not None else 0,
            leave=False,
            disable=tqdm_disable,
        )
        for sec, url, p, lp in work:
            scrape_leaf(page, state, sec, url, p, lp, f_out)
            bar.update(1)
        bar.close()


def scrape_leaf(
    parent_page: Page,
    state: str,
    sec_name: str,
    sec_url: str,
    path_so_far: List[str],
    lex_order: List[int],
    f_out,
) -> None:
    """
    Scrape a leaf node (actual statute) from FindLaw.

    Args:
    - parent_page (Page): The Playwright page object from the parent section.
    - state (str): The state abbreviation.
    - sec_name (str): The name of the statute.
    - sec_url (str): The URL of the statute.
    - path_so_far (List[str]): The hierarchical path to this statute.
    - f_out: The output file handle.
    """

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Referer": "https://www.google.com/",
        }
    )
    response = session.get(sec_url, timeout=30)
    if response.status_code != 200:
        logging.error(
            f"Failed to retrieve statute page at {sec_url}. Status code: {response.status_code}"
        )
        return
    soup = BeautifulSoup(response.content, "html.parser")
    statute_name = soup.select("h1")[0].get_text(strip=True)
    content_div = soup.select("div.codes-content p")[0].get_text(strip=True)
    parsed_heading = _parse_statute_heading(statute_name, sec_name)
    hierarchy = _build_hierarchy_metadata(
        path_so_far,
        lex_order,
        state,
        leaf_title=parsed_heading.get("article_title", ""),
        leaf_heading=parsed_heading.get("heading", ""),
        leaf_code=parsed_heading.get("article_code", "") or sec_name,
    )
    statute_data = {
        "url": sec_url,
        "state": state.upper(),
        "path": "›".join(path_so_far),
        "path_nodes": path_so_far,
        "parent_path": "›".join(path_so_far[:-1]) if len(path_so_far) > 1 else "",
        "title": f"{state.upper()} Statutes › {' › '.join(path_so_far)}",
        "univ_cite": False,
        "citation": f"{state.upper()} Stat § {sec_name} (2023)",
        "statute_name": statute_name,
        "article_heading": parsed_heading.get("heading", ""),
        "article_code": parsed_heading.get("article_code", "") or sec_name,
        "article_title": parsed_heading.get("article_title", ""),
        "content": content_div,
        "lex_path": lex_order,
        "hierarchy": hierarchy,
        "parent_nodes": hierarchy["parent_nodes"],
    }
    # we write to a jsonl with the state abbreviation as the filename in the folder output_dir
    import json

    f_out.write(json.dumps(statute_data, ensure_ascii=False) + "\n")


import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests

WRITE_LOCK = Lock()


def _is_blocked_findlaw_response(response_text: str) -> bool:
    txt = (response_text or "").lower()
    return "just a moment" in txt or "performing security verification" in txt


def _build_statute_record(sec_name, sec_url, path_so_far, lex_path, state, statute_name, content_div):
    parsed_heading = _parse_statute_heading(statute_name, sec_name)
    hierarchy = _build_hierarchy_metadata(
        path_so_far,
        lex_path,
        state,
        leaf_title=parsed_heading.get("article_title", ""),
        leaf_heading=parsed_heading.get("heading", ""),
        leaf_code=parsed_heading.get("article_code", "") or sec_name,
    )
    return {
        "url": sec_url,
        "state": state.upper(),
        "path": "›".join(path_so_far),
        "path_nodes": path_so_far,
        "parent_path": "›".join(path_so_far[:-1]) if len(path_so_far) > 1 else "",
        "title": f"{state.upper()} Statutes › {' › '.join(path_so_far)}",
        "univ_cite": False,
        "citation": f"{state.upper()} Stat § {sec_name} (2023)",
        "statute_name": statute_name,
        "article_heading": parsed_heading.get("heading", ""),
        "article_code": parsed_heading.get("article_code", "") or sec_name,
        "article_title": parsed_heading.get("article_title", ""),
        "content": content_div,
        "lex_path": lex_path,
        "hierarchy": hierarchy,
        "parent_nodes": hierarchy["parent_nodes"],
    }


def _write_statute_record(record_queue, f_out, data):
    import json

    line = json.dumps(data, ensure_ascii=False)
    if record_queue is not None:
        record_queue.put(line + "\n")
    else:
        with WRITE_LOCK:
            f_out.write(line + "\n")


def _scrape_leafs_with_playwright_fallback(work, state, output_file):
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup
    import json

    profile_dir = os.path.join("findlaw_codes", "playwright_profile")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            viewport={"width": 1366, "height": 900},
        )
        page = context.pages[0] if context.pages else context.new_page()

        tmp_file = f"{output_file}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f_out:
            for sec_name, sec_url, path_so_far, lex_path in tqdm(work, desc="Fetching (Playwright)"):
                page.goto(sec_url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(2000)
                html = page.content()
                if _is_blocked_findlaw_response(html):
                    continue
                soup = BeautifulSoup(html, "html.parser")
                h1 = soup.select_one("h1")
                if h1 is None:
                    continue
                statute_name = h1.get_text(strip=True)
                content_div = clean_paragraphs(soup)
                if not content_div:
                    continue
                data = _build_statute_record(
                    sec_name,
                    sec_url,
                    path_so_far,
                    lex_path,
                    state,
                    statute_name,
                    content_div,
                )
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

        context.close()
        if os.path.exists(output_file):
            os.remove(output_file)
        os.replace(tmp_file, output_file)


async def _scrape_leafs_with_playwright_fallback_async(page, work, state, output_file, section_url):
    from bs4 import BeautifulSoup
    import asyncio
    import json

    # The page/session is already validated on the section root before this fallback is called.
    # Reuse that context to avoid losing Cloudflare/session state.
    context = page.context
    worker_count = max(1, min(4, len(work) // 40 + 1))
    pages = [page]
    for _ in range(worker_count - 1):
        pages.append(await context.new_page())

    queue = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)

    lines = []
    lines_lock = asyncio.Lock()
    pbar = tqdm(total=len(work), desc=f"Fetching (Playwright x{worker_count})")

    async def _worker(worker_page):
        local_lines = []
        while True:
            try:
                sec_name, sec_url, path_so_far, lex_path = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            html = ""
            blocked = True
            for attempt in range(2):
                await worker_page.goto(sec_url, wait_until="domcontentloaded", timeout=45000)
                try:
                    await worker_page.wait_for_selector("h1, div.codes-content p", timeout=5000)
                except Exception:
                    pass
                html = await worker_page.content()
                blocked = _is_blocked_findlaw_response(html)
                if not blocked:
                    break
                await worker_page.wait_for_timeout(800 * (attempt + 1))

            if not blocked:
                soup = BeautifulSoup(html, "html.parser")
                h1 = soup.select_one("h1")
                if h1 is not None:
                    statute_name = h1.get_text(strip=True)
                    content_div = clean_paragraphs(soup)
                    if content_div:
                        data = _build_statute_record(
                            sec_name,
                            sec_url,
                            path_so_far,
                            lex_path,
                            state,
                            statute_name,
                            content_div,
                        )
                        local_lines.append(json.dumps(data, ensure_ascii=False) + "\n")

            pbar.update(1)

        if local_lines:
            async with lines_lock:
                lines.extend(local_lines)

    await asyncio.gather(*(_worker(worker_page) for worker_page in pages))
    pbar.close()

    for worker_page in pages[1:]:
        await worker_page.close()

    tmp_file = f"{output_file}.tmp"
    if len(lines) == 0:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise RuntimeError(f"Playwright fallback produced no records for {section_url}")

    with open(tmp_file, "w", encoding="utf-8") as f_out:
        f_out.writelines(lines)

    if os.path.exists(output_file):
        os.remove(output_file)
    os.replace(tmp_file, output_file)


def fetch_leaf_threadsafe(
    sec_name, sec_url, path_so_far, lex_path, state, session, f_out, record_queue=None
):
    # light retry with backoff
    for attempt in range(4):
        try:
            r = session.get(sec_url, timeout=30)
            if r.status_code == 200:
                if _is_blocked_findlaw_response(r.text):
                    break
                soup = BeautifulSoup(r.content, "html.parser")
                statute_name = soup.select_one("h1").get_text(strip=True)
                content_div = clean_paragraphs(soup)
                data = _build_statute_record(
                    sec_name,
                    sec_url,
                    path_so_far,
                    lex_path,
                    state,
                    statute_name,
                    content_div,
                )
                _write_statute_record(record_queue, f_out, data)
                return
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1) + random.random())
    return

async def scrape_section_url_async(section_url: str, state: str, output_file: str) -> None:
    from queue import Queue
    from threading import Thread
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from playwright.async_api import async_playwright

    # Writer thread
    record_q = Queue()
    def _writer():
        with open(output_file, "a", encoding="utf-8") as f:
            while True:
                item = record_q.get()
                if item is None:
                    break
                f.write(item)
    writer_t = Thread(target=_writer, daemon=True)
    writer_t.start()

    leaf_session = requests.Session()
    leaf_session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Referer": "https://www.google.com/",
    })

    futures_list = []
    executor = ThreadPoolExecutor(max_workers=8)
    profile_dir = os.path.join("findlaw_codes", "playwright_profile")

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            viewport={"width": 1366, "height": 768},
        )
        page = context.pages[0] if context.pages else await context.new_page()

        print(f"Loading {section_url} ...")
        await page.goto(section_url, wait_until="domcontentloaded", timeout=60000)
        found_tree = False
        for _ in range(12):
            if await page.locator(".fl-expandable-tree-accordion > .fl-accordion-item").count() > 0:
                found_tree = True
                break
            await page.wait_for_timeout(5000)
        if not found_tree:
            raise RuntimeError(f"Accordion tree not visible for {section_url}")

        # Collect all leaf links by expanding accordions
        work = await _collect_links_async(page, section_url, [state.upper()], [1])
        print(f"Found {len(work)} statutes. Fetching...")

        # If the first leaf is blocked, switch the whole run to Playwright-only.
        probe_session = requests.Session()
        probe_session.headers.update(leaf_session.headers)
        try:
            probe_response = probe_session.get(work[0][1], timeout=30)
            if probe_response.status_code != 200 or _is_blocked_findlaw_response(probe_response.text):
                record_q.put(None)
                writer_t.join()
                if os.path.exists(output_file) and os.path.getsize(output_file) == 0:
                    os.remove(output_file)
                await _scrape_leafs_with_playwright_fallback_async(page, work, state, output_file, section_url)
                await context.close()
                print(f"Done. Saved to {output_file}")
                return
        finally:
            probe_session.close()

        await context.close()

    # Submit all leaf fetches to thread pool
    for sec_name, url, path, lex in work:
        fut = executor.submit(
            fetch_leaf_threadsafe,
            sec_name, url, path, lex, state, leaf_session, None, record_q
        )
        futures_list.append(fut)

    for _ in tqdm(as_completed(futures_list), total=len(futures_list), desc="Fetching"):
        pass
    executor.shutdown(wait=True)

    record_q.put(None)
    writer_t.join()
    print(f"Done. Saved to {output_file}")


async def _wait_links_or_subaccordions_async(scope, timeout=8000):
    """
    Wait briefly for either direct leaf links or direct nested accordion items
    under the current scope. Returns True if something appears.
    """
    sel = (
        ":scope > .fl-accordion-content .fl-recursive-tree-accordion a[href], "
        ":scope > .fl-accordion-content .fl-recursive-tree-accordion-list .fl-accordion-item"
    )
    try:
        if await scope.locator(sel).count() > 0:
            return True
    except Exception:
        pass
    try:
        await scope.locator(sel).first.wait_for(state="attached", timeout=timeout)
        return True
    except Exception:
        return False


async def _collect_links_async(page, section_url: str, base_path: list, base_lex: list) -> list:
    """Expand all accordions and collect (name, url, path, lex) tuples."""
    from urllib.parse import urljoin
    results = []

    items = page.locator(".fl-expandable-tree-accordion > .fl-accordion-item")
    count = await items.count()

    for i in range(count):
        item = items.nth(i)
        btn = item.locator(":scope > h2 .fl-accordion-button")
        await btn.wait_for(state="attached", timeout=10000)

        try:
            top_label = await btn.locator(".fl-text-left").inner_text(timeout=3000)
            top_label = top_label.strip()
        except Exception:
            top_label = f"Section {i+1}"

        if (await btn.get_attribute("aria-expanded") or "").lower() != "true":
            await btn.click()

        await _wait_links_or_subaccordions_async(item, timeout=8000)

        results.extend(await _collect_recursive_async(item, section_url, base_path + [top_label], base_lex + [i+1]))

    return results


async def _collect_recursive_async(scope, section_url: str, base_path: list, base_lex: list) -> list:
    from urllib.parse import urljoin
    results = []

    # If content is still loading, wait briefly before counting.
    await _wait_links_or_subaccordions_async(scope, timeout=8000)

    # Case A: direct links
    link_list = scope.locator(
        ":scope > .fl-accordion-content .fl-recursive-tree-accordion a[href]"
    )
    direct_count = await link_list.count()

    if direct_count > 0:
        for k in range(direct_count):
            a = link_list.nth(k)
            sec_name = (await a.inner_text()).strip()
            href = await a.get_attribute("href")
            if not href:
                continue
            url = urljoin(section_url, href)
            results.append((sec_name, url, base_path + [sec_name], base_lex + [k+1]))

    # Case B: nested accordions
    nested_items = scope.locator(
        ":scope > .fl-accordion-content .fl-recursive-tree-accordion-list .fl-accordion-item"
    )
    nested_count = await nested_items.count()

    for j in range(nested_count):
        n_item = nested_items.nth(j)
        n_btn = n_item.locator("button.fl-accordion-button")
        try:
            await n_btn.wait_for(state="attached", timeout=3000)
        except Exception:
            continue

        try:
            label = await n_btn.locator(".fl-text-left").inner_text(timeout=2000)
            label = label.strip()
        except Exception:
            label = f"Section {j+1}"

        if (await n_btn.get_attribute("aria-expanded") or "").lower() != "true":
            await n_btn.click()

        # NEU: aktiv auf Links oder weitere nested items warten statt fixed timeout
        try:
            await n_item.locator(
                ":scope > .fl-accordion-content .fl-recursive-tree-accordion a[href], "
                ":scope > .fl-accordion-content .fl-recursive-tree-accordion-list .fl-accordion-item"
            ).first.wait_for(state="attached", timeout=8000)
        except Exception:
            pass

        results.extend(await _collect_recursive_async(
            n_item, section_url, base_path + [label], base_lex + [j+1]
        ))

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape FindLaw codes for a given state.")
    parser.add_argument("state", help="State abbreviation (e.g., nd, ky, pa)")
    parser.add_argument("--output-dir", "-o", default=DEFAULT_DIR,
                        help=f"Output directory for JSONL (default: {DEFAULT_DIR})")
    parser.add_argument("--processes", "-p", type=int, default=6,
                        help="Number of browser processes to use (default: 6)")
    parser.add_argument("--threads", "-t", type=int, default=8,
                        help="Threads per process for leaf fetches (default: 8)")
    parser.add_argument("--chunks-per-proc", "-c", type=int, default=4,
                        help="Work chunk factor per process for progress responsiveness (default: 4)")
    args = parser.parse_args()

    scrape_state(
        args.state,
        args.output_dir,
        processes=args.processes,
        threads=args.threads,
        chunks_per_proc=args.chunks_per_proc,
    )
