import os
import requests
import asyncio
from playwright.async_api import async_playwright
import os
import time
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import random
from requests.exceptions import RequestException

import requests
from duckduckgo_search import DDGS
from langchain_community.document_loaders import UnstructuredHTMLLoader
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import math
import logging
import tempfile

from lmms_eval.tasks.mmsearch.utils.web_content_utils import *
from lmms_eval.tasks.mmsearch.constants import *

from loguru import logger as eval_logger


### Proxy setting
def get_proxy_settings():
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    
    proxies = {}
    if http_proxy:
        proxies['http'] = http_proxy
    if https_proxy:
        proxies['https'] = https_proxy
    
    # Try to obtain environ proxies
    if not proxies:
        try:
            system_proxies = requests.utils.get_environ_proxies("")
            if system_proxies:
                proxies = system_proxies
        except Exception as e:
            eval_logger.warning(f"Cannot obtain environ proxies: {e}")
    
    return proxies

PROXY = get_proxy_settings() # get proxy if exist

### Brief Results

def search_text_brief_result(query, max_result_num, screenshot_dir):
    os.makedirs(screenshot_dir, exist_ok=True)
    return asyncio.run(run_query(query, screenshot_dir, max_result_num))

async def run_query(query: str, screenshot_dir_path: str, max_result_num: int):
    engine = DDGSQueryRun(max_results=max_result_num)
    results = await engine(query, screenshot_dir_path)
    return results

## Search Engine API
class RapidAPI:
    def __init__(self, rapidapi_name):
        self.rapidapi_name = rapidapi_name
        self.ddgs = DDGS(proxy=PROXY['https'], timeout=50) if len(PROXY) != 0 else DDGS(timeout=50)
    
    def query(self, text: str, max_results: int) -> List[Dict[str, Any]]:
        initial_delay = 1
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                time.sleep(5)  # Avoid frequent requests
                response = list(self.ddgs.text(' '.join(text.strip("'").split(' ')[:100]), max_results=max_results))
                return response[:max_results]
            except Exception as e:
                error_message = str(e)
                if "202" in error_message or "Accepted" in error_message:
                    delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Received 202 status code, waiting {delay:.2f} seconds before retrying... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                elif isinstance(e, RequestException):
                    print(f"Network error: {e}")
                    time.sleep(random.uniform(1, 3))
                else:
                    print(f"Unknown error: {e}")
                    raise ValueError
## API: Search Engine Retrieval + Screenshot of top section
class DDGSQueryRun:
    name = "duckduckgo_search"
    signature = f"{name}(query: str) -> str"
    
    def __init__(self, max_results: int, rapidapi_name: str = "one"):
        self.max_results = max_results
        self.api_wrapper = RapidAPI(rapidapi_name)
    
    async def __call__(self, query: str, screenshot_dir_path: str) -> List[Dict[str, Any]]:
        try:
            output = self.api_wrapper.query(query, max_results=self.max_results+20) # account for error website
        except Exception as e:
            eval_logger.error(f"DDGSQueryRun call failed: {e}")
            output = []

        evidences = []
        for idx, result in enumerate(output):
            evidence = {
                "title": result["title"],
                "snippet": result.get("description", result.get("body", "")),
                'url': result['href'],
                'screenshot_path': os.path.join(screenshot_dir_path, f"{idx}.jpg")
            }
            success = await take_screenshot_async(evidence['url'], os.path.join(screenshot_dir_path, f"{idx}.jpg"))
            if success:
                evidences.append(evidence)
            if len(evidences) == self.max_results:
                break
    
        if not evidences:
            evidences = None
        
        return evidences
## Screenshot of top section. Set the size to be 1024*1024
async def take_screenshot_async(url: str, screenshot_path: str, timeout: int = BRIEF_TIMEOUT):
    async with async_playwright() as p:
        if len(PROXY) != 0:
            browser = await p.chromium.launch(headless=True, proxy={'server': PROXY['https']})
            context = await browser.new_context(
                user_agent=USER_AGENT,
                proxy={'server': PROXY['https']},
                viewport={'width': 1024, 'height': 1024}
            )
        else:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=USER_AGENT,
                viewport={'width': 1024, 'height': 1024}
            )


        page = await context.new_page()
        try:
            await page.goto(url, wait_until='load', timeout=timeout)
            await page.screenshot(path=screenshot_path)
            eval_logger.info(f"Successfully taking screenshot of current state: {url}")
        except PlaywrightTimeoutError:
            eval_logger.info(f"Timeout occurred while loading {url}. Taking screenshot of current state.")
            try:
                await page.screenshot(path=screenshot_path)
            except Exception as e:
                eval_logger.info(f"An error occurred while taking screenshot of {url}: {str(e)}")
                await context.close()
                await browser.close()
                return False
        except Exception as e:
            eval_logger.info(f"An error occurred while taking screenshot of {url}: {str(e)}")
            await context.close()
            await browser.close()
            return False
        finally:
            await context.close()
            await browser.close()
        return True




### Full page content
def search_url_full_result(urls, screenshot_dir):

    results = []
    for idx, url in enumerate(urls):
        save_dir_path = os.path.join(screenshot_dir, str(idx))
        os.makedirs(save_dir_path, exist_ok=True)
        fullpage_success = take_fullpage_screenshot(url, f"{save_dir_path}/fullpage.png")
        if not fullpage_success:
            eval_logger.info(f"take_fullpage_screenshot failed. Save a blank image")
            # Create a 512x512 pixel blank image
            fig, ax = plt.subplots(figsize=(512/100, 512/100), dpi=100)
            # Remove coordinate axes
            ax.axis('off')
            # Add text
            ax.text(0.5, 0.5, 'No Content', fontsize=50, ha='center', va='center')
            # Adjust layout and set image boundaries
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # Save image
            plt.savefig(f"{save_dir_path}/fullpage.png", dpi=100, pad_inches=0)

        results.append(dict(
            content=get_fullpage_content(url),
            screenshot_fullpage_path=f"{str(idx)}/fullpage.png"),
        )
    return results

## Fullpage screenshot
def take_fullpage_screenshot(url: str, screenshot_path: str, timeout: int = FULLPAGE_TIMEOUT):
    return asyncio.run(_take_fullpage_screenshot(url, screenshot_path))

async def _take_fullpage_screenshot(url: str, screenshot_path: str, timeout: int = FULLPAGE_TIMEOUT):
    async with async_playwright() as p:
        if len(PROXY) != 0:
            browser = await p.chromium.launch(headless=True, proxy={'server': PROXY['https']})
            context = await browser.new_context(
                user_agent=USER_AGENT,
                proxy={'server': PROXY['https']},
                viewport={'width': 512, 'height': 512},
                is_mobile=True
            )
        else:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=USER_AGENT,
                viewport={'width': 512, 'height': 512},
                is_mobile=True,
            )

        page = await context.new_page()
        try:
            await page.goto(url, wait_until='networkidle', timeout=timeout)
            # Scroll the full page for all image to be visible
            await scroll_full_page(page)
            await page.wait_for_timeout(2000)
            await page.screenshot(path=screenshot_path, full_page=True)
            eval_logger.info(f"Successfully took full page screenshot: {url}")
            return True
        except PlaywrightTimeoutError:
            eval_logger.info(f"Timeout occurred while loading {url}. Taking screenshot of current state.")
            try:
                await scroll_full_page(page)
                await page.wait_for_timeout(2000)
                await page.screenshot(path=screenshot_path, full_page=True)
                return True
            except Exception as e:
                eval_logger.error(f"An error occurred while taking full page screenshot of {url}: {str(e)}")
                return False
        except Exception as e:
            eval_logger.error(f"An error occurred while taking full page screenshot of {url}: {str(e)}")
            return False
        finally:
            await context.close()
            await browser.close()

## Fullpage textual content
def get_fullpage_content(url: str, timeout: int = FULLPAGE_TIMEOUT) -> Optional[str]:
    return asyncio.run(_get_fullpage_content(url, timeout))

async def _get_fullpage_content(url: str, timeout: int = FULLPAGE_CONTENT_TIMEOUT) -> Optional[str]:
    async with async_playwright() as p:
        if len(PROXY) != 0:
            browser = await p.chromium.launch(headless=True, proxy={'server': PROXY['https']})
            context = await browser.new_context(
                user_agent=USER_AGENT,
                proxy={'server': PROXY['https']},
            )
        else:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=USER_AGENT,
            )


        page = await context.new_page()
        
        try:
            # Set navigation timeout (milliseconds)
            page.set_default_navigation_timeout(timeout)
            
            # Navigate to the specified URL
            await page.goto(url, wait_until='load', timeout=timeout)
            
            html_content = await page.content() 

            # use UnstructuredHTMLLoader to extract main content
            # setup a temporary file
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.html', delete=False) as temp_file:
                temp_file.write(html_content)
                temp_file_path = temp_file.name

            loader = UnstructuredHTMLLoader(temp_file_path)
            data = loader.load()
            # delete the temporary file
            os.unlink(temp_file_path)
            main_text = data[0].page_content

            eval_logger.info(f"Successfully scraping content of current state: {url}")

            return main_text
        
        except PlaywrightTimeoutError:
            eval_logger.info(f"Timeout occurred while loading {url}. Scraping content of current state.")
            try:
                html_content = await page.content() 
                main_text = extract_main_content(html_content)
                return main_text
            except Exception as e:  
                eval_logger.info(f"An error occurred while processing content of {url}: {str(e)}")
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
        finally:
            await browser.close()


### Utils for screenshot
async def scroll_full_page(page, max_height=10000):
    return await page.evaluate(f"""
        async () => {{
            const js_height = () => {{
                try {{
                    return Math.min(document.body.clientHeight, {max_height});
                }} catch (error) {{
                    console.warn("Unable to get clientHeight, using max_height:", error);
                    return {max_height};
                }}
            }};
            
            let height = js_height();
            let k = 1;
            const scrollStep = 300;  // Scroll step length
            const pauseDuration = 1000;  // Pause duration after each scroll (milliseconds)
            const maxHeight = {max_height};  // Maximum scroll height

            while (true) {{
                if (k * scrollStep < height && k * scrollStep < maxHeight) {{
                    window.scrollTo(0, k * scrollStep);
                    await new Promise(resolve => setTimeout(resolve, pauseDuration));
                    height = js_height();
                    k += 1;
                }} else {{
                    break;
                }}
            }}
            
            // Scroll back to top
            window.scrollTo(0, 0);
            await new Promise(resolve => setTimeout(resolve, pauseDuration));
        }}
    """)

async def load_all_images(page):
    # Save current scroll position
    original_position = await page.evaluate('() => ({ x: window.scrollX, y: window.scrollY })')

    # Find all image elements
    locators = page.locator('//img')

    # Create an array of Promises, each corresponding to the loading of an image
    promises = await locators.evaluate_all("""
    elements => elements.map(img => {
        if (img.complete) return Promise.resolve();
        return new Promise(resolve => {
            img.onload = resolve;
            img.onerror = resolve;  // Also handle loading failure
            // If the image doesn't have a src, it might be a lazy-loaded image
            if (!img.src && img.dataset.src) {
                img.src = img.dataset.src;
            }
        });
    })
    """)

    # Wait for all images to finish loading
    await page.evaluate('promises => Promise.all(promises)', promises)

    # Restore original scroll position
    await page.evaluate('position => window.scrollTo(position.x, position.y)', original_position)

    # Give the page some time to stabilize
    await page.wait_for_timeout(1000)

### Search image for google lens. Only will be used for new queries to MMSearch-Engine. Can only be used with English Browers.
def search_by_image(url, screenshot_path):
    return asyncio.run(_search_by_image(url, screenshot_path))

async def _search_by_image(image_url, screenshot_path='search_results.png', delay=5., headless=True):
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, args=['--lang=en-US'])
        context = await browser.new_context(            
            locale='en-US',
            viewport={'width': 1280, 'height': 800}
        )
        page = await context.new_page()

        await page.goto('https://images.google.com')
        await page.wait_for_selector('div[role="button"][aria-label="Search by image"]', state='visible')
        await page.click('div[role="button"][aria-label="Search by image"]')
        await page.wait_for_selector('input[placeholder="Paste image link"]', state='visible')
        await page.fill('input[placeholder="Paste image link"]', image_url)
        await page.wait_for_selector('div[jsname="ZtOxCb"]', state='visible')
        await page.click('div[jsname="ZtOxCb"]')

        await page.wait_for_selector('img', state='visible')
        await load_all_images(page)
        await asyncio.sleep(delay)

        # Extract search results
        result_cards = await page.query_selector_all('.Vd9M6')
        count = 0
        for card in result_cards:
            image_element = await card.query_selector('img.wETe9b')
            snippet_element = await card.query_selector('.UAiK1e')
            a_element = await card.query_selector('a.GZrdsf')
            
            if image_element and snippet_element:
                image_url = await image_element.get_attribute('src')
                snippet = await snippet_element.inner_text()
                web_url = await a_element.get_attribute('href')

                if image_url.startswith('dat:image'):
                    print(image_url)
                    continue
                
                results.append({
                    'image_url': image_url,
                    'snippet': snippet,
                    "web_url": web_url
                })
                count += 1
                if count == IMAGE_SEARCH_RESULT:
                    break

        await page.screenshot(path=screenshot_path, full_page=True)
        await browser.close()

    return results






