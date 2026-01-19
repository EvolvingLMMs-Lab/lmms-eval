import re

from bs4 import BeautifulSoup


def clean_text(text):
    # Remove excess whitespace characters
    text = re.sub(r"\s+", " ", text).strip()
    # Remove excess newline characters
    text = re.sub(r"\n+", "\n", text)
    return text


def extract_main_content(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, navigation, and footer elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # Try to find the main content area (assuming it uses <main> tag or id/class containing "content")
    main_content = soup.find("main") or soup.find(id=re.compile("content", re.I)) or soup.find(class_=re.compile("content", re.I))

    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        # If no clear main content area is found, use the content of <body>
        text = soup.body.get_text(separator="\n", strip=True)

    return clean_text(text)
