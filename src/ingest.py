import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify



def fetch_page(url:str)-> str:
    response = httpx.get(url)
    soup = BeautifulSoup(response.text,"html.parser")
    main_content = soup.find("main")
    print(main_content)
    return markdownify(str(main_content))

def save(content:str, filename:str):
    with open(f"knowledge_base/{filename}", "w") as f:
        f.write(content)


if __name__ == "__main__":
    url = "https://tokio.rs/tokio/tutorial"
    content = fetch_page(url)
    save(content, "tokio_tutorial.md")
    print(f"Saved {len(content)} characters")