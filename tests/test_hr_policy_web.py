from html.parser import HTMLParser
from pathlib import Path


WEB_DIR = Path(__file__).parents[1] / "use-cases" / "hr-policy-assistant" / "web"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.labels = set()
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if tag == "label" and "for" in attributes:
            self.labels.add(attributes["for"])
        if tag == "script" and "src" in attributes:
            self.scripts.append(attributes["src"])


def test_browser_client_has_accessible_controls_and_result_states():
    parser = PageParser()
    parser.feed((WEB_DIR / "index.html").read_text())

    controls = {"apiUrl", "pdfFile", "promptSelect", "ingestButton", "queryButton"}
    assert controls <= parser.ids
    assert {"apiUrl", "pdfFile", "promptSelect"} <= parser.labels
    assert {"ingestStatus", "resultState", "warningPanel", "answerPanel", "evidencePanel"} <= parser.ids
    assert parser.scripts == ["app.mjs"]
    assert (WEB_DIR / "styles.css").is_file()
