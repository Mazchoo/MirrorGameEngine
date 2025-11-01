from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from IPython.display import HTML


def show_syntax_highlighted_file(path, lexer_name="glsl", line_range=None):
    """Read certain lines of a file and syntax highlight them in html."""
    with open(path) as f:
        contents = f.read()

    if line_range is not None:
        start, end = line_range
        contents = "\n".join(contents.split("\n")[start:end])

    lexer = get_lexer_by_name(lexer_name)
    formatter = HtmlFormatter(style="colorful")
    highlighted_code = highlight(contents, lexer, formatter)
    highlighted_code = (
        f"<style>{formatter.get_style_defs('.highlight')}</style>{highlighted_code}"
    )
    return HTML(highlighted_code)
