
def indent_with_tabs(text: str, num: int) -> str:
    """
    Вставляет num табуляций в начало каждой строки
    """
    indent_space = '\t' * num
    text = indent_space + text.replace('\n', '\n' + indent_space)
    return text
