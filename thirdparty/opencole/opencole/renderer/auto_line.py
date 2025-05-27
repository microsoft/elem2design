import skia
import huggingface_hub
from opencole.renderer.fonts import FontManager


font_manager = FontManager(
    huggingface_hub.hf_hub_download(
        repo_id="cyberagent/opencole",
        filename="resources/fonts.pickle",
        repo_type="dataset",
    )
)


def text_width(text, font, size, letter_spacing, capitalize):
    text = text.upper() if capitalize else text
    text = text.strip("\n")

    ttf_data = font_manager.lookup(font)
    typeface = skia.Typeface.MakeFromData(ttf_data)
    font = skia.Font(typeface, int(size))
    glyphs = font.textToGlyphs(text)

    total_width = sum(font.getWidths(glyphs))
    total_width += letter_spacing * max(0, len(glyphs) - 1)
    return total_width


def wrap_text_to_fit_box(w, text, font, size, letter_spacing, capitalize = False):    
    words = text.split()  # Split text into words
    lines = []
    current_line = []
    current_width = 0

    for word in words:
        word_width = text_width(word, font, size, letter_spacing, capitalize)
        if current_width + word_width <= w:  # Word fits in the current line
            current_line.append(word)
            current_width += word_width + text_width(" ", font, size, letter_spacing, capitalize)  # Add word and a space
        else:  # Start a new line
            lines.append(" ".join(current_line))
            current_line = [word]
            current_width = word_width + text_width(" ", font, size, letter_spacing, capitalize)  # Start with the current word

    if current_line:  # Add the last line
        lines.append(" ".join(current_line))

    return "\n".join(lines)
