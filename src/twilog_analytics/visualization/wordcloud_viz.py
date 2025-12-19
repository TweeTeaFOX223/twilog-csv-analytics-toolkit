from __future__ import annotations

from typing import Dict, Optional

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud

matplotlib.use("Agg")


class WordCloudGenerator:
    """Generate word cloud images."""

    def __init__(self, font_path: Optional[str] = None) -> None:
        self.font_path = font_path or self._find_japanese_font()

    @staticmethod
    def _find_japanese_font() -> Optional[str]:
        preferred_fonts = [
            "Yu Gothic",
            "MS Gothic",
            "Meiryo",
            "IPAexGothic",
            "IPAGothic",
            "Hiragino Sans",
            "Noto Sans CJK JP",
        ]
        available_fonts = {f.name: f.fname for f in fm.fontManager.ttflist}
        for font_name in preferred_fonts:
            if font_name in available_fonts:
                return available_fonts[font_name]
        return None

    def generate_wordcloud(
        self,
        word_freq: Dict[str, int],
        width: int = 900,
        height: int = 500,
        background_color: str = "white",
        colormap: str = "viridis",
        max_words: int = 150,
    ) -> Image.Image:
        if not word_freq:
            return Image.new("RGB", (width, height), color=background_color)

        wc_kwargs = {
            "width": width,
            "height": height,
            "background_color": background_color,
            "colormap": colormap,
            "max_words": max_words,
            "relative_scaling": 0.5,
            "min_font_size": 10,
        }
        if self.font_path:
            wc_kwargs["font_path"] = self.font_path

        wordcloud = WordCloud(**wc_kwargs)
        wordcloud.generate_from_frequencies(word_freq)
        return wordcloud.to_image()

    def generate_wordcloud_figure(
        self,
        word_freq: Dict[str, int],
        width: int = 900,
        height: int = 500,
        background_color: str = "white",
        colormap: str = "viridis",
        max_words: int = 150,
        title: str = "Word Cloud",
    ) -> plt.Figure:
        image = self.generate_wordcloud(
            word_freq, width, height, background_color, colormap, max_words
        )
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(image, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=16, pad=20)
        plt.tight_layout()
        return fig

    @staticmethod
    def get_available_colormaps() -> Dict[str, str]:
        return {
            "viridis": "Viridis",
            "plasma": "Plasma",
            "inferno": "Inferno",
            "magma": "Magma",
            "cividis": "Cividis",
            "twilight": "Twilight",
            "Blues": "Blues",
            "Reds": "Reds",
            "Greens": "Greens",
            "Purples": "Purples",
            "Oranges": "Oranges",
            "RdYlBu": "RdYlBu",
            "Spectral": "Spectral",
            "coolwarm": "Coolwarm",
        }
