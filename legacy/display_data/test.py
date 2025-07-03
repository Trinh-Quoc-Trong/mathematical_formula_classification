"""Visualize InkML handwriting traces.

Cách dùng (từ terminal):
    python test.py path/to/file.inkml

Script sẽ:
1. Phân tích file .inkml.
2. Vẽ lại toàn bộ nét vẽ bằng matplotlib.
3. Hiển thị LaTeX ground-truth làm tiêu đề (nếu có).
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib


def _parse_trace(trace_text: str) -> List[Tuple[float, float]]:
    """Chuyển chuỗi "x y, x y, …" thành list[(x, y)]."""
    points: List[Tuple[float, float]] = []
    for pair in trace_text.strip().split(','):
        if pair.strip():
            x_str, y_str = pair.strip().split()
            points.append((float(x_str), float(y_str)))
    return points


def visualize_inkml(file_path: Path, *, flip_y: bool = True, figsize=(8, 8)) -> None:
    """Đọc & hiển thị tất cả trace trong file .inkml.

    Args:
        file_path: Đường dẫn tới file .inkml.
        flip_y:   InkML gốc có trục y ngược với matplotlib -> lật lại cho dễ nhìn.
        figsize:  Kích thước figure.
    """
    # Tắt hoàn toàn mathtext parser
    matplotlib.rcParams['text.usetex'] = False
    
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Lấy LaTeX ground-truth nếu có.
    latex = None
    for ann in root.findall('.//{http://www.w3.org/2003/InkML}annotation'):
        if ann.get('type') == 'truth':
            latex = ann.text.strip()
            break

    # Vẽ tất cả trace.
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    ns = '{http://www.w3.org/2003/InkML}'
    for tr in root.findall(f'.//{ns}trace'):
        pts = _parse_trace(tr.text or '')
        if not pts:
            continue
        xs, ys = zip(*pts)
        if flip_y:
            ys = [-y for y in ys]
        ax.plot(xs, ys, linewidth=2, color='black')

    ax.set_aspect('equal')
    ax.axis('off')

    if latex:
        # Hiển thị LaTeX dưới dạng plain text, tránh mọi việc parse
        cleaned = latex.strip()
        # Sử dụng text thay vì figtext và escape $ để tránh mathtext
        cleaned = cleaned.replace('$', r'\$')
        fig.text(0.5, 0.02, f'LaTeX ground-truth: {cleaned}', 
                ha='center', va='bottom', fontsize=10, 
                transform=fig.transFigure, wrap=True)

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize InkML handwriting traces.')
    parser.add_argument('inkml_file', type=Path, help='Path to .inkml file')
    args = parser.parse_args()

    if not args.inkml_file.exists():
        parser.error(f'File not found: {args.inkml_file}')
    visualize_inkml(args.inkml_file)


if __name__ == '__main__':
    main()
