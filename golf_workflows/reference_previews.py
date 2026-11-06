"""Contact sheets of saved P1-P10 reference renders; no inference required."""
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def create_reference_previews(folder):
    folder = Path(folder)
    positions = {}
    for path in folder.iterdir():
        match = re.match(r'^P(10|[1-9])(?:\D|$)', path.name, re.IGNORECASE)
        if path.is_dir() and match:
            index = int(match.group(1))
            if index in positions:
                raise ValueError(f'Duplicate preview position P{index} in {folder}')
            positions[index] = path
    try:
        title_font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 28)
        label_font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    except OSError:
        title_font = label_font = ImageFont.load_default()
    columns, tile_w, tile_h, margin, header = 5, 500, 406, 18, 66
    width = columns * tile_w + (columns + 1) * margin
    height = header + 2 * (tile_h + margin) + margin

    def sheet(filename, title, overlay=False):
        canvas = Image.new('RGB', (width, height), '#edf1f5')
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 17), title, font=title_font, fill='#15283b')
        for index in range(1, 11):
            left = margin + ((index - 1) % columns) * (tile_w + margin)
            top = header + ((index - 1) // columns) * (tile_h + margin)
            draw.rectangle((left, top, left + tile_w, top + tile_h), fill='white', outline='#ccd5df')
            path = positions.get(index)
            label = f'P{index}'
            if overlay and path and (path / 'joints.json').exists():
                info = json.loads((path / 'joints.json').read_text(encoding='utf-8')).get('reference_club_overlay', {})
                if info.get('applied'):
                    label += ' | ' + str(info.get('source_view', 'unknown'))
                    if info.get('inferred_club_landmarks'):
                        label += ' | inferred points'
            draw.text((left + 10, top + 9), label, font=label_font, fill='#15283b')
            source = path / filename if path else None
            if source and source.is_file():
                with Image.open(source) as source_image:
                    image = ImageOps.contain(source_image.convert('RGB'), (480, 360))
                    canvas.paste(image, (left + (tile_w - image.width) // 2, top + 40 + (360 - image.height) // 2))
            else:
                draw.text((left + 40, top + 175), 'Overlay unavailable' if overlay else 'Reference unavailable',
                          font=label_font, fill='#855b27')
        return canvas

    body = sheet('reference_3d.jpg', 'P1-P10 | 3D reference bodies')
    overlays = sheet('reference_3d_club_overlay.jpg', 'P1-P10 | 3D reference club overlays', overlay=True)
    combined = Image.new('RGB', (width, 2 * height), '#edf1f5')
    combined.paste(body, (0, 0))
    combined.paste(overlays, (0, height))
    files = {'body': 'reference_3d_preview.png', 'club_overlay': 'reference_3d_club_overlay_preview.png',
             'combined': 'reference_all_preview.png'}
    for key, image in [('body', body), ('club_overlay', overlays), ('combined', combined)]:
        image.save(folder / files[key])
    return files
