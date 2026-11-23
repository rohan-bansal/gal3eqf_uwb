import sys, glob
from PIL import Image
files = sorted(glob.glob('slide-*.jpg'))
for k in range(0, len(files), 4):
    ims = [Image.open(f) for f in files[k:k+4]]
    w, h = ims[0].size
    sheet = Image.new('RGB', (2 * w + 10, 2 * h + 10), 'black')
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % 2) * (w + 10), (i // 2) * (h + 10)))
    sheet.save(f'sheet-{k // 4 + 1}.jpg', quality=85)
print(len(files), 'slides')
