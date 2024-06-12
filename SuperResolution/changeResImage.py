from PIL import Image

im = Image.open("pdf-test.png")
im = im.resize((125, 120))
im.save("pdf-test125.png", dpi=(300,300))