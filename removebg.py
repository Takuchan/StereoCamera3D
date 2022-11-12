rom rembg import remove
from PIL import Image
input_path f= 'test.png'
print("画像すよく")
output_path = 'output.png'
input = Image.open(input_path)
output = remove(input)
output.save(output_path)
