import numpy
from PIL import Image

# f = open('lena16bit.raw', 'rb') # opening a binary file
# f = open('lena16bit.bin', 'rb') # opening a binary file
f = open('ducks-16.bin', 'rb') # opening a binary file

content = f.read()

# img = Image.fromarray(content, 'RGB')
# img = Image.frombytes('I;16B', (256,256), content, 'raw')
img = Image.frombytes('I;16B', (260,260), content)
array = numpy.array(img)
array.astype(numpy.uint16)
array = array / 100
print(array)
# r = img.split()
# r = r[0].point(lambda i: i * 5000)
img = Image.fromarray(array)
img.convert('RGB').save('lena16bit2.png')
# img.convert('RGB').save('my.png')
# img.save('my.png')
img.show()

# make file
# array.astype('uint16').tofile('uint16_file.bin')
# newFile = open("lena16bit2.raw", "wb")
# # write to file
# newFileByteArray = bytearray(array)
# newFile.write(newFileByteArray)