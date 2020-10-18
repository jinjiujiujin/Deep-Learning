from PIL import Image
import os

def merge():
    src_list = os.listdir(src)
    output_list = os.listdir(output)

    for i in src_list:
        for j in output_list:
            if i==j and not os.path.exists(save+i):
                print(i)
                src_img = Image.open(src+"/"+i)
                output_img = Image.open(output+"/"+i)
                w, h = src_img.size
                res = Image.new(src_img.mode, (w*2, h))
                res.paste(src_img, box=(0,0))
                res.paste(output_img, box=(w, 0))
                res.save(save+i)
                break

src = "originImg"
output = "sketch"
save="merge/"
merge()