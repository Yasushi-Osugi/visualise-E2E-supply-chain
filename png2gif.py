
# 
# <gifファイル生成の前提>
#
#1. ディレクトリー内にあるpngファイルをソート順にgifファイルにしている。
#2. 前処理のpngファイル出力で、マイナス週のソート順を保つために+100している
#3. pngファイル生成の前にディレクトリーに残っているpngファイルは削除しておく
#

from PIL import Image

import glob

files = sorted(glob.glob('./*.png'), reverse=False)  
#files = sorted(glob.glob('./*.png'), reverse=True)  

print('files',files)

images = list(map(lambda file : Image.open(file) , files))

images[0].save('image.gif' , save_all = True , append_images = images[1:] , 
                duration = 600 , loop = 0)


