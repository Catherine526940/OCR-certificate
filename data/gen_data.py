import os


root = 'D:/DLDataset/line-01'  # 数据集根目录

sub_dir = 'test'  # 子集，我只选了40万数据

train_txt = open(f"data/{sub_dir}.txt",'w',encoding='UTF-8')

for tag in os.listdir(f"{root}/{sub_dir}"):
    file_dir = f"{root}/{sub_dir}/{tag}"
    for file_name in os.listdir(file_dir):
        file_path = f"{file_dir}/{file_name}"
        suffix = file_path.split('.')
        if suffix[1] == 'txt':
            fp = open(file_path,'r',encoding='utf-8')
            img_path = suffix[0]+'.jpg'
            train_txt.write(f'{img_path} {fp.readline().replace(" ","")}\n')
            fp.close()
