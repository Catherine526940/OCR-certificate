import os

label_path = r"D:\DLDataset\label.txt"
root_path = r"D:\DLDataset\line-01\train\cqz-enhanced\17"


def create_label_file(file_path, label:str):
    with open(file_path, "w", encoding="utf8") as file:
        label = label.replace("：", ":")
        file.write(label)


with open(label_path, "r", encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        file_name, label = line.strip("\n").split(" ")
        file_name = file_name[:-3] + "txt"
        create_label_file(os.path.join(root_path, file_name), label)
