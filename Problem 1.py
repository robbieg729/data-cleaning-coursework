from zipfile import ZipFile
import os

out_directory = "Kaggle"
try:
    os.mkdir(out_directory)
except:
    None

file_name = "kaggle-survey.zip"

with ZipFile(file_name, "r") as zip:
    zip.extractall(out_directory)

new_directory = out_directory + "/" + file_name.replace(".zip", "")

for file in os.scandir(new_directory):
    if file.path.endswith(".zip"):
        with ZipFile(file.path, "r") as zip:
            zip.extractall(file.path.replace(".zip", ""))

print("Done")