from tqdm import tqdm
import glob


def combine_MSA(in_dir_path, out_file_path):
    with open(out_file_path, "w+") as out_file:
        for file in tqdm(glob.glob(in_dir_path + "/**/*", recursive=False)):
            if not file.endswith(".xml"):
                with open(file, "r") as in_file:
                    out_file.write(in_file.read() + "\n")


def combine_CA(in_dir_path, out_file_path):
    with open(out_file_path, "w+") as out_file:
        for file in tqdm(glob.glob(in_dir_path + "/*.txt", recursive=False)):
            if file.endswith(".txt"):
                with open(file, "r") as in_file:
                    out_file.write(in_file.read() + "\n")
