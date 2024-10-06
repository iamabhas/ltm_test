def append_to_file(file_name, content):
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(content + "\n")
