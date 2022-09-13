import utility


def main():
    data = utility.read_file_to_sents()
    data = utility.split_data(data)


if __name__ == "__main__":
    main()