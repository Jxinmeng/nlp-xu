def handle(scrf):
    vocab = {}
    with open(scrf,"rb") as frd:
        for line in frd:
            tmp = line.strip()
