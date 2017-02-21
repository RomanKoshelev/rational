def fields(recs: list, width=8) -> str:
    s = ""
    t = "%%%ds: %%s\n" % width
    for r in recs:
        s += t % (r[0], r[1])
    return s
