def parse_line(line_txt):
    """Parser a single entry of the dat file.

    Parameters
    ==========
    line_txt: str
    an entry in pure text, ending WITHOUT '\\n'"""
    y, *x_str = line_txt.split(' ')
    x_list = [x.split(":") for x in x_str]
    return y, x_list


def dat_parser(dat_path, line_num=0, skip=0):
    """customable dat parser

    Parameters
    ==========
    line_num : 0, int
        number of entries to be read
    skip: 0, int
        skip initial some entries

    Note
    ====
    By default it reads all entries from the file.

    If the number given in `line_num` or `skip` is larger than the
    total entries of the file, no further entries will be return
    """
    dat_f = open(dat_path)
    for i, line_txt in enumerate(dat_f):
        if line_num > 0 and i >= line_num + skip:
            break
        elif i < skip:
            continue
        else:
            yield parse_line(line_txt[:-1])
    dat_f.close()
