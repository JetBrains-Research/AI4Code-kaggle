def get_string_from_code(node, code_list, lines):
    line_start = node.start_point[0]
    line_end = node.end_point[0]
    char_start = node.start_point[1]
    char_end = node.end_point[1]
    if line_start != line_end:
        code_list.append(
            ' '.join([lines[line_start][char_start:]] + lines[line_start + 1:line_end] + [lines[line_end][:char_end]]))
    else:
        code_list.append(lines[line_start][char_start:char_end])


def my_traverse(node, code, code_list):
    lines = code.split('\n')
    if node.child_count == 0:
        get_string_from_code(node, code_list, lines)
    elif node.type == 'string':
        get_string_from_code(node, code_list, lines)
    else:
        for n in node.children:
            my_traverse(n, code, code_list)

    return ' '.join(code_list)
