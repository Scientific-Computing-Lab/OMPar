import tree_sitter
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

C_LANGUAGE = Language(tsc.language())
CPP_LANGUAGE = Language(tscpp.language())

languages = {'c': C_LANGUAGE,
            'cpp': CPP_LANGUAGE}


def parse_code(code, language):
    parser = Parser(language)
    tree = parser.parse(bytes(code, 'utf8'))
    return tree


def remove_pragmas(code, language='cpp'):
    tree = parse_code(code, languages[language])
    root_node = tree.root_node
    code_lines = code.splitlines()
    
    def collect_pragmas(node):
        pragmas = []
        if node.type == 'preproc_call':
            start = node.start_byte
            end = node.end_byte
            pragmas.append((start, end))
        for child in node.children:
            pragmas.extend(collect_pragmas(child))
        return pragmas

    pragmas = collect_pragmas(root_node)

    # Remove the pragmas from the code
    pragmas.sort(key=lambda x: x[0], reverse=True)
    for start, end in pragmas:
        code = code[:start] + code[end:]

    return code


def remove_lines_with_text(code, text_to_remove):
    lines = code.splitlines()
    filtered_lines = [line for line in lines if text_to_remove not in line]
    return "\n".join(filtered_lines)


def remove_omp(code):
    code = remove_pragmas(code)
    return remove_lines_with_text(code, 'omp.h')



def remove_wrapping_braces(code, language='cpp'):
    tree = parse_code(code, languages[language])
    root_node = tree.root_node

    def is_wrapping_brace(node):
        if node.type == 'block' and len(node.children) == 1:
            child = node.children[0]
            if child.type in {'block', 'empty_statement'}:
                return True
        return False

    def collect_wrapping_braces(node):
        import pdb; pdb.set_trace()
        braces_to_remove = []
        if is_wrapping_brace(node):
            start = node.start_byte
            end = node.end_byte
            braces_to_remove.append((start, end))
        for child in node.children:
            braces_to_remove.extend(collect_wrapping_braces(child))
        return braces_to_remove

    braces_to_remove = collect_wrapping_braces(root_node)

    # Sort and remove wrapping braces from the code
    braces_to_remove.sort(key=lambda x: x[0], reverse=True)
    for start, end in braces_to_remove:
        code = code[:start] + code[end:]

    return code



if __name__=='__main__':

    code = """
        #pragma omp target teams distribute num_teams(ngrid)
        {
        for (int row = 0; row < nrows; row++) {
          const int label_data = label[row];
          const float label_pred = data[row * ndims + label_data];
          int ngt = 0;
          #pragma omp parallel for reduction(+:ngt) num_threads(NUM_THREADS)
            {
        for (int col = 0; col < ndims; col++) {
            const float pred = data[row * ndims + col];
            }
        }
        }
        }
"""

    code = """
# Your Python code with redundant braces
if (a > b) {
    print(a)
}
"""

    cleaned_code = remove_pragmas(code)
    print(remove_wrapping_braces(cleaned_code))

