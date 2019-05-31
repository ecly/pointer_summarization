"""
Copyright (c) 2019 Emil Lynegaard
Distributed under the MIT software license, see the
accompanying LICENSE.md or https://opensource.org/licenses/MIT

Convert GLoVe file for compatability with gensim
"""
import sys
from gensim.scripts.glove2word2vec import glove2word2vec


def main():
    glove_input_file = sys.argv[1]
    word2vec_output_file = sys.argv[2]
    glove2word2vec(glove_input_file, word2vec_output_file)


if __name__ == "__main__":
    main()
